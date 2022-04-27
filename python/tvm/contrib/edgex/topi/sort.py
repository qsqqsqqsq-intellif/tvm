# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# encoding=utf-8
# pylint: disable=line-too-long,invalid-name,too-many-nested-blocks,unused-variable,missing-function-docstring
"""sorting implementation on edgex"""
import math
import tvm
import numpy as np
from tvm.contrib.edgex.tir.transform.transform import InlinePrimFuncCalls
from tvm.ir.expr import Range
from tvm.ir.module import IRModule
from tvm.script import tir as T
from tvm.topi.utils import get_const_int, get_const_tuple
from tvm.contrib.edgex.tir.schedule import EdgexSchedule
from tvm import te
from tvm import topi
from tvm import tir


def wrap_fprimfunc_topk(fprimfunc):
    """Wrap topk fprimfunc"""

    def _wrap_func(attrs, inputs, _):
        if attrs.k is not None:
            k = attrs.k
        else:
            k = inputs[1]
        axis = get_const_int(attrs.axis)
        ret_type = attrs.ret_type
        is_ascend = bool(get_const_int(attrs.is_ascend))
        dtype = attrs.dtype
        return fprimfunc(inputs[0], k, axis, ret_type, is_ascend, dtype)

    return _wrap_func


@T.prim_func
def naive_bubble_sort_dm_i32(data: T.handle, indices: T.handle, K: T.int32, N: T.int32):
    Data = T.match_buffer(data, (N,), dtype="int32", scope="dm", offset_factor=1)
    Indices = T.match_buffer(indices, (N,), dtype="int32", scope="dm", offset_factor=1)
    Chunk: T.int32 = T.int32(4096)
    BubbleRounds: T.int32 = (N + (Chunk // 2) - 1) // (Chunk // 2) - 1
    for i in range((K + Chunk - 1) // Chunk):
        for j in range(BubbleRounds):
            offset: T.int32 = T.max(0, N - (j + 2) * (Chunk // 2))
            Data_vm = T.allocate([Chunk], "int32", "vm")
            Indices_vm = T.allocate([Chunk], "int32", "vm")
            for k in T.serial(0, Chunk, annotations={"pragma_nnp_dma_scope": "vidma"}):
                Data_vm[k] = Data[offset + k]
            for k in T.serial(0, Chunk, annotations={"pragma_nnp_dma_scope": "vidma"}):
                Indices_vm[k] = Indices[offset + k]
            T.evaluate(T.call_extern("vm_sort_impl", Data_vm.data, Indices_vm.data, dtype=""))
            for k in T.serial(0, Chunk, annotations={"pragma_nnp_dma_scope": "vodma"}):
                Data[k + offset] = Data_vm[k]
            for k in T.serial(0, Chunk, annotations={"pragma_nnp_dma_scope": "vodma"}):
                Indices[k + offset] = Indices_vm[k]


@T.prim_func
def naive_bubble_sort_dm_fp16(data: T.handle, indices: T.handle, K: T.int32, N: T.int32):
    Data = T.match_buffer(data, (N,), dtype="float16", scope="dm", offset_factor=1)
    Indices = T.match_buffer(indices, (N,), dtype="int32", scope="dm", offset_factor=1)
    Chunk: T.int32 = T.int32(8192)
    BubbleRounds: T.int32 = (N + (Chunk // 2) - 1) // (Chunk // 2) - 1
    for i in range((K + Chunk - 1) // Chunk):
        for j in range(BubbleRounds):
            offset: T.int32 = T.max(0, N - (j + 2) * (Chunk // 2))
            Data_vm = T.allocate([Chunk], "float16", "vm")
            Indices_vm = T.allocate([Chunk], "int32", "vm")
            for k in T.serial(0, Chunk, annotations={"pragma_nnp_dma_scope": "vidma"}):
                Data_vm[k] = Data[offset + k]
            for k in T.serial(0, Chunk, annotations={"pragma_nnp_dma_scope": "vidma"}):
                Indices_vm[k] = Indices[offset + k]
            T.evaluate(T.call_extern("vm_sort_impl", Data_vm.data, Indices_vm.data, dtype=""))
            for k in T.serial(0, Chunk, annotations={"pragma_nnp_dma_scope": "vodma"}):
                Data[k + offset] = Data_vm[k]
            for k in T.serial(0, Chunk, annotations={"pragma_nnp_dma_scope": "vodma"}):
                Indices[k + offset] = Indices_vm[k]


@T.prim_func
def bitonic_sort_non_recursive_i32(
    data: T.handle, indices: T.handle, N: T.int32, is_ascend: T.int32
):
    Data = T.match_buffer(data, (N,), dtype="int32", offset_factor=1)
    Indices = T.match_buffer(indices, (N,), dtype="int32", offset_factor=1)
    T.attr(None, "pragma_aliased_buffer_var", Data.data)
    T.attr(None, "pragma_aliased_buffer_var", Indices.data)
    ChunkSize: T.int32 = T.int32(16)
    TotalChunks: T.int32 = T.int32(N // ChunkSize)
    for i in range(TotalChunks // 4):
        offset: T.int32 = i * ChunkSize * 4
        sort: None = T.nnp_inline_asm_vcu(
            "={vv},={vv},={vv},={vv},={vv},={vv},={vv},={vv},0,1,2,3,4,5,6,7",
            """
                nop.10
                vsort.s32.as.op0 $1 $0
                vsort.s32.ds.op0 $3 $2
                vsort.s32.as.op0 $5 $4
                vsort.s32.ds.op0 $7 $6
                vsort.s32.as.op1 $1 $0
                vsort.s32.ds.op1 $3 $2
                vsort.s32.as.op1 $5 $4
                vsort.s32.ds.op1 $7 $6

                vsort.s32.as.op0 $1 $0
                vsort.s32.ds.op0 $3 $2
                vsort.s32.as.op0 $5 $4
                vsort.s32.ds.op0 $7 $6
                vsort.s32.as.op1 $1 $0
                vsort.s32.ds.op1 $3 $2
                vsort.s32.as.op1 $5 $4
                vsort.s32.ds.op1 $7 $6

                vsort.s32.as.op0 $1 $0
                vsort.s32.ds.op0 $3 $2
                vsort.s32.as.op0 $5 $4
                vsort.s32.ds.op0 $7 $6
                vsort.s32.as.op1 $1 $0
                vsort.s32.ds.op1 $3 $2
                vsort.s32.as.op1 $5 $4
                vsort.s32.ds.op1 $7 $6

                vsort.s32.as.op0 $1 $0
                vsort.s32.ds.op0 $3 $2
                vsort.s32.as.op0 $5 $4
                vsort.s32.ds.op0 $7 $6
                vsort.s32.as.op1 $1 $0
                vsort.s32.ds.op1 $3 $2
                vsort.s32.as.op1 $5 $4
                vsort.s32.ds.op1 $7 $6

                vsort.s32.as.op0 $1 $0
                vsort.s32.ds.op0 $3 $2
                vsort.s32.as.op0 $5 $4
                vsort.s32.ds.op0 $7 $6
                vsort.s32.as.op1 $1 $0
                vsort.s32.ds.op1 $3 $2
                vsort.s32.as.op1 $5 $4
                vsort.s32.ds.op1 $7 $6

                vsort.s32.as.op0 $1 $0
                vsort.s32.ds.op0 $3 $2
                vsort.s32.as.op0 $5 $4
                vsort.s32.ds.op0 $7 $6
                vsort.s32.as.op1 $1 $0
                vsort.s32.ds.op1 $3 $2
                vsort.s32.as.op1 $5 $4
                vsort.s32.ds.op1 $7 $6

                vsort.s32.as.op0 $1 $0
                vsort.s32.ds.op0 $3 $2
                vsort.s32.as.op0 $5 $4
                vsort.s32.ds.op0 $7 $6
                vsort.s32.as.op1 $1 $0
                vsort.s32.ds.op1 $3 $2
                vsort.s32.as.op1 $5 $4
                vsort.s32.ds.op1 $7 $6

                vsort.s32.as.op0 $1 $0
                vsort.s32.ds.op0 $3 $2
                vsort.s32.as.op0 $5 $4
                vsort.s32.ds.op0 $7 $6
                vsort.s32.as.op1 $1 $0
                vsort.s32.ds.op1 $3 $2
                vsort.s32.as.op1 $5 $4
                vsort.s32.ds.op1 $7 $6
                nop.10
            """,
            0,  # no vectorize factor
            0,  # no state regs
            8,  # 8 inputs
            Data[T.ramp(offset, 1, 16)],
            Indices[T.ramp(offset, 1, 16)],
            Data[T.ramp(offset + 16, 1, 16)],
            Indices[T.ramp(offset + 16, 1, 16)],
            Data[T.ramp(offset + 32, 1, 16)],
            Indices[T.ramp(offset + 32, 1, 16)],
            Data[T.ramp(offset + 48, 1, 16)],
            Indices[T.ramp(offset + 48, 1, 16)],
            0,  # no extra placeholders
            8,  # 8 output type annotation
            T.type_annotation(dtype="int32x16"),
            T.type_annotation(dtype="int32x16"),
            T.type_annotation(dtype="int32x16"),
            T.type_annotation(dtype="int32x16"),
            T.type_annotation(dtype="int32x16"),
            T.type_annotation(dtype="int32x16"),
            T.type_annotation(dtype="int32x16"),
            T.type_annotation(dtype="int32x16"),
            dtype="",
        )
        Data[T.ramp(offset, 1, 16)] = T.nnp_extract_field(sort, 0, dtype="int32x16")
        Indices[T.ramp(offset, 1, 16)] = T.nnp_extract_field(sort, 1, dtype="int32x16")
        Data[T.ramp(offset + 16, 1, 16)] = T.nnp_extract_field(sort, 2, dtype="int32x16")
        Indices[T.ramp(offset + 16, 1, 16)] = T.nnp_extract_field(sort, 3, dtype="int32x16")
        Data[T.ramp(offset + 32, 1, 16)] = T.nnp_extract_field(sort, 4, dtype="int32x16")
        Indices[T.ramp(offset + 32, 1, 16)] = T.nnp_extract_field(sort, 5, dtype="int32x16")
        Data[T.ramp(offset + 48, 1, 16)] = T.nnp_extract_field(sort, 6, dtype="int32x16")
        Indices[T.ramp(offset + 48, 1, 16)] = T.nnp_extract_field(sort, 7, dtype="int32x16")

    Chunks = T.allocate((), "int32", "vm")
    Chunks[()] = 2
    Groups = T.allocate((), "int32", "vm")
    Groups[()] = TotalChunks // 2
    MergeRounds = T.allocate((), "int32", "vm")
    MergeRounds[()] = 1
    Dir = T.allocate((), "int32", "vm")
    Dir[()] = is_ascend

    while Chunks[()] <= TotalChunks:
        for i in range(Groups[()]):
            Dir[()] = 1 - Dir[()]
            for merge_round in range(MergeRounds[()]):
                Parts: T.int32 = T.shift_left(1, merge_round, dtype="int32")
                for j in range(Parts):
                    ChunkPerPart: T.int32 = T.shift_right(Chunks[()], merge_round, dtype="int32")
                    for k in range(ChunkPerPart // 2):
                        l: T.int32 = (i * Chunks[()] + j * ChunkPerPart + k) * 16
                        r: T.int32 = (
                            i * Chunks[()] + j * ChunkPerPart + k + ChunkPerPart // 2
                        ) * 16
                        if Dir[()] == 0:
                            sort: None = T.nnp_inline_asm_vcu(
                                "={vv},={vv},={vv},={vv},0,1,2,3",
                                """
                                        nop.10
                                        vsort.data.s32.as.op2 $2.hi $0.hi
                                        nop.3
                                        vsort.index.s32.as.op2 $3.hi $1.hi
                                        nop.3
                                        vsort.data.s32.as.op2 $2.lo $0.lo
                                        nop.3
                                        vsort.index.s32.as.op2 $3.lo $1.lo
                                        nop.10
                                    """,
                                0,  # no vectorize factor
                                0,  # no state regs
                                4,  # 4 inputs
                                Data[T.ramp(l, 1, 16)],
                                Indices[T.ramp(l, 1, 16)],
                                Data[T.ramp(r, 1, 16)],
                                Indices[T.ramp(r, 1, 16)],
                                0,  # no extra placeholders
                                4,  # 4 output type annotation
                                T.type_annotation(dtype="int32x16"),
                                T.type_annotation(dtype="int32x16"),
                                T.type_annotation(dtype="int32x16"),
                                T.type_annotation(dtype="int32x16"),
                                dtype="",
                            )
                            Data[T.ramp(l, 1, 16)] = T.nnp_extract_field(sort, 0, dtype="int32x16")
                            Indices[T.ramp(l, 1, 16)] = T.nnp_extract_field(
                                sort, 1, dtype="int32x16"
                            )
                            Data[T.ramp(r, 1, 16)] = T.nnp_extract_field(sort, 2, dtype="int32x16")
                            Indices[T.ramp(r, 1, 16)] = T.nnp_extract_field(
                                sort, 3, dtype="int32x16"
                            )
                        else:
                            sort: None = T.nnp_inline_asm_vcu(
                                "={vv},={vv},={vv},={vv},0,1,2,3",
                                """
                                        nop.10
                                        vsort.data.s32.ds.op2 $2.hi $0.hi
                                        nop.3
                                        vsort.index.s32.ds.op2 $3.hi $1.hi
                                        nop.3
                                        vsort.data.s32.ds.op2 $2.lo $0.lo
                                        nop.3
                                        vsort.index.s32.ds.op2 $3.lo $1.lo
                                        nop.10
                                    """,
                                0,  # no vectorize factor
                                0,  # no state regs
                                4,  # 4 inputs
                                Data[T.ramp(l, 1, 16)],
                                Indices[T.ramp(l, 1, 16)],
                                Data[T.ramp(r, 1, 16)],
                                Indices[T.ramp(r, 1, 16)],
                                0,  # no extra placeholders
                                4,  # 4 output type annotation
                                T.type_annotation(dtype="int32x16"),
                                T.type_annotation(dtype="int32x16"),
                                T.type_annotation(dtype="int32x16"),
                                T.type_annotation(dtype="int32x16"),
                                dtype="",
                            )
                            Data[T.ramp(l, 1, 16)] = T.nnp_extract_field(sort, 0, dtype="int32x16")
                            Indices[T.ramp(l, 1, 16)] = T.nnp_extract_field(
                                sort, 1, dtype="int32x16"
                            )
                            Data[T.ramp(r, 1, 16)] = T.nnp_extract_field(sort, 2, dtype="int32x16")
                            Indices[T.ramp(r, 1, 16)] = T.nnp_extract_field(
                                sort, 3, dtype="int32x16"
                            )
            for j in range(Chunks[()]):
                offset: T.int32 = (i * Chunks[()] + j) * ChunkSize
                if Dir[()] == 0:
                    sort: None = T.nnp_inline_asm_vcu(
                        "={vv},={vv},0,1",
                        """
                            nop.10
                            vsort.s32.as.op0 $1 $0
                            nop.3
                            vsort.s32.as.op1 $1 $0
                            nop.3

                            vsort.s32.as.op0 $1 $0
                            nop.3
                            vsort.s32.as.op1 $1 $0
                            nop.3

                            vsort.s32.as.op0 $1 $0
                            nop.3
                            vsort.s32.as.op1 $1 $0
                            nop.3

                            vsort.s32.as.op0 $1 $0
                            nop.3
                            vsort.s32.as.op1 $1 $0
                            nop.3

                            vsort.s32.as.op0 $1 $0
                            nop.3
                            vsort.s32.as.op1 $1 $0
                            nop.3

                            vsort.s32.as.op0 $1 $0
                            nop.3
                            vsort.s32.as.op1 $1 $0
                            nop.3

                            vsort.s32.as.op0 $1 $0
                            nop.3
                            vsort.s32.as.op1 $1 $0
                            nop.3

                            vsort.s32.as.op0 $1 $0
                            nop.3
                            vsort.s32.as.op1 $1 $0
                            nop.10
                        """,
                        0,  # no vectorize factor
                        0,  # no state regs
                        2,  # 4 inputs
                        Data[T.ramp(offset, 1, 16)],
                        Indices[T.ramp(offset, 1, 16)],
                        0,  # no extra placeholders
                        2,  # 4 output type annotation
                        T.type_annotation(dtype="int32x16"),
                        T.type_annotation(dtype="int32x16"),
                        dtype="",
                    )
                    Data[T.ramp(offset, 1, 16)] = T.nnp_extract_field(sort, 0, dtype="int32x16")
                    Indices[T.ramp(offset, 1, 16)] = T.nnp_extract_field(sort, 1, dtype="int32x16")
                else:
                    sort: None = T.nnp_inline_asm_vcu(
                        "={vv},={vv},0,1",
                        """
                            nop.10
                            vsort.s32.ds.op0 $1 $0
                            nop.3
                            vsort.s32.ds.op1 $1 $0
                            nop.3

                            vsort.s32.ds.op0 $1 $0
                            nop.3
                            vsort.s32.ds.op1 $1 $0
                            nop.3

                            vsort.s32.ds.op0 $1 $0
                            nop.3
                            vsort.s32.ds.op1 $1 $0
                            nop.3

                            vsort.s32.ds.op0 $1 $0
                            nop.3
                            vsort.s32.ds.op1 $1 $0
                            nop.3

                            vsort.s32.ds.op0 $1 $0
                            nop.3
                            vsort.s32.ds.op1 $1 $0
                            nop.3

                            vsort.s32.ds.op0 $1 $0
                            nop.3
                            vsort.s32.ds.op1 $1 $0
                            nop.3

                            vsort.s32.ds.op0 $1 $0
                            nop.3
                            vsort.s32.ds.op1 $1 $0
                            nop.3

                            vsort.s32.ds.op0 $1 $0
                            nop.3
                            vsort.s32.ds.op1 $1 $0
                            nop.10
                        """,
                        0,  # no vectorize factor
                        0,  # no state regs
                        2,  # 4 inputs
                        Data[T.ramp(offset, 1, 16)],
                        Indices[T.ramp(offset, 1, 16)],
                        0,  # no extra placeholders
                        2,  # 4 output type annotation
                        T.type_annotation(dtype="int32x16"),
                        T.type_annotation(dtype="int32x16"),
                        dtype="",
                    )
                    Data[T.ramp(offset, 1, 16)] = T.nnp_extract_field(sort, 0, dtype="int32x16")
                    Indices[T.ramp(offset, 1, 16)] = T.nnp_extract_field(sort, 1, dtype="int32x16")

        Chunks[()] = Chunks[()] * 2
        Groups[()] = Groups[()] // 2
        MergeRounds[()] = MergeRounds[()] + 1


@T.prim_func
def bitonic_sort_non_recursive_fp16(
    data: T.handle, indices: T.handle, N: T.int32, is_ascend: T.int32
):
    Data = T.match_buffer(data, (N,), dtype="float16", offset_factor=1)
    Indices = T.match_buffer(indices, (N,), dtype="int32", offset_factor=1)
    T.attr(None, "pragma_aliased_buffer_var", Data.data)
    T.attr(None, "pragma_aliased_buffer_var", Indices.data)
    ChunkSize: T.int32 = T.int32(32)
    TotalChunks: T.int32 = T.int32(N // ChunkSize)
    for i in range(TotalChunks // 2):
        offset: T.int32 = i * ChunkSize * 2
        sort: None = T.nnp_inline_asm_vcu(
            "={vv},={vv},={vv},={vv},={vv},={vv},=&{vv},=&{vv},=&{vv},=&{vv},=&{vv},=&{vv},0,1,2,3,4,5",
            """
                nop.10
                vint.s32ts16.lo $6 $1 0
                vint.s32ts16.hi $6 $2 0
                vint.s32ts16.lo $8 $4 0
                vint.s32ts16.hi $8 $5 0
                vlsr.s32 $1 $1 16
                vlsr.s32 $2 $2 16
                vlsr.s32 $4 $4 16
                vlsr.s32 $5 $5 16
                vmov.f16 $10 $0
                vmov.f16 $11 $3
                nop.3
                vint.s32ts16.lo $7 $1 0
                vint.s32ts16.hi $7 $2 0
                vint.s32ts16.lo $9 $4 0
                vint.s32ts16.hi $9 $5 0
                nop.10

                vsort.f16.as.op0 $6 $0
                vsort.f16.as.op0 $7 $10
                vsort.f16.ds.op0 $8 $3
                vsort.f16.ds.op0 $9 $11
                vsort.f16.as.op1 $6 $0
                vsort.f16.as.op1 $7 $10
                vsort.f16.ds.op1 $8 $3
                vsort.f16.ds.op1 $9 $11

                vsort.f16.as.op0 $6 $0
                vsort.f16.as.op0 $7 $10
                vsort.f16.ds.op0 $8 $3
                vsort.f16.ds.op0 $9 $11
                vsort.f16.as.op1 $6 $0
                vsort.f16.as.op1 $7 $10
                vsort.f16.ds.op1 $8 $3
                vsort.f16.ds.op1 $9 $11

                vsort.f16.as.op0 $6 $0
                vsort.f16.as.op0 $7 $10
                vsort.f16.ds.op0 $8 $3
                vsort.f16.ds.op0 $9 $11
                vsort.f16.as.op1 $6 $0
                vsort.f16.as.op1 $7 $10
                vsort.f16.ds.op1 $8 $3
                vsort.f16.ds.op1 $9 $11

                vsort.f16.as.op0 $6 $0
                vsort.f16.as.op0 $7 $10
                vsort.f16.ds.op0 $8 $3
                vsort.f16.ds.op0 $9 $11
                vsort.f16.as.op1 $6 $0
                vsort.f16.as.op1 $7 $10
                vsort.f16.ds.op1 $8 $3
                vsort.f16.ds.op1 $9 $11

                vsort.f16.as.op0 $6 $0
                vsort.f16.as.op0 $7 $10
                vsort.f16.ds.op0 $8 $3
                vsort.f16.ds.op0 $9 $11
                vsort.f16.as.op1 $6 $0
                vsort.f16.as.op1 $7 $10
                vsort.f16.ds.op1 $8 $3
                vsort.f16.ds.op1 $9 $11

                vsort.f16.as.op0 $6 $0
                vsort.f16.as.op0 $7 $10
                vsort.f16.ds.op0 $8 $3
                vsort.f16.ds.op0 $9 $11
                vsort.f16.as.op1 $6 $0
                vsort.f16.as.op1 $7 $10
                vsort.f16.ds.op1 $8 $3
                vsort.f16.ds.op1 $9 $11

                vsort.f16.as.op0 $6 $0
                vsort.f16.as.op0 $7 $10
                vsort.f16.ds.op0 $8 $3
                vsort.f16.ds.op0 $9 $11
                vsort.f16.as.op1 $6 $0
                vsort.f16.as.op1 $7 $10
                vsort.f16.ds.op1 $8 $3
                vsort.f16.ds.op1 $9 $11

                vsort.f16.as.op0 $6 $0
                vsort.f16.as.op0 $7 $10
                vsort.f16.ds.op0 $8 $3
                vsort.f16.ds.op0 $9 $11
                vsort.f16.as.op1 $6 $0
                vsort.f16.as.op1 $7 $10
                vsort.f16.ds.op1 $8 $3
                vsort.f16.ds.op1 $9 $11

                vsort.f16.as.op0 $6 $0
                vsort.f16.as.op0 $7 $10
                vsort.f16.ds.op0 $8 $3
                vsort.f16.ds.op0 $9 $11
                vsort.f16.as.op1 $6 $0
                vsort.f16.as.op1 $7 $10
                vsort.f16.ds.op1 $8 $3
                vsort.f16.ds.op1 $9 $11

                vsort.f16.as.op0 $6 $0
                vsort.f16.as.op0 $7 $10
                vsort.f16.ds.op0 $8 $3
                vsort.f16.ds.op0 $9 $11
                vsort.f16.as.op1 $6 $0
                vsort.f16.as.op1 $7 $10
                vsort.f16.ds.op1 $8 $3
                vsort.f16.ds.op1 $9 $11

                vsort.f16.as.op0 $6 $0
                vsort.f16.as.op0 $7 $10
                vsort.f16.ds.op0 $8 $3
                vsort.f16.ds.op0 $9 $11
                vsort.f16.as.op1 $6 $0
                vsort.f16.as.op1 $7 $10
                vsort.f16.ds.op1 $8 $3
                vsort.f16.ds.op1 $9 $11

                vsort.f16.as.op0 $6 $0
                vsort.f16.as.op0 $7 $10
                vsort.f16.ds.op0 $8 $3
                vsort.f16.ds.op0 $9 $11
                vsort.f16.as.op1 $6 $0
                vsort.f16.as.op1 $7 $10
                vsort.f16.ds.op1 $8 $3
                vsort.f16.ds.op1 $9 $11

                vsort.f16.as.op0 $6 $0
                vsort.f16.as.op0 $7 $10
                vsort.f16.ds.op0 $8 $3
                vsort.f16.ds.op0 $9 $11
                vsort.f16.as.op1 $6 $0
                vsort.f16.as.op1 $7 $10
                vsort.f16.ds.op1 $8 $3
                vsort.f16.ds.op1 $9 $11

                vsort.f16.as.op0 $6 $0
                vsort.f16.as.op0 $7 $10
                vsort.f16.ds.op0 $8 $3
                vsort.f16.ds.op0 $9 $11
                vsort.f16.as.op1 $6 $0
                vsort.f16.as.op1 $7 $10
                vsort.f16.ds.op1 $8 $3
                vsort.f16.ds.op1 $9 $11

                vsort.f16.as.op0 $6 $0
                vsort.f16.as.op0 $7 $10
                vsort.f16.ds.op0 $8 $3
                vsort.f16.ds.op0 $9 $11
                vsort.f16.as.op1 $6 $0
                vsort.f16.as.op1 $7 $10
                vsort.f16.ds.op1 $8 $3
                vsort.f16.ds.op1 $9 $11

                vsort.f16.as.op0 $6 $0
                vsort.f16.as.op0 $7 $10
                vsort.f16.ds.op0 $8 $3
                vsort.f16.ds.op0 $9 $11
                vsort.f16.as.op1 $6 $0
                vsort.f16.as.op1 $7 $10
                vsort.f16.ds.op1 $8 $3
                vsort.f16.ds.op1 $9 $11

                vint.s16ts32.lo $1 $7 0
                vint.s16ts32.hi $2 $7 0
                vint.s16ts32.lo $10 $6 0
                vint.s16ts32.hi $11 $6 0
                nop.10
                vasl.s32 $1 $1 16
                vasl.s32 $2 $2 16
                nop.10
                vbit.s32.or $1 $1 $10
                vbit.s32.or $2 $2 $11
                nop.10
                vint.s16ts32.lo $4 $9 0
                vint.s16ts32.hi $5 $9 0
                vint.s16ts32.lo $10 $8 0
                vint.s16ts32.hi $11 $8 0
                nop.10
                vasl.s32 $4 $4 16
                vasl.s32 $5 $5 16
                nop.3
                vbit.s32.or $4 $4 $10
                vbit.s32.or $5 $5 $11
                nop.10
            """,
            0,  # no vectorize factor
            6,  # 6 state regs
            T.type_annotation(dtype="int32x16"),  # "int16x32"
            T.type_annotation(dtype="int32x16"),  # "int16x32"
            T.type_annotation(dtype="int32x16"),  # "int16x32"
            T.type_annotation(dtype="int32x16"),  # "int16x32"
            T.type_annotation(dtype="float16x32"),
            T.type_annotation(dtype="float16x32"),
            6,  # 6 inputs
            Data[T.ramp(offset, 1, 32)],
            Indices[T.ramp(offset, 1, 16)],
            Indices[T.ramp(offset + 16, 1, 16)],
            Data[T.ramp(offset + 32, 1, 32)],
            Indices[T.ramp(offset + 32, 1, 16)],
            Indices[T.ramp(offset + 48, 1, 16)],
            0,  # no extra placeholders
            6,  # 6 output type annotation
            T.type_annotation(dtype="float16x32"),
            T.type_annotation(dtype="int32x16"),
            T.type_annotation(dtype="int32x16"),
            T.type_annotation(dtype="float16x32"),
            T.type_annotation(dtype="int32x16"),
            T.type_annotation(dtype="int32x16"),
            dtype="",
        )
        Data[T.ramp(offset, 1, 32)] = T.nnp_extract_field(sort, 0, dtype="float16x32")
        Indices[T.ramp(offset, 1, 16)] = T.nnp_extract_field(sort, 1, dtype="int32x16")
        Indices[T.ramp(offset + 16, 1, 16)] = T.nnp_extract_field(sort, 2, dtype="int32x16")
        Data[T.ramp(offset + 32, 1, 32)] = T.nnp_extract_field(sort, 3, dtype="float16x32")
        Indices[T.ramp(offset + 32, 1, 16)] = T.nnp_extract_field(sort, 4, dtype="int32x16")
        Indices[T.ramp(offset + 48, 1, 16)] = T.nnp_extract_field(sort, 5, dtype="int32x16")

    Chunks = T.allocate((), "int32", "vm")
    Chunks[()] = 2
    Groups = T.allocate((), "int32", "vm")
    Groups[()] = TotalChunks // 2
    MergeRounds = T.allocate((), "int32", "vm")
    MergeRounds[()] = 1
    Dir = T.allocate((), "int32", "vm")
    Dir[()] = is_ascend
    while Chunks[()] <= TotalChunks:
        for i in range(Groups[()]):
            Dir[()] = 1 - Dir[()]
            for merge_round in range(MergeRounds[()]):
                Parts: T.int32 = T.shift_left(1, merge_round, dtype="int32")
                for j in range(Parts):
                    ChunkPerPart: T.int32 = T.shift_right(Chunks[()], merge_round, dtype="int32")
                    for k in range(ChunkPerPart // 2):
                        l: T.int32 = (i * Chunks[()] + j * ChunkPerPart + k) * ChunkSize
                        r: T.int32 = (
                            i * Chunks[()] + j * ChunkPerPart + k + ChunkPerPart // 2
                        ) * ChunkSize
                        if Dir[()] == 0:
                            sort: None = T.nnp_inline_asm_vcu(
                                "={vv},={vv},={vv},={vv},={vv},={vv},=&{vv},=&{vv},=&{vv},=&{vv},0,1,2,3,4,5",
                                """
                                    nop.10
                                    vint.s32ts16.lo $6 $1 0
                                    vint.s32ts16.hi $6 $2 0
                                    vint.s32ts16.lo $8 $4 0
                                    vint.s32ts16.hi $8 $5 0
                                    vlsr.s32 $1 $1 16
                                    vlsr.s32 $2 $2 16
                                    vlsr.s32 $4 $4 16
                                    vlsr.s32 $5 $5 16
                                    nop.3
                                    vint.s32ts16.lo $7 $1 0
                                    vint.s32ts16.hi $7 $2 0
                                    vint.s32ts16.lo $9 $4 0
                                    vint.s32ts16.hi $9 $5 0

                                    nop.10
                                    vsort.data.f16.as.op2 $3.hi $0.hi
                                    nop.3
                                    vsort.index.f16.as.op2 $8.hi $6.hi
                                    vsort.index.f16.as.op2 $9.hi $7.hi
                                    nop.3
                                    vsort.data.f16.as.op2 $3.lo $0.lo
                                    nop.3
                                    vsort.index.f16.as.op2 $8.lo $6.lo
                                    vsort.index.f16.as.op2 $9.lo $7.lo
                                    nop.3

                                    vint.s16ts32.lo $1 $7 0
                                    vint.s16ts32.hi $2 $7 0
                                    vint.s16ts32.lo $4 $9 0
                                    vint.s16ts32.hi $5 $9 0
                                    vint.s16ts32.lo $7 $6 0
                                    vint.s16ts32.hi $9 $6 0
                                    nop.3
                                    vasl.s32 $1 $1 16
                                    vasl.s32 $2 $2 16
                                    vasl.s32 $4 $4 16
                                    vasl.s32 $5 $5 16
                                    nop.4
                                    vbit.s32.or $1 $1 $7
                                    vbit.s32.or $2 $2 $9
                                    vint.s16ts32.lo $7 $8 0
                                    vint.s16ts32.hi $9 $8 0
                                    nop.10
                                    vbit.s32.or $4 $4 $7
                                    vbit.s32.or $5 $5 $9
                                    nop.10
                                    """,
                                0,  # no vectorize factor
                                4,  # 4 state regs
                                T.type_annotation(dtype="int32x16"),  # "int16x32"
                                T.type_annotation(dtype="int32x16"),  # "int16x32"
                                T.type_annotation(dtype="int32x16"),  # "int16x32"
                                T.type_annotation(dtype="int32x16"),  # "int16x32"
                                6,  # 6 inputs
                                Data[T.ramp(l, 1, 32)],
                                Indices[T.ramp(l, 1, 16)],
                                Indices[T.ramp(l + 16, 1, 16)],
                                Data[T.ramp(r, 1, 32)],
                                Indices[T.ramp(r, 1, 16)],
                                Indices[T.ramp(r + 16, 1, 16)],
                                0,  # no extra placeholders
                                6,  # 6 output type annotation
                                T.type_annotation(dtype="float16x32"),
                                T.type_annotation(dtype="int32x16"),
                                T.type_annotation(dtype="int32x16"),
                                T.type_annotation(dtype="float16x32"),
                                T.type_annotation(dtype="int32x16"),
                                T.type_annotation(dtype="int32x16"),
                                dtype="",
                            )
                            Data[T.ramp(l, 1, 32)] = T.nnp_extract_field(
                                sort, 0, dtype="float16x32"
                            )
                            Indices[T.ramp(l, 1, 16)] = T.nnp_extract_field(
                                sort, 1, dtype="int32x16"
                            )
                            Indices[T.ramp(l + 16, 1, 16)] = T.nnp_extract_field(
                                sort, 2, dtype="int32x16"
                            )
                            Data[T.ramp(r, 1, 32)] = T.nnp_extract_field(
                                sort, 3, dtype="float16x32"
                            )
                            Indices[T.ramp(r, 1, 16)] = T.nnp_extract_field(
                                sort, 4, dtype="int32x16"
                            )
                            Indices[T.ramp(r + 16, 1, 16)] = T.nnp_extract_field(
                                sort, 5, dtype="int32x16"
                            )
                        else:
                            sort: None = T.nnp_inline_asm_vcu(
                                "={vv},={vv},={vv},={vv},={vv},={vv},=&{vv},=&{vv},=&{vv},=&{vv},0,1,2,3,4,5",
                                """
                                    nop.10
                                    vint.s32ts16.lo $6 $1 0
                                    vint.s32ts16.hi $6 $2 0
                                    vint.s32ts16.lo $8 $4 0
                                    vint.s32ts16.hi $8 $5 0
                                    vlsr.s32 $1 $1 16
                                    vlsr.s32 $2 $2 16
                                    vlsr.s32 $4 $4 16
                                    vlsr.s32 $5 $5 16
                                    nop.3
                                    vint.s32ts16.lo $7 $1 0
                                    vint.s32ts16.hi $7 $2 0
                                    vint.s32ts16.lo $9 $4 0
                                    vint.s32ts16.hi $9 $5 0

                                    nop.10
                                    vsort.data.f16.ds.op2 $3.hi $0.hi
                                    nop.3
                                    vsort.index.f16.ds.op2 $8.hi $6.hi
                                    vsort.index.f16.ds.op2 $9.hi $7.hi
                                    nop.3
                                    vsort.data.f16.ds.op2 $3.lo $0.lo
                                    nop.3
                                    vsort.index.f16.ds.op2 $8.lo $6.lo
                                    vsort.index.f16.ds.op2 $9.lo $7.lo
                                    nop.3

                                    vint.s16ts32.lo $1 $7 0
                                    vint.s16ts32.hi $2 $7 0
                                    vint.s16ts32.lo $4 $9 0
                                    vint.s16ts32.hi $5 $9 0
                                    vint.s16ts32.lo $7 $6 0
                                    vint.s16ts32.hi $9 $6 0
                                    nop.3
                                    vasl.s32 $1 $1 16
                                    vasl.s32 $2 $2 16
                                    vasl.s32 $4 $4 16
                                    vasl.s32 $5 $5 16
                                    nop.4
                                    vbit.s32.or $1 $1 $7
                                    vbit.s32.or $2 $2 $9
                                    vint.s16ts32.lo $7 $8 0
                                    vint.s16ts32.hi $9 $8 0
                                    nop.10
                                    vbit.s32.or $4 $4 $7
                                    vbit.s32.or $5 $5 $9
                                    nop.10
                                    """,
                                0,  # no vectorize factor
                                4,  # 4 state regs
                                T.type_annotation(dtype="int32x16"),  # "int16x32"
                                T.type_annotation(dtype="int32x16"),  # "int16x32"
                                T.type_annotation(dtype="int32x16"),  # "int16x32"
                                T.type_annotation(dtype="int32x16"),  # "int16x32"
                                6,  # 6 inputs
                                Data[T.ramp(l, 1, 32)],
                                Indices[T.ramp(l, 1, 16)],
                                Indices[T.ramp(l + 16, 1, 16)],
                                Data[T.ramp(r, 1, 32)],
                                Indices[T.ramp(r, 1, 16)],
                                Indices[T.ramp(r + 16, 1, 16)],
                                0,  # no extra placeholders
                                6,  # 6 output type annotation
                                T.type_annotation(dtype="float16x32"),
                                T.type_annotation(dtype="int32x16"),
                                T.type_annotation(dtype="int32x16"),
                                T.type_annotation(dtype="float16x32"),
                                T.type_annotation(dtype="int32x16"),
                                T.type_annotation(dtype="int32x16"),
                                dtype="",
                            )
                            Data[T.ramp(l, 1, 32)] = T.nnp_extract_field(
                                sort, 0, dtype="float16x32"
                            )
                            Indices[T.ramp(l, 1, 16)] = T.nnp_extract_field(
                                sort, 1, dtype="int32x16"
                            )
                            Indices[T.ramp(l + 16, 1, 16)] = T.nnp_extract_field(
                                sort, 2, dtype="int32x16"
                            )
                            Data[T.ramp(r, 1, 32)] = T.nnp_extract_field(
                                sort, 3, dtype="float16x32"
                            )
                            Indices[T.ramp(r, 1, 16)] = T.nnp_extract_field(
                                sort, 4, dtype="int32x16"
                            )
                            Indices[T.ramp(r + 16, 1, 16)] = T.nnp_extract_field(
                                sort, 5, dtype="int32x16"
                            )

            for j in range(Chunks[()]):
                offset: T.int32 = (i * Chunks[()] + j) * ChunkSize
                if Dir[()] == 0:
                    sort: None = T.nnp_inline_asm_vcu(
                        "={vv},={vv},={vv},=&{vv},=&{vv},=&{vv},0,1,2",
                        """
                            nop.10
                            vint.s32ts16.lo $3 $1 0
                            vint.s32ts16.hi $3 $2 0
                            vlsr.s32 $1 $1 16
                            vlsr.s32 $2 $2 16
                            vmov.f16 $5 $0
                            nop.4
                            vint.s32ts16.lo $4 $1 0
                            vint.s32ts16.hi $4 $2 0
                            nop.10
                            vsort.f16.as.op0 $3 $0
                            vsort.f16.as.op0 $4 $5
                            nop.2
                            vsort.f16.as.op1 $3 $0
                            vsort.f16.as.op1 $4 $5
                            nop.2
                            vsort.f16.as.op0 $3 $0
                            vsort.f16.as.op0 $4 $5
                            nop.2
                            vsort.f16.as.op1 $3 $0
                            vsort.f16.as.op1 $4 $5
                            nop.2
                            vsort.f16.as.op0 $3 $0
                            vsort.f16.as.op0 $4 $5
                            nop.2
                            vsort.f16.as.op1 $3 $0
                            vsort.f16.as.op1 $4 $5
                            nop.2
                            vsort.f16.as.op0 $3 $0
                            vsort.f16.as.op0 $4 $5
                            nop.2
                            vsort.f16.as.op1 $3 $0
                            vsort.f16.as.op1 $4 $5
                            nop.2
                            vsort.f16.as.op0 $3 $0
                            vsort.f16.as.op0 $4 $5
                            nop.2
                            vsort.f16.as.op1 $3 $0
                            vsort.f16.as.op1 $4 $5
                            nop.2
                            vsort.f16.as.op0 $3 $0
                            vsort.f16.as.op0 $4 $5
                            nop.2
                            vsort.f16.as.op1 $3 $0
                            vsort.f16.as.op1 $4 $5
                            nop.2
                            vsort.f16.as.op0 $3 $0
                            vsort.f16.as.op0 $4 $5
                            nop.2
                            vsort.f16.as.op1 $3 $0
                            vsort.f16.as.op1 $4 $5
                            nop.2
                            vsort.f16.as.op0 $3 $0
                            vsort.f16.as.op0 $4 $5
                            nop.2
                            vsort.f16.as.op1 $3 $0
                            vsort.f16.as.op1 $4 $5
                            nop.2
                            vsort.f16.as.op0 $3 $0
                            vsort.f16.as.op0 $4 $5
                            nop.2
                            vsort.f16.as.op1 $3 $0
                            vsort.f16.as.op1 $4 $5
                            nop.2
                            vsort.f16.as.op0 $3 $0
                            vsort.f16.as.op0 $4 $5
                            nop.2
                            vsort.f16.as.op1 $3 $0
                            vsort.f16.as.op1 $4 $5
                            nop.2
                            vsort.f16.as.op0 $3 $0
                            vsort.f16.as.op0 $4 $5
                            nop.2
                            vsort.f16.as.op1 $3 $0
                            vsort.f16.as.op1 $4 $5
                            nop.2
                            vsort.f16.as.op0 $3 $0
                            vsort.f16.as.op0 $4 $5
                            nop.2
                            vsort.f16.as.op1 $3 $0
                            vsort.f16.as.op1 $4 $5
                            nop.2
                            vsort.f16.as.op0 $3 $0
                            vsort.f16.as.op0 $4 $5
                            nop.2
                            vsort.f16.as.op1 $3 $0
                            vsort.f16.as.op1 $4 $5
                            nop.2
                            vsort.f16.as.op0 $3 $0
                            vsort.f16.as.op0 $4 $5
                            nop.2
                            vsort.f16.as.op1 $3 $0
                            vsort.f16.as.op1 $4 $5
                            nop.2
                            vsort.f16.as.op0 $3 $0
                            vsort.f16.as.op0 $4 $5
                            nop.2
                            vsort.f16.as.op1 $3 $0
                            vsort.f16.as.op1 $4 $5
                            nop.2
                            vsort.f16.as.op0 $3 $0
                            vsort.f16.as.op0 $4 $5
                            nop.2
                            vsort.f16.as.op1 $3 $0
                            vsort.f16.as.op1 $4 $5
                            nop.3

                            vint.s16ts32.lo $1 $4 0
                            vint.s16ts32.hi $2 $4 0
                            vint.s16ts32.lo $3 $3 0
                            vint.s16ts32.hi $5 $3 0
                            nop.4
                            vasl.s32 $1 $1 16
                            vasl.s32 $2 $2 16
                            nop.6
                            vbit.s32.or $1 $1 $3
                            vbit.s32.or $2 $2 $5
                            nop.10
                        """,
                        0,  # no vectorize factor
                        3,  # no state regs
                        T.type_annotation(dtype="float16x32"),
                        T.type_annotation(dtype="int32x16"),
                        T.type_annotation(dtype="int32x16"),
                        3,  # 3 inputs
                        Data[T.ramp(offset, 1, 32)],
                        Indices[T.ramp(offset, 1, 16)],
                        Indices[T.ramp(offset + 16, 1, 16)],
                        0,  # no extra placeholders
                        3,  # 3 output type annotation
                        T.type_annotation(dtype="float16x32"),
                        T.type_annotation(dtype="int32x16"),
                        T.type_annotation(dtype="int32x16"),
                        dtype="",
                    )
                    Data[T.ramp(offset, 1, 32)] = T.nnp_extract_field(sort, 0, dtype="float16x32")
                    Indices[T.ramp(offset, 1, 16)] = T.nnp_extract_field(sort, 1, dtype="int32x16")
                    Indices[T.ramp(offset + 16, 1, 16)] = T.nnp_extract_field(
                        sort, 2, dtype="int32x16"
                    )
                else:
                    sort: None = T.nnp_inline_asm_vcu(
                        "={vv},={vv},={vv},=&{vv},=&{vv},=&{vv},0,1,2",
                        """
                            nop.10
                            vint.s32ts16.lo $3 $1 0
                            vint.s32ts16.hi $3 $2 0
                            vlsr.s32 $1 $1 16
                            vlsr.s32 $2 $2 16
                            vmov.f16 $5 $0
                            nop.4
                            vint.s32ts16.lo $4 $1 0
                            vint.s32ts16.hi $4 $2 0
                            nop.10

                            vsort.f16.ds.op0 $3 $0
                            vsort.f16.ds.op0 $4 $5
                            nop.2
                            vsort.f16.ds.op1 $3 $0
                            vsort.f16.ds.op1 $4 $5
                            nop.2
                            vsort.f16.ds.op0 $3 $0
                            vsort.f16.ds.op0 $4 $5
                            nop.2
                            vsort.f16.ds.op1 $3 $0
                            vsort.f16.ds.op1 $4 $5
                            nop.2
                            vsort.f16.ds.op0 $3 $0
                            vsort.f16.ds.op0 $4 $5
                            nop.2
                            vsort.f16.ds.op1 $3 $0
                            vsort.f16.ds.op1 $4 $5
                            nop.2
                            vsort.f16.ds.op0 $3 $0
                            vsort.f16.ds.op0 $4 $5
                            nop.2
                            vsort.f16.ds.op1 $3 $0
                            vsort.f16.ds.op1 $4 $5
                            nop.2
                            vsort.f16.ds.op0 $3 $0
                            vsort.f16.ds.op0 $4 $5
                            nop.2
                            vsort.f16.ds.op1 $3 $0
                            vsort.f16.ds.op1 $4 $5
                            nop.2
                            vsort.f16.ds.op0 $3 $0
                            vsort.f16.ds.op0 $4 $5
                            nop.2
                            vsort.f16.ds.op1 $3 $0
                            vsort.f16.ds.op1 $4 $5
                            nop.2
                            vsort.f16.ds.op0 $3 $0
                            vsort.f16.ds.op0 $4 $5
                            nop.2
                            vsort.f16.ds.op1 $3 $0
                            vsort.f16.ds.op1 $4 $5
                            nop.2
                            vsort.f16.ds.op0 $3 $0
                            vsort.f16.ds.op0 $4 $5
                            nop.2
                            vsort.f16.ds.op1 $3 $0
                            vsort.f16.ds.op1 $4 $5
                            nop.2
                            vsort.f16.ds.op0 $3 $0
                            vsort.f16.ds.op0 $4 $5
                            nop.2
                            vsort.f16.ds.op1 $3 $0
                            vsort.f16.ds.op1 $4 $5
                            nop.2
                            vsort.f16.ds.op0 $3 $0
                            vsort.f16.ds.op0 $4 $5
                            nop.2
                            vsort.f16.ds.op1 $3 $0
                            vsort.f16.ds.op1 $4 $5
                            nop.2
                            vsort.f16.ds.op0 $3 $0
                            vsort.f16.ds.op0 $4 $5
                            nop.2
                            vsort.f16.ds.op1 $3 $0
                            vsort.f16.ds.op1 $4 $5
                            nop.2
                            vsort.f16.ds.op0 $3 $0
                            vsort.f16.ds.op0 $4 $5
                            nop.2
                            vsort.f16.ds.op1 $3 $0
                            vsort.f16.ds.op1 $4 $5
                            nop.2
                            vsort.f16.ds.op0 $3 $0
                            vsort.f16.ds.op0 $4 $5
                            nop.2
                            vsort.f16.ds.op1 $3 $0
                            vsort.f16.ds.op1 $4 $5
                            nop.2
                            vsort.f16.ds.op0 $3 $0
                            vsort.f16.ds.op0 $4 $5
                            nop.2
                            vsort.f16.ds.op1 $3 $0
                            vsort.f16.ds.op1 $4 $5
                            nop.2
                            vsort.f16.ds.op0 $3 $0
                            vsort.f16.ds.op0 $4 $5
                            nop.2
                            vsort.f16.ds.op1 $3 $0
                            vsort.f16.ds.op1 $4 $5
                            nop.2
                            vsort.f16.ds.op0 $3 $0
                            vsort.f16.ds.op0 $4 $5
                            nop.2
                            vsort.f16.ds.op1 $3 $0
                            vsort.f16.ds.op1 $4 $5
                            nop.3

                            vint.s16ts32.lo $1 $4 0
                            vint.s16ts32.hi $2 $4 0
                            vint.s16ts32.lo $3 $3 0
                            vint.s16ts32.hi $5 $3 0
                            nop.4
                            vasl.s32 $1 $1 16
                            vasl.s32 $2 $2 16
                            nop.6
                            vbit.s32.or $1 $1 $3
                            vbit.s32.or $2 $2 $5
                            nop.10
                        """,
                        0,  # no vectorize factor
                        3,  # no state regs
                        T.type_annotation(dtype="float16x32"),
                        T.type_annotation(dtype="int32x16"),
                        T.type_annotation(dtype="int32x16"),
                        3,  # 3 inputs
                        Data[T.ramp(offset, 1, 32)],
                        Indices[T.ramp(offset, 1, 16)],
                        Indices[T.ramp(offset + 16, 1, 16)],
                        0,  # no extra placeholders
                        3,  # 3 output type annotation
                        T.type_annotation(dtype="float16x32"),
                        T.type_annotation(dtype="int32x16"),
                        T.type_annotation(dtype="int32x16"),
                        dtype="",
                    )
                    Data[T.ramp(offset, 1, 32)] = T.nnp_extract_field(sort, 0, dtype="float16x32")
                    Indices[T.ramp(offset, 1, 16)] = T.nnp_extract_field(sort, 1, dtype="int32x16")
                    Indices[T.ramp(offset + 16, 1, 16)] = T.nnp_extract_field(
                        sort, 2, dtype="int32x16"
                    )

        Chunks[()] = Chunks[()] * 2
        Groups[()] = Groups[()] // 2
        MergeRounds[()] = MergeRounds[()] + 1


def sort_fprimfunc_naive_impl(
    data, k=-1, axis=-1, ret_type="both", is_ascend=False, indice_dtype="int32"
):
    """Get the top k elements in an input tensor along the given axis.

    Parameters
    ----------
    data : tvm.te.Tensor
        The input tensor.

    k : int or tvm.te.Tensor, optional
        Number of top elements to select. Return all elements if k < 1.

    axis : int, optional
        Axis long which to sort the input tensor.

    ret_type: str, optional
        The return type [both, values, indices].
        "both": return both top k data and indices.
        "values": return top k data only.
        "indices": return top k indices only.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    indice_dtype : string, optional
        The data type of the indices output.

    Returns
    -------
    primfunc : tir.PrimFunc
    """
    assert ret_type in ["both", "values", "indices"]
    assert indice_dtype == "int32"
    ndim = len(data.shape)
    if axis < 0:
        axis = ndim + axis
    if axis < 0 or axis >= ndim:
        raise ValueError(f"Illegal topk axis {axis}")
    out_shape = list(get_const_tuple(data.shape))
    kvar = tvm.te.size_var("k")
    if not isinstance(k, (int, tvm.tir.IntImm)):
        out_shape[axis] = kvar
    elif k >= 1:
        out_shape[axis] = k

    # transpose sort axis to innermost
    input_data = data
    rev_indices = list(range(ndim))
    rev_indices[axis] = ndim - 1
    rev_indices[ndim - 1] = axis
    data = topi.transpose(data, [i for i in range(ndim) if i != axis] + [axis])

    # pad to power of 2 if N is small enough
    N = int(data.shape[-1])
    has_padding = False
    vm_thread_hold = 64 * 1024 - 4096
    total_data_bytes = N * (4 + tvm.DataType(data.dtype).bits // 8)
    if total_data_bytes < vm_thread_hold:
        if N < 64:
            N = 63
        if N & (N - 1) != 0:
            N = 1 << (int(math.log2(N - 1)) + 1)
            dtype_info = (
                np.iinfo(data.dtype) if data.dtype.startswith("int") else np.finfo(data.dtype)
            )
            pad_value = dtype_info.max if is_ascend else dtype_info.min
            data = topi.nn.pad(data, [0] * ndim, [0] * (ndim - 1) + [N - data.shape[-1]], pad_value)
            has_padding = True

    # determine sort implementation
    if data.dtype == "int32":
        dm_sort_impl = naive_bubble_sort_dm_i32
        vm_sort_impl = bitonic_sort_non_recursive_i32
    elif data.dtype == "float16":
        dm_sort_impl = naive_bubble_sort_dm_fp16
        vm_sort_impl = bitonic_sort_non_recursive_fp16
    else:
        raise ValueError("Do not support sort dtype " + data.dtype)
    total_data_bytes = N * (4 + tvm.DataType(data.dtype).bits // 8)
    if total_data_bytes >= vm_thread_hold:
        sort_impl = dm_sort_impl
        sort_impl = sort_impl.specialize({sort_impl.params[2]: k, sort_impl.params[3]: N})
        vm_sort_impl = vm_sort_impl.specialize(
            {
                vm_sort_impl.params[2]: 4096 if data.dtype == "int32" else 8192,
                vm_sort_impl.params[3]: 1 if is_ascend else 0,
            }
        )
        sort_impl = InlinePrimFuncCalls({"vm_sort_impl": vm_sort_impl})(
            IRModule.from_expr(sort_impl)
        )["main"]
        intrin_storage_level = "dm"
    else:
        sort_impl = vm_sort_impl.specialize(
            {vm_sort_impl.params[2]: N, vm_sort_impl.params[3]: 1 if is_ascend else 0}
        )
        intrin_storage_level = "vm"

    # create primfunc return values and indices
    index_seq = te.compute(data.shape, lambda *indices: indices[ndim - 1], name="indices")
    sort = te.compute(
        index_seq.shape,
        lambda *indices: (te.TensorSlice(data, indices), te.TensorSlice(index_seq, indices)),
        name="sort",
    )
    if k <= 0:
        k = data.shape[-1]

    def fetch_output(res, indices):
        return te.TensorSlice(res, tuple([indices[rev_indices[i]] for i in range(ndim)]))

    sort_values = te.compute(
        out_shape, lambda *indices: fetch_output(sort[0], indices), "fetch_topk_values"
    )
    sort_indices = te.compute(
        out_shape, lambda *indices: fetch_output(sort[1], indices), "fetch_topk_indices"
    )
    if ret_type == "both":
        primfunc = te.create_prim_func([input_data, sort_values, sort_indices])
    elif ret_type == "values":
        primfunc = te.create_prim_func([input_data, sort_values])
    else:
        primfunc = te.create_prim_func([input_data, sort_indices])

    # basic schedule to tensorize template sort computations
    s = EdgexSchedule(primfunc)
    values_block = s.get_block("sort.v0")
    indices_block = s.get_block("sort.v1")
    init_indices_block = s.get_block("indices")
    if ndim > 1:
        if has_padding:
            s.compute_at(s.get_block("PadInput"), s.get_loops(values_block)[-2])
        s.compute_at(s.get_block("T_transpose"), s.get_loops(values_block)[-2])
        s.compute_at(init_indices_block, s.get_loops(indices_block)[-2])
        if ret_type in ["both", "values"]:
            s.reverse_compute_at(s.get_block("fetch_topk_values"), s.get_loops(values_block)[-2])
        if ret_type in ["both", "indices"]:
            s.reverse_compute_at(s.get_block("fetch_topk_indices"), s.get_loops(indices_block)[-2])

    # schedule indices, we should not directly vectorize (0, N) seq for llvm build ir cost
    # also note currently we have to do it before inplace
    if intrin_storage_level == "dm":
        indices_dm = s.cache_write(init_indices_block, 0, "vm")
        s.set_scope(indices_dm, 0, "dm")
        v_o, v_i = s.split(s.get_loops(init_indices_block)[-1], factors=[None, 1024])
        s.reverse_compute_at(indices_dm, v_o)

        v_io, v_ii = s.split(v_i, factors=[None, 64])
        s.vectorize(v_ii)
        s.loop_partition([v_o, v_io, v_ii, s.get_loops(indices_dm)[-1]])
        s.pragma(s.get_loops(indices_dm)[-1], "nnp_dma_scope", "vodma")
        s.vectorize(v_ii)
    else:
        s.set_scope(init_indices_block, 0, "vm")
        _, v_i = s.split(s.get_loops(init_indices_block)[-1], factors=[None, 64])
        s.vectorize(v_i)

    # schedule padding
    if has_padding:
        pad_block = s.get_block("PadInput")
        s.loop_partition(s.get_loops(pad_block)[-1])
        s.vectorize(s.get_loops(pad_block)[-1])

    s.inplace_buffer(values_block, 0, 0, unsafe=True)
    s.inplace_buffer(indices_block, 0, 0, unsafe=True)
    s.set_scope(values_block, 0, intrin_storage_level)
    s.set_scope(indices_block, 0, intrin_storage_level)

    # match region for data
    data_1d = tir.decl_buffer([N], data.dtype, "data", offset_factor=1, scope=intrin_storage_level)
    data_buffer = s.get_sref(values_block).stmt.reads[0].buffer
    regions = []
    for i in range(ndim - 1):
        regions.append(
            Range.from_min_extent(s.get_sref(s.get_loops(values_block)[i]).stmt.loop_var, 1)
        )
    regions.append(Range.from_min_extent(0, data_buffer.shape[-1]))
    data_region = tir.BufferRegion(data_buffer, regions)

    # match region for indices
    indices_1d = tir.decl_buffer(
        [N], "int32", "indices", offset_factor=1, scope=intrin_storage_level
    )
    indices_buffer = s.get_sref(indices_block).stmt.reads[0].buffer
    regions = []
    for i in range(ndim - 1):
        regions.append(
            Range.from_min_extent(s.get_sref(s.get_loops(indices_block)[i]).stmt.loop_var, 1)
        )
    regions.append(Range.from_min_extent(0, indices_buffer.shape[-1]))
    indices_region = tir.BufferRegion(indices_buffer, regions)

    s.state.replace(
        s.get_sref(s.get_loops(values_block)[-1]),
        tir.BlockRealize(
            [],
            True,
            tir.Block(
                [],
                [data_region, indices_region],
                [data_region, indices_region],
                "sort",
                tir.Evaluate(tir.call_extern("", "sort_impl", data_1d.data, indices_1d.data)),
                match_buffers=[
                    tir.MatchBufferRegion(data_1d, data_region),
                    tir.MatchBufferRegion(indices_1d, indices_region),
                ],
            ),
        ),
    )

    # schedule input/output dma
    input_block = s.get_block("T_transpose")
    if intrin_storage_level == "vm":
        blk = s.cache_read(input_block, 0, "dm")
        if ndim > 1:
            s.compute_at(blk, s.get_loops(input_block)[-2])
        s.set_scope(input_block, 0, "vm")
        s.pragma(s.get_loops(input_block)[-1], "nnp_dma_scope", "vidma")
        s.pragma(s.get_loops(blk)[-1], "nnp_dma_scope", "eidma")
    else:
        s.pragma(s.get_loops(input_block)[0], "nnp_dma_scope", "eidma")

    if ret_type in {"both", "indices"}:
        fetch_indices_block = s.get_block("fetch_topk_indices")
        if intrin_storage_level == "vm":
            blk = s.cache_write(fetch_indices_block, 0, "dm")
            if ndim > 1:
                s.reverse_compute_at(blk, s.get_loops(fetch_indices_block)[-2])
            s.pragma(s.get_loops(fetch_indices_block)[-1], "nnp_dma_scope", "vodma")
            s.pragma(s.get_loops(blk)[-1], "nnp_dma_scope", "eodma")
        else:
            s.pragma(s.get_loops(fetch_indices_block)[-1], "nnp_dma_scope", "eodma")

    if ret_type in {"both", "values"}:
        fetch_values_block = s.get_block("fetch_topk_values")
        if intrin_storage_level == "vm":
            blk = s.cache_write(fetch_values_block, 0, "dm")
            if ndim > 1:
                s.reverse_compute_at(blk, s.get_loops(fetch_values_block)[-2])
            s.pragma(s.get_loops(fetch_values_block)[-1], "nnp_dma_scope", "vodma")
            s.pragma(s.get_loops(blk)[-1], "nnp_dma_scope", "eodma")
        else:
            s.pragma(s.get_loops(fetch_values_block)[-1], "nnp_dma_scope", "eodma")

    # inline and wrap as an opaque block to make compatible with general schedule
    primfunc = InlinePrimFuncCalls({"sort_impl": sort_impl})(s.mod)["main"]
    return primfunc
