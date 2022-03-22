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
import pytest
import tvm
from tvm import tir
from tvm.ir.expr import GlobalVar, PrimExpr
from tvm.ir.module import IRModule
from tvm.script import tir as T
from tvm.contrib.edgex.topi import naive_vu_schedule
from tvm.contrib.edgex.testing import check_edgex_tir_build
import numpy as np


# fmt: off
@T.prim_func
def qat_quantize_pattern1_func(input_0: T.handle, input_1: T.handle, input_2: T.handle, placeholder_3: T.Buffer[(1, 1, 1, 1), "int64"], placeholder_4: T.Buffer[(1, 1, 1, 1), "int64"], out: T.handle) -> None:
    h = T.var("int32")
    w = T.var("int32")    
    c = T.var("int32")    
    placeholder_0 = T.match_buffer(input_0, [1, h, w, c], dtype="int32")
    placeholder_1 = T.match_buffer(input_1, [1, h, w, c], dtype="int32")
    placeholder_2 = T.match_buffer(input_2, [c], dtype="int32")
    T_cast = T.match_buffer(out, [1, h, w, c], dtype="uint8")

    T_round_right_shift_intrin = T.alloc_buffer([1, h, w, c], dtype="int64")
    T_multiply = T.alloc_buffer([1, h, w, c], dtype="int64")
    compute = T.alloc_buffer([1, h, w, c], dtype="int64")
    T_add = T.alloc_buffer([1, h, w, c], dtype="int32")
    T_cast_1 = T.alloc_buffer([1, h, w, c], dtype="int64")
    T_subtract = T.alloc_buffer([1, h, w, c], dtype="int32")
    for i0, i1, i2, i3 in T.grid(1, h, w, c):
        with T.block("T_subtract"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(placeholder_0[ax0, ax1, ax2, ax3], placeholder_1[ax0, ax1, ax2, ax3])
            T.writes(T_subtract[ax0, ax1, ax2, ax3])
            T_subtract[ax0, ax1, ax2, ax3] = placeholder_0[ax0, ax1, ax2, ax3] - placeholder_1[ax0, ax1, ax2, ax3]
    for i0, i1, i2, i3 in T.grid(1, h, w, c):
        with T.block("T_add"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_subtract[ax0, ax1, ax2, ax3], placeholder_2[ax3])
            T.writes(T_add[ax0, ax1, ax2, ax3])
            T_add[ax0, ax1, ax2, ax3] = T_subtract[ax0, ax1, ax2, ax3] + placeholder_2[ax3]
    for i0, i1, i2, i3 in T.grid(1, h, w, c):
        with T.block("T_cast"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_add[ax0, ax1, ax2, ax3])
            T.writes(T_cast_1[ax0, ax1, ax2, ax3])
            T_cast_1[ax0, ax1, ax2, ax3] = T.cast(T_add[ax0, ax1, ax2, ax3], "int64")
    for i0, i1, i2, i3 in T.grid(1, h, w, c):
        with T.block("T_multiply"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_cast_1[ax0, ax1, ax2, ax3], placeholder_3[ax0, 0, 0, 0])
            T.writes(T_multiply[ax0, ax1, ax2, ax3])
            T_multiply[ax0, ax1, ax2, ax3] = T_cast_1[ax0, ax1, ax2, ax3] * placeholder_3[ax0, 0, 0, 0]
    for i0, i1, i2, i3 in T.grid(1, h, w, c):
        with T.block("T_round_right_shift_intrin"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_multiply[ax0, ax1, ax2, ax3], placeholder_4[ax0, 0, 0, 0])
            T.writes(T_round_right_shift_intrin[ax0, ax1, ax2, ax3])
            T_round_right_shift_intrin[ax0, ax1, ax2, ax3] = T.nnp_round_right_shift(T_multiply[ax0, ax1, ax2, ax3], placeholder_4[ax0, 0, 0, 0], dtype="int64")
    for i0, i1, i2, i3 in T.grid(1, h, w, c):
        with T.block("compute"):
            i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_round_right_shift_intrin[i0_1, i1_1, i2_1, i3_1])
            T.writes(compute[i0_1, i1_1, i2_1, i3_1])
            compute[i0_1, i1_1, i2_1, i3_1] = T.max(T.min(T_round_right_shift_intrin[i0_1, i1_1, i2_1, i3_1], T.int64(255)), T.int64(0))
    for i0, i1, i2, i3 in T.grid(1, h, w, c):
        with T.block("T_cast_1"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(compute[ax0, ax1, ax2, ax3])
            T.writes(T_cast[ax0, ax1, ax2, ax3])
            T_cast[ax0, ax1, ax2, ax3] = T.cast(compute[ax0, ax1, ax2, ax3], "uint8")




@T.prim_func
def qat_quantize_pattern2_func(input: T.handle, placeholder_1: T.Buffer[(1, 1, 1, 1), "int32"], placeholder_2: T.Buffer[(1, 1, 1, 1), "int64"], placeholder_3: T.Buffer[(1, 1, 1, 1), "int64"], output: T.handle) -> None:
    h = T.var("int32")
    w = T.var("int32")    
    c = T.var("int32")
    placeholder_0 = T.match_buffer(input, [1, h, w, c], dtype="int32")
    T_cast = T.match_buffer(output, [1, h, w, c], dtype="int32")
    T_round_right_shift_intrin = T.alloc_buffer([1, h, w, c], dtype="int64")
    T_cast_1 = T.alloc_buffer([1, h, w, c], dtype="int64")
    T_subtract = T.alloc_buffer([1, h, w, c], dtype="int32")
    T_multiply = T.alloc_buffer([1, h, w, c], dtype="int64")
    for i0, i1, i2, i3 in T.grid(1, h, w, c):
        with T.block("T_subtract"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(placeholder_0[ax0, ax1, ax2, ax3], placeholder_1[ax0, 0, 0, 0])
            T.writes(T_subtract[ax0, ax1, ax2, ax3])
            T.block_attr({"relay_op_name":"subtract"})
            T_subtract[ax0, ax1, ax2, ax3] = placeholder_0[ax0, ax1, ax2, ax3] - placeholder_1[ax0, 0, 0, 0]
    for i0, i1, i2, i3 in T.grid(1, h, w, c):
        with T.block("T_cast"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_subtract[ax0, ax1, ax2, ax3])
            T.writes(T_cast_1[ax0, ax1, ax2, ax3])
            T_cast_1[ax0, ax1, ax2, ax3] = T.cast(T_subtract[ax0, ax1, ax2, ax3], "int64")
    for i0, i1, i2, i3 in T.grid(1, h, w, c):
        with T.block("T_multiply"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_cast_1[ax0, ax1, ax2, ax3], placeholder_2[ax0, 0, 0, 0])
            T.writes(T_multiply[ax0, ax1, ax2, ax3])
            T_multiply[ax0, ax1, ax2, ax3] = T_cast_1[ax0, ax1, ax2, ax3] * placeholder_2[ax0, 0, 0, 0]
    for i0, i1, i2, i3 in T.grid(1, h, w, c):
        with T.block("T_round_right_shift_intrin"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_multiply[ax0, ax1, ax2, ax3], placeholder_3[ax0, 0, 0, 0])
            T.writes(T_round_right_shift_intrin[ax0, ax1, ax2, ax3])
            T_round_right_shift_intrin[ax0, ax1, ax2, ax3] = T.nnp_round_right_shift(T_multiply[ax0, ax1, ax2, ax3], placeholder_3[ax0, 0, 0, 0], dtype="int64")
    for i0, i1, i2, i3 in T.grid(1, h, w, c):
        with T.block("T_cast_1"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_round_right_shift_intrin[ax0, ax1, ax2, ax3])
            T.writes(T_cast[ax0, ax1, ax2, ax3])
            T_cast[ax0, ax1, ax2, ax3] = T.cast(T_round_right_shift_intrin[ax0, ax1, ax2, ax3], "int32")



@T.prim_func
def veltadd_binary_relu(a: T.handle, b: T.handle, mullt_norm: T.handle, shift_norm: T.handle, c: T.handle) -> None:
    n = T.var("int32")
    h = T.var("int32")
    w = T.var("int32")
    A = T.match_buffer(a, [1, n, h, w], dtype="int8")
    B = T.match_buffer(b, [1, n, h, w], dtype="int8")
    MulNorm = T.match_buffer(mullt_norm, [n, 1, 1], dtype="uint8")
    ShiftNorm = T.match_buffer(shift_norm, [n, 1, 1], dtype="uint8")
    C = T.match_buffer(c, [1, n, h, w], dtype="int8")
    arg0_i32 = T.alloc_buffer([1, n, h, w], dtype="int32")
    arg1_i32 = T.alloc_buffer([1, n, h, w], dtype="int32")
    T_add = T.alloc_buffer([1, n, h, w], dtype="int32")
    T_cast_add = T.alloc_buffer([1, n, h, w], dtype="int64")
    T_cast_mulnorm = T.alloc_buffer([n, 1, 1], dtype="int64")
    T_cast_shiftnorm = T.alloc_buffer([n, 1, 1], dtype="int64")
    T_multiply = T.alloc_buffer([1, n, h, w], dtype="int64")
    T_round_right_shift = T.alloc_buffer([1, n, h, w], dtype="int64")
    T_clip = T.alloc_buffer([1, n, h, w], dtype="int64")
    T_cast_i8 = T.alloc_buffer([1, n, h, w], dtype="int8")
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("cast0"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            arg0_i32[v0, v1, v2, v3] = T.cast(A[v0, v1, v2, v3], "int32")
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("cast1"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            arg1_i32[v0, v1, v2, v3] = T.cast(B[v0, v1, v2, v3], "int32")
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("compute"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_add[v0, v1, v2, v3] = arg0_i32[v0, v1, v2, v3] + arg1_i32[v0, v1, v2, v3]
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("cast_add"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_cast_add[v0, v1, v2, v3] = T.cast(T_add[v0, v1, v2, v3], "int64")
    for i0, i1, i2 in T.grid(n, 1, 1):
        with T.block("cast_mult_norm"):
            v0, v1, v2 = T.axis.remap("SSS", [i0, i1, i2])
            T_cast_mulnorm[v0, v1, v2] = T.cast(MulNorm[v0, v1, v2], "int64")
    for i0, i1, i2 in T.grid(n, 1, 1):
        with T.block("cast_shift_norm"):
            v0, v1, v2 = T.axis.remap("SSS", [i0, i1, i2])
            T_cast_shiftnorm[v0, v1, v2] = T.cast(ShiftNorm[v0, v1, v2], "int64")
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("multiply"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_multiply[v0, v1, v2, v3] = T_cast_add[v0, v1, v2, v3] * T_cast_mulnorm[v1, 0, 0]
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("round_right_shift"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_round_right_shift[v0, v1, v2, v3] = T.nnp_round_right_shift(
                T_multiply[v0, v1, v2, v3], T_cast_shiftnorm[v1, 0, 0], dtype="int64")
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("clip"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_clip[v0, v1, v2, v3] = T.max(T.min(T_round_right_shift[v0, v1, v2, v3], T.int64(127)), T.int64(-128))
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("cast_back_i8"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_cast_i8[v0, v1, v2, v3] = T.cast(T_clip[v0, v1, v2, v3], "int8")
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("relu"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            C[v0, v1, v2, v3] = T.max(T_cast_i8[v0, v1, v2, v3], T.int8(0))


@T.prim_func
def veltadd_binary(a: T.handle, b: T.handle, mullt_norm: T.handle, shift_norm: T.handle, c: T.handle) -> None:
    n = T.var("int32")
    h = T.var("int32")
    w = T.var("int32")
    A = T.match_buffer(a, [1, n, h, w], dtype="int8")
    B = T.match_buffer(b, [1, n, h, w], dtype="int8")
    MulNorm = T.match_buffer(mullt_norm, [n, 1, 1], dtype="uint8")
    ShiftNorm = T.match_buffer(shift_norm, [n, 1, 1], dtype="uint8")
    C = T.match_buffer(c, [1, n, h, w], dtype="int8")
    arg0_i32 = T.alloc_buffer([1, n, h, w], dtype="int32")
    arg1_i32 = T.alloc_buffer([1, n, h, w], dtype="int32")
    T_add = T.alloc_buffer([1, n, h, w], dtype="int32")
    T_cast_add = T.alloc_buffer([1, n, h, w], dtype="int64")
    T_cast_mulnorm = T.alloc_buffer([n, 1, 1], dtype="int64")
    T_cast_shiftnorm = T.alloc_buffer([n, 1, 1], dtype="int64")
    T_multiply = T.alloc_buffer([1, n, h, w], dtype="int64")
    T_round_right_shift = T.alloc_buffer([1, n, h, w], dtype="int64")
    T_clip = T.alloc_buffer([1, n, h, w], dtype="int64")
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("cast0"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            arg0_i32[v0, v1, v2, v3] = T.cast(A[v0, v1, v2, v3], "int32")
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("cast1"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            arg1_i32[v0, v1, v2, v3] = T.cast(B[v0, v1, v2, v3], "int32")
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("compute"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_add[v0, v1, v2, v3] = arg0_i32[v0, v1, v2, v3] + arg1_i32[v0, v1, v2, v3]
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("cast_add"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_cast_add[v0, v1, v2, v3] = T.cast(T_add[v0, v1, v2, v3], "int64")
    for i0, i1, i2 in T.grid(n, 1, 1):
        with T.block("cast_mult_norm"):
            v0, v1, v2 = T.axis.remap("SSS", [i0, i1, i2])
            T_cast_mulnorm[v0, v1, v2] = T.cast(MulNorm[v0, v1, v2], "int64")
    for i0, i1, i2 in T.grid(n, 1, 1):
        with T.block("cast_shift_norm"):
            v0, v1, v2 = T.axis.remap("SSS", [i0, i1, i2])
            T_cast_shiftnorm[v0, v1, v2] = T.cast(ShiftNorm[v0, v1, v2], "int64")
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("multiply"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_multiply[v0, v1, v2, v3] = T_cast_add[v0, v1, v2, v3] * T_cast_mulnorm[v1, 0, 0]
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("round_right_shift"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_round_right_shift[v0, v1, v2, v3] = T.nnp_round_right_shift(
                T_multiply[v0, v1, v2, v3], T_cast_shiftnorm[v1, 0, 0], dtype="int64")
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("clip"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_clip[v0, v1, v2, v3] = T.max(T.min(T_round_right_shift[v0, v1, v2, v3], T.int64(127)), T.int64(-128))
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("cast_back_i8"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            C[v0, v1, v2, v3] = T.cast(T_clip[v0, v1, v2, v3], "int8")


@T.prim_func
def veltadd_unary_relu(a: T.handle, mullt_norm: T.handle, shift_norm: T.handle, c: T.handle) -> None:
    n = T.var("int32")
    h = T.var("int32")
    w = T.var("int32")
    A = T.match_buffer(a, [1, n, h, w], dtype="int8")
    MulNorm = T.match_buffer(mullt_norm, [n, 1, 1], dtype="uint8")
    ShiftNorm = T.match_buffer(shift_norm, [n, 1, 1], dtype="uint8")
    C = T.match_buffer(c, [1, n, h, w], dtype="int8")
    T_cast_add = T.alloc_buffer([1, n, h, w], dtype="int64")
    T_cast_mulnorm = T.alloc_buffer([n, 1, 1], dtype="int64")
    T_cast_shiftnorm = T.alloc_buffer([n, 1, 1], dtype="int64")
    T_multiply = T.alloc_buffer([1, n, h, w], dtype="int64")
    T_round_right_shift = T.alloc_buffer([1, n, h, w], dtype="int64")
    T_clip = T.alloc_buffer([1, n, h, w], dtype="int64")
    T_cast_i8 = T.alloc_buffer([1, n, h, w], dtype="int8")
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("cast_add"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_cast_add[v0, v1, v2, v3] = T.cast(A[v0, v1, v2, v3], "int64")
    for i0, i1, i2 in T.grid(n, 1, 1):
        with T.block("cast_mult_norm"):
            v0, v1, v2 = T.axis.remap("SSS", [i0, i1, i2])
            T_cast_mulnorm[v0, v1, v2] = T.cast(MulNorm[v0, v1, v2], "int64")
    for i0, i1, i2 in T.grid(n, 1, 1):
        with T.block("cast_shift_norm"):
            v0, v1, v2 = T.axis.remap("SSS", [i0, i1, i2])
            T_cast_shiftnorm[v0, v1, v2] = T.cast(ShiftNorm[v0, v1, v2], "int64")
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("multiply"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_multiply[v0, v1, v2, v3] = T_cast_add[v0, v1, v2, v3] * T_cast_mulnorm[v1, 0, 0]
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("round_right_shift"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_round_right_shift[v0, v1, v2, v3] = T.nnp_round_right_shift(
                T_multiply[v0, v1, v2, v3], T_cast_shiftnorm[v1, 0, 0], dtype="int64")
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("clip"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_clip[v0, v1, v2, v3] = T.max(T.min(T_round_right_shift[v0, v1, v2, v3], T.int64(127)), T.int64(-128))
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("cast_back_i8"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_cast_i8[v0, v1, v2, v3] = T.cast(T_clip[v0, v1, v2, v3], "int8")
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("relu"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            C[v0, v1, v2, v3] = T.max(T_cast_i8[v0, v1, v2, v3], T.int8(0))

@T.prim_func
def veltadd_unary(a: T.handle, mullt_norm: T.handle, shift_norm: T.handle, c: T.handle) -> None:
    n = T.var("int32")
    h = T.var("int32")
    w = T.var("int32")
    A = T.match_buffer(a, [1, n, h, w], dtype="int8")
    MulNorm = T.match_buffer(mullt_norm, [n, 1, 1], dtype="uint8")
    ShiftNorm = T.match_buffer(shift_norm, [n, 1, 1], dtype="uint8")
    C = T.match_buffer(c, [1, n, h, w], dtype="int8")
    T_cast_add = T.alloc_buffer([1, n, h, w], dtype="int64")
    T_cast_mulnorm = T.alloc_buffer([n, 1, 1], dtype="int64")
    T_cast_shiftnorm = T.alloc_buffer([n, 1, 1], dtype="int64")
    T_multiply = T.alloc_buffer([1, n, h, w], dtype="int64")
    T_round_right_shift = T.alloc_buffer([1, n, h, w], dtype="int64")
    T_clip = T.alloc_buffer([1, n, h, w], dtype="int64")
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("cast_add"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_cast_add[v0, v1, v2, v3] = T.cast(A[v0, v1, v2, v3], "int64")
    for i0, i1, i2 in T.grid(n, 1, 1):
        with T.block("cast_mult_norm"):
            v0, v1, v2 = T.axis.remap("SSS", [i0, i1, i2])
            T_cast_mulnorm[v0, v1, v2] = T.cast(MulNorm[v0, v1, v2], "int64")
    for i0, i1, i2 in T.grid(n, 1, 1):
        with T.block("cast_shift_norm"):
            v0, v1, v2 = T.axis.remap("SSS", [i0, i1, i2])
            T_cast_shiftnorm[v0, v1, v2] = T.cast(ShiftNorm[v0, v1, v2], "int64")
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("multiply"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_multiply[v0, v1, v2, v3] = T_cast_add[v0, v1, v2, v3] * T_cast_mulnorm[v1, 0, 0]
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("round_right_shift"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_round_right_shift[v0, v1, v2, v3] = T.nnp_round_right_shift(
                T_multiply[v0, v1, v2, v3], T_cast_shiftnorm[v1, 0, 0], dtype="int64")
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("clip"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_clip[v0, v1, v2, v3] = T.max(T.min(T_round_right_shift[v0, v1, v2, v3], T.int64(127)), T.int64(-128))
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("cast_back_i8"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            C[v0, v1, v2, v3] = T.cast(T_clip[v0, v1, v2, v3], "int8")

@T.prim_func
def quantize_i32_input(a: T.handle, mullt_norm: T.handle, shift_norm: T.handle, c: T.handle) -> None:
    n = T.var("int32")
    h = T.var("int32")
    w = T.var("int32")
    A = T.match_buffer(a, [1, n, h, w], dtype="int32")
    MulNorm = T.match_buffer(mullt_norm, [n, 1, 1], dtype="uint8")
    ShiftNorm = T.match_buffer(shift_norm, [n, 1, 1], dtype="uint8")
    C = T.match_buffer(c, [1, n, h, w], dtype="int8")
    T_cast_add = T.alloc_buffer([1, n, h, w], dtype="int64")
    T_cast_mulnorm = T.alloc_buffer([n, 1, 1], dtype="int64")
    T_cast_shiftnorm = T.alloc_buffer([n, 1, 1], dtype="int64")
    T_multiply = T.alloc_buffer([1, n, h, w], dtype="int64")
    T_round_right_shift = T.alloc_buffer([1, n, h, w], dtype="int64")
    T_clip = T.alloc_buffer([1, n, h, w], dtype="int64")
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("cast_add"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_cast_add[v0, v1, v2, v3] = T.cast(A[v0, v1, v2, v3], "int64")
    for i0, i1, i2 in T.grid(n, 1, 1):
        with T.block("cast_mult_norm"):
            v0, v1, v2 = T.axis.remap("SSS", [i0, i1, i2])
            T_cast_mulnorm[v0, v1, v2] = T.cast(MulNorm[v0, v1, v2], "int64")
    for i0, i1, i2 in T.grid(n, 1, 1):
        with T.block("cast_shift_norm"):
            v0, v1, v2 = T.axis.remap("SSS", [i0, i1, i2])
            T_cast_shiftnorm[v0, v1, v2] = T.cast(ShiftNorm[v0, v1, v2], "int32")
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("multiply"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_multiply[v0, v1, v2, v3] = T_cast_add[v0, v1, v2, v3] * T_cast_mulnorm[v1, 0, 0]
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("round_right_shift"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_round_right_shift[v0, v1, v2, v3] = T.nnp_round_right_shift(
                T_multiply[v0, v1, v2, v3], T_cast_shiftnorm[v1, 0, 0], dtype="int64")
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("clip"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_clip[v0, v1, v2, v3] = T.max(T.min(T_round_right_shift[v0, v1, v2, v3], T.int64(127)), T.int64(-128))
    for i0, i1, i2, i3 in T.grid(1, n, h, w):
        with T.block("cast_back_i8"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            C[v0, v1, v2, v3] = T.cast(T_clip[v0, v1, v2, v3], "int8")
# fmt: on


@T.prim_func
def quantize_per_tensor_i32_input(
    a: T.handle, mullt_norm: T.handle, shift_norm: T.handle, out: T.handle
) -> None:
    c = T.var("int32")
    h = T.var("int32")
    w = T.var("int32")
    A = T.match_buffer(a, [1, c, h, w], dtype="int32")
    MulNorm = T.match_buffer(mullt_norm, [1, 1, 1, 1], dtype="uint8")
    ShiftNorm = T.match_buffer(shift_norm, [1, 1, 1, 1], dtype="uint8")
    C = T.match_buffer(out, [1, c, h, w], dtype="int8")
    T_cast_add = T.alloc_buffer([1, c, h, w], dtype="int64")
    T_cast_mulnorm = T.alloc_buffer([1, 1, 1, 1], dtype="int64")
    T_cast_shiftnorm = T.alloc_buffer([1, 1, 1, 1], dtype="int64")
    T_multiply = T.alloc_buffer([1, c, h, w], dtype="int64")
    T_round_right_shift = T.alloc_buffer([1, c, h, w], dtype="int64")
    T_clip = T.alloc_buffer([1, c, h, w], dtype="int64")
    for i0, i1, i2, i3 in T.grid(1, c, h, w):
        with T.block("cast_add"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_cast_add[v0, v1, v2, v3] = T.cast(A[v0, v1, v2, v3], "int64")
    for i0, i1, i2, i3 in T.grid(1, 1, 1, 1):
        with T.block("cast_mult_norm"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_cast_mulnorm[v0, v1, v2, v3] = T.cast(MulNorm[v0, v1, v2, v3], "int64")
    for i0, i1, i2, i3 in T.grid(1, 1, 1, 1):
        with T.block("cast_shift_norm"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_cast_shiftnorm[v0, v1, v2, v3] = T.cast(ShiftNorm[v0, v1, v2, v3], "int64")
    for i0, i1, i2, i3 in T.grid(1, c, h, w):
        with T.block("multiply"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_multiply[v0, v1, v2, v3] = T_cast_add[v0, v1, v2, v3] * T_cast_mulnorm[v0, 0, 0, 0]
    for i0, i1, i2, i3 in T.grid(1, c, h, w):
        with T.block("round_right_shift"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_round_right_shift[v0, v1, v2, v3] = T.nnp_round_right_shift(
                T_multiply[v0, v1, v2, v3], T_cast_shiftnorm[v0, 0, 0, 0], dtype="int64"
            )
    for i0, i1, i2, i3 in T.grid(1, c, h, w):
        with T.block("clip"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_clip[v0, v1, v2, v3] = T.max(
                T.min(T_round_right_shift[v0, v1, v2, v3], T.int64(127)), T.int64(-128)
            )
    for i0, i1, i2, i3 in T.grid(1, c, h, w):
        with T.block("cast_back_i8"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            C[v0, v1, v2, v3] = T.cast(T_clip[v0, v1, v2, v3], "int8")


def do_test_veltadd(channels, height, weight, has_add, has_relu):
    shape = [1, channels, height, weight]
    name = "veltadd%s%s_%d_%d_%d" % (
        "_binary" if has_add else "_unary",
        "_relu" if has_relu else "",
        channels,
        height,
        weight,
    )
    if has_add and has_relu:
        func = veltadd_binary_relu
    elif has_add:
        func = veltadd_binary
    elif has_relu:
        func = veltadd_unary_relu
    else:
        func = veltadd_unary
    func = func.specialize({func.params[0]: tir.decl_buffer(shape)})
    func = naive_vu_schedule(func)
    x = np.random.randint(-128, 127, shape).astype("int8")
    y = np.random.randint(-128, 127, shape).astype("int8")
    m = np.random.randint(0, 5, [channels, 1, 1]).astype("uint8")
    s = np.random.randint(0, 9, [channels, 1, 1]).astype("uint8")
    input_data = [x, y, m, s] if has_add else [x, m, s]
    check_edgex_tir_build(name, func, check_cpu=True, input_data=input_data)


def do_test_i32_quantize(channels, height, weight, per_tensor):
    shape = [1, channels, height, weight]
    name = "quantize_i32_input_%d_%d_%d" % (channels, height, weight)
    if per_tensor:
        func = quantize_per_tensor_i32_input
        m = np.random.randint(0, 5, [1, 1, 1, 1]).astype("uint8")
        s = np.random.randint(0, 9, [1, 1, 1, 1]).astype("uint8")
    else:
        func = quantize_i32_input
        m = np.random.randint(0, 5, [channels, 1, 1]).astype("uint8")
        s = np.random.randint(0, 9, [channels, 1, 1]).astype("uint8")
    x = np.random.randint(-128, 127, shape).astype("int32")
    func = func.specialize({func.params[0]: tir.decl_buffer(shape)})
    func = naive_vu_schedule(func)

    check_edgex_tir_build(name, func, check_cpu=True, input_data=[x, m, s])


def test_veltadd_binary_with_relu():
    do_test_veltadd(channels=32, height=1, weight=1, has_add=True, has_relu=True)
    do_test_veltadd(channels=32, height=28, weight=28, has_add=True, has_relu=True)


def test_veltadd_binary_no_relu():
    do_test_veltadd(channels=32, height=1, weight=1, has_add=True, has_relu=False)
    do_test_veltadd(channels=32, height=28, weight=28, has_add=True, has_relu=False)


def test_veltadd_unary_with_relu():
    do_test_veltadd(channels=32, height=1, weight=1, has_add=False, has_relu=True)
    do_test_veltadd(channels=32, height=28, weight=28, has_add=False, has_relu=True)


def test_veltadd_unary_no_relu():
    do_test_veltadd(channels=32, height=1, weight=1, has_add=False, has_relu=False)
    do_test_veltadd(channels=32, height=28, weight=28, has_add=False, has_relu=False)


def test_i32_quantize():
    do_test_i32_quantize(channels=32, height=1, weight=1, per_tensor=False)
    do_test_i32_quantize(channels=32, height=1, weight=1, per_tensor=True)
    do_test_i32_quantize(channels=32, height=32, weight=32, per_tensor=False)
    do_test_i32_quantize(channels=16, height=14, weight=14, per_tensor=False)


@pytest.mark.skip("skip because result mismatch.")
# mobilenet_v2_qat quantize pattern1: # cast_i64 -> multiply -> shift -> clip(0,255) -> cast_u8
def test_qat_quantize_pattern1():
    shape = [1, 7, 7, 960]
    primfunc = qat_quantize_pattern1_func
    primfunc = primfunc.specialize({primfunc.params[0]: tir.decl_buffer(shape)})
    edgex_schedule = naive_vu_schedule(primfunc, is_cpu=False, allow_multi_block=True)
    cpu_schedule = naive_vu_schedule(primfunc, is_cpu=True, allow_multi_block=True)

    x = np.random.randint(-10000, 10000, shape).astype("int32")
    y = np.random.randint(-10000, 10000, shape).astype("int32")
    z = np.random.randint(-10000, 10000, shape[-1]).astype("int32")
    m = np.random.randint(0, 5, [1, 1, 1, 1]).astype("uint8")
    s = np.random.randint(0, 9, [1, 1, 1, 1]).astype("uint8")
    check_edgex_tir_build(
        "qat_quantize_pattern1",
        edgex_schedule,
        cpu_prim_func=cpu_schedule,
        check_cpu=True,
        input_data=[x, y, z, m, s],
    )


# mobilenet_v2_qat quantize pattern2: cast_i64 -> multiply -> shift -> cast_i32
def test_qat_quantize_pattern2():
    shape = [1, 28, 28, 32]
    primfunc = qat_quantize_pattern2_func
    primfunc = primfunc.specialize({primfunc.params[0]: tir.decl_buffer(shape)})
    edgex_schedule = naive_vu_schedule(primfunc, is_cpu=False, allow_multi_block=True)
    cpu_schedule = naive_vu_schedule(primfunc, is_cpu=True, allow_multi_block=True)

    x = np.random.randint(-1000, 1000, shape).astype("int32")
    y = np.random.randint(-1000, 1000, [1, 1, 1, 1]).astype("int32")
    m = np.random.randint(0, 5, [1, 1, 1, 1]).astype("uint8")
    s = np.random.randint(0, 9, [1, 1, 1, 1]).astype("uint8")
    check_edgex_tir_build(
        "qat_quantize_pattern2",
        edgex_schedule,
        cpu_prim_func=cpu_schedule,
        check_cpu=True,
        input_data=[x, y, m, s],
    )


def test_setmode_side_effect():
    @T.prim_func
    def veltadd_of_different_relu_mode(
        a: T.handle, mullt_norm: T.handle, shift_norm: T.handle, c1: T.handle, c2: T.handle
    ) -> None:
        A = T.match_buffer(a, [1, 32, 1, 1], dtype="int8")
        MulNorm = T.match_buffer(mullt_norm, [32, 1, 1], dtype="uint8")
        ShiftNorm = T.match_buffer(shift_norm, [32, 1, 1], dtype="uint8")
        C1 = T.match_buffer(c1, [1, 32, 1, 1], dtype="int8")
        C2 = T.match_buffer(c2, [1, 32, 1, 1], dtype="int8")
        T.evaluate(
            T.call_extern(
                "veltadd_use_relu", A.data, MulNorm.data, ShiftNorm.data, C1.data, dtype=""
            )
        )
        T.evaluate(
            T.call_extern(
                "veltadd_no_relu", A.data, MulNorm.data, ShiftNorm.data, C2.data, dtype=""
            )
        )

    shape = [1, 32, 1, 1]
    extern_primfuncs = {
        "veltadd_use_relu": veltadd_unary_relu.specialize(
            {veltadd_unary_relu.params[0]: tir.decl_buffer(shape)}
        ),
        "veltadd_no_relu": veltadd_unary.specialize(
            {veltadd_unary.params[0]: tir.decl_buffer(shape)}
        ),
    }
    mod = IRModule.from_expr(veltadd_of_different_relu_mode)
    mod = tvm.contrib.edgex.tir.transform.InlinePrimFuncCalls(extern_primfuncs)(mod)
    func = naive_vu_schedule(mod["main"], allow_multi_block=True)
    x = np.random.randint(-128, 127, shape).astype("int8")
    m = np.random.randint(0, 5, [32, 1, 1]).astype("uint8")
    s = np.random.randint(0, 9, [32, 1, 1]).astype("uint8")
    check_edgex_tir_build("veltadd_relu_and_norelu", func, check_cpu=True, input_data=[x, m, s])


if __name__ == "__main__":
    test_i32_quantize()
    test_veltadd_binary_with_relu()
    test_veltadd_binary_no_relu()
    test_veltadd_unary_with_relu()
    test_veltadd_unary_no_relu()
    test_setmode_side_effect()
    test_qat_quantize_pattern2()
    # test_qat_quantize_pattern1()
