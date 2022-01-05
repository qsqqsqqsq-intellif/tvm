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
import tvm
from tvm import tir
from tvm.ir.expr import GlobalVar, PrimExpr
from tvm.ir.module import IRModule
from tvm.script import tir as T
from tvm.contrib.edgex.topi import naive_vu_schedule
from tvm.contrib.edgex.topi import NaiveVuSchedule
from tvm.contrib.edgex.testing import check_edgex_tir_build
import numpy as np


# fmt: off
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
# fmt: on


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
    func = NaiveVuSchedule(func).schedule()
    x = np.random.randint(-128, 127, shape).astype("int8")
    y = np.random.randint(-128, 127, shape).astype("int8")
    m = np.random.randint(0, 5, [channels, 1, 1]).astype("uint8")
    s = np.random.randint(0, 9, [channels, 1, 1]).astype("uint8")
    input_data = [x, y, m, s] if has_add else [x, m, s]
    check_edgex_tir_build(name, func, check_cpu=True, input_data=input_data)


def do_test_i32_quantize(channels, height, weight):
    shape = [1, channels, height, weight]
    name = "quantize_i32_input_%d_%d_%d" % (channels, height, weight)
    func = quantize_i32_input
    func = func.specialize({func.params[0]: tir.decl_buffer(shape)})
    func = NaiveVuSchedule(func).schedule()
    x = np.random.randint(-128, 127, shape).astype("int32")
    m = np.random.randint(0, 5, [channels, 1, 1]).astype("uint8")
    s = np.random.randint(0, 9, [channels, 1, 1]).astype("uint8")
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
    do_test_i32_quantize(channels=32, height=1, weight=1)
    # do_test_i32_quantize(channels=16, height=14, weight=14)


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
    func = NaiveVuSchedule(mod["main"], allow_multi_block=True).schedule()
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
