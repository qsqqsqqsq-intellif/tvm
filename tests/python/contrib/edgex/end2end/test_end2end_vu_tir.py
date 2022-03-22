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
from tvm import tir
from tvm.script import tir as T
from tvm.contrib.edgex.topi import naive_vu_schedule
from tvm.contrib.edgex.testing import check_edgex_tir_build
import numpy as np

# fmt: off
@T.prim_func
def fused_multiply_cast_multiply_round_right_shift_clip_cast(placeholder_0: T.Buffer[(1, 16, 1, 1), "int32"], placeholder_1: T.Buffer[(1, 16, 56, 56), "int32"], placeholder_2: T.Buffer[(16, 1, 1), "int64"], placeholder_3: T.Buffer[(16, 1, 1), "int64"], T_cast: T.Buffer[(1, 16, 56, 56), "int8"]) -> None:
    T_round_right_shift_intrin = T.alloc_buffer([1, 16, 56, 56], dtype="int64")
    compute = T.alloc_buffer([1, 16, 56, 56], dtype="int64")
    T_cast_1 = T.alloc_buffer([1, 16, 56, 56], dtype="int64")
    T_multiply = T.alloc_buffer([1, 16, 56, 56], dtype="int64")
    T_multiply_1 = T.alloc_buffer([1, 16, 56, 56], dtype="int32")
    for i0, i1, i2, i3 in T.grid(1, 16, 56, 56):
        with T.block("T_multiply"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(placeholder_0[ax0, ax1, 0, 0], placeholder_1[ax0, ax1, ax2, ax3])
            T.writes(T_multiply_1[ax0, ax1, ax2, ax3])
            T_multiply_1[ax0, ax1, ax2, ax3] = placeholder_0[ax0, ax1, 0, 0] * placeholder_1[ax0, ax1, ax2, ax3]
    for i0, i1, i2, i3 in T.grid(1, 16, 56, 56):
        with T.block("T_cast"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_multiply_1[ax0, ax1, ax2, ax3])
            T.writes(T_cast_1[ax0, ax1, ax2, ax3])
            T_cast_1[ax0, ax1, ax2, ax3] = T.cast(T_multiply_1[ax0, ax1, ax2, ax3], "int64")
    for i0, i1, i2, i3 in T.grid(1, 16, 56, 56):
        with T.block("T_multiply_1"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_cast_1[ax0, ax1, ax2, ax3], placeholder_2[ax1, 0, 0])
            T.writes(T_multiply[ax0, ax1, ax2, ax3])
            T_multiply[ax0, ax1, ax2, ax3] = T_cast_1[ax0, ax1, ax2, ax3] * placeholder_2[ax1, 0, 0]
    for i0, i1, i2, i3 in T.grid(1, 16, 56, 56):
        with T.block("T_round_right_shift_intrin"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_multiply[ax0, ax1, ax2, ax3], placeholder_3[ax1, 0, 0])
            T.writes(T_round_right_shift_intrin[ax0, ax1, ax2, ax3])
            T_round_right_shift_intrin[ax0, ax1, ax2, ax3] = T.nnp_round_right_shift(T_multiply[ax0, ax1, ax2, ax3], placeholder_3[ax1, 0, 0], dtype="int64")
    for i0, i1, i2, i3 in T.grid(1, 16, 56, 56):
        with T.block("compute"):
            i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_round_right_shift_intrin[i0_1, i1_1, i2_1, i3_1])
            T.writes(compute[i0_1, i1_1, i2_1, i3_1])
            compute[i0_1, i1_1, i2_1, i3_1] = T.max(T.min(T_round_right_shift_intrin[i0_1, i1_1, i2_1, i3_1], T.int64(127)), T.int64(-128))
    for i0, i1, i2, i3 in T.grid(1, 16, 56, 56):
        with T.block("T_cast_1"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(compute[ax0, ax1, ax2, ax3])
            T.writes(T_cast[ax0, ax1, ax2, ax3])
            T_cast[ax0, ax1, ax2, ax3] = T.cast(compute[ax0, ax1, ax2, ax3], "int8")



@T.prim_func
def fused_round_clip_cast(placeholder_0: T.Buffer[(1, 288, 14, 14), "float16"], T_cast: T.Buffer[(1, 288, 14, 14), "int8"]) -> None:
    T_round = T.alloc_buffer([1, 288, 14, 14], dtype="float16")
    compute = T.alloc_buffer([1, 288, 14, 14], dtype="float16")
    for i0, i1, i2, i3 in T.grid(1, 288, 14, 14):
        with T.block("T_round"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(placeholder_0[ax0, ax1, ax2, ax3])
            T.writes(T_round[ax0, ax1, ax2, ax3])
            T_round[ax0, ax1, ax2, ax3] = T.round(placeholder_0[ax0, ax1, ax2, ax3], dtype="float16")
    for i0, i1, i2, i3 in T.grid(1, 288, 14, 14):
        with T.block("compute"):
            i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_round[i0_1, i1_1, i2_1, i3_1])
            T.writes(compute[i0_1, i1_1, i2_1, i3_1])
            compute[i0_1, i1_1, i2_1, i3_1] = T.max(T.min(T_round[i0_1, i1_1, i2_1, i3_1], T.float16(127)), T.float16(-128))
    for i0, i1, i2, i3 in T.grid(1, 288, 14, 14):
        with T.block("T_cast"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(compute[ax0, ax1, ax2, ax3])
            T.writes(T_cast[ax0, ax1, ax2, ax3])
            T_cast[ax0, ax1, ax2, ax3] = T.cast(compute[ax0, ax1, ax2, ax3], "int8")



@T.prim_func
def fused_subtract_nn_bias_add_cast(input_0: T.handle, input_1: T.handle, input_2: T.handle,  out: T.handle) -> None:
    h = T.var("int32")
    w = T.var("int32")    
    c = T.var("int32")    
    placeholder_0 = T.match_buffer(input_0, [1, h, w, c], dtype="int32")
    placeholder_1 = T.match_buffer(input_1, [1, h, w, 1], dtype="int32")
    placeholder_2 = T.match_buffer(input_2, [c], dtype="int32")
    T_cast = T.match_buffer(out, [1, h, w, c], dtype="uint8")
    T_add = T.alloc_buffer([1, h, w, c], dtype="int32")
    T_subtract = T.alloc_buffer([1, h, w, c], dtype="int32")
    for i0, i1, i2, i3 in T.grid(1, h, w, c):
        with T.block("T_subtract"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(placeholder_0[ax0, ax1, ax2, ax3], placeholder_1[ax0, ax1, ax2, 0])
            T.writes(T_subtract[ax0, ax1, ax2, ax3])
            T_subtract[ax0, ax1, ax2, ax3] = placeholder_0[ax0, ax1, ax2, ax3] - placeholder_1[ax0, ax1, ax2, 0]
    for i0, i1, i2, i3 in T.grid(1, h, w, c):
        with T.block("T_add"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_subtract[ax0, ax1, ax2, ax3], placeholder_2[ax3])
            T.writes(T_add[ax0, ax1, ax2, ax3])
            T_add[ax0, ax1, ax2, ax3] = T_subtract[ax0, ax1, ax2, ax3] + placeholder_2[ax3]
    for i0, i1, i2, i3 in T.grid(1, h, w, c):
        with T.block("T_cast_1"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_add[ax0, ax1, ax2, ax3])
            T.writes(T_cast[ax0, ax1, ax2, ax3])
            T_cast[ax0, ax1, ax2, ax3] = T.cast(T_add[ax0, ax1, ax2, ax3], "uint8")


@T.prim_func
def fused_reduce_sum_multiply(placeholder_0: T.Buffer[(1, 1, 1, 1), "int32"], placeholder_1: T.Buffer[(1, 128, 128, 32), "uint8"], T_multiply: T.Buffer[(1, 128, 128, 1), "int32"]) -> None:
    T_cast = T.alloc_buffer([1, 128, 128, 32], dtype="int32")
    placeholder_red = T.alloc_buffer([1, 128, 128, 1], dtype="int32")
    for i0, i1, i2, i3 in T.grid(1, 128, 128, 32):
        with T.block("T_cast"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(placeholder_1[ax0, ax1, ax2, ax3])
            T.writes(T_cast[ax0, ax1, ax2, ax3])
            T_cast[ax0, ax1, ax2, ax3] = T.cast(placeholder_1[ax0, ax1, ax2, ax3], "int32")
    for i0, i1, i2, i3, i4 in T.grid(1, 128, 128, 1, 32):
        with T.block("placeholder_red"):
            ax0, ax1, ax2, ax3, k3 = T.axis.remap("SSSSR", [i0, i1, i2, i3, i4])
            T.reads(placeholder_red[ax0, ax1, ax2, ax3], T_cast[ax0, ax1, ax2, k3])
            T.writes(placeholder_red[ax0, ax1, ax2, ax3])
            with T.init():
                placeholder_red[ax0, ax1, ax2, ax3] = 0
            placeholder_red[ax0, ax1, ax2, ax3] = placeholder_red[ax0, ax1, ax2, ax3] + T_cast[ax0, ax1, ax2, k3]
    for i0, i1, i2, i3 in T.grid(1, 128, 128, 1):
        with T.block("T_multiply"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(placeholder_0[ax0, 0, 0, ax3], placeholder_red[ax0, ax1, ax2, ax3])
            T.writes(T_multiply[ax0, ax1, ax2, ax3])
            T_multiply[ax0, ax1, ax2, ax3] = placeholder_0[ax0, 0, 0, ax3] * placeholder_red[ax0, ax1, ax2, ax3]
# fmt: on


def test_fused_reduce_sum_multiply():
    primfunc = fused_reduce_sum_multiply
    edgex_schedule = naive_vu_schedule(primfunc, is_cpu=False, allow_multi_block=True)
    cpu_schedule = naive_vu_schedule(primfunc, is_cpu=True, allow_multi_block=True)
    check_edgex_tir_build(
        "fused_reduce_sum_multiply",
        edgex_schedule,
        cpu_prim_func=cpu_schedule,
        check_cpu=True,
    )


def test_fused_subtract_nn_bias_add_cast():
    shape = [1, 7, 7, 1280]
    primfunc = fused_subtract_nn_bias_add_cast
    primfunc = primfunc.specialize({primfunc.params[0]: tir.decl_buffer(shape)})
    edgex_schedule = naive_vu_schedule(primfunc, is_cpu=False, allow_multi_block=True)
    cpu_schedule = naive_vu_schedule(primfunc, is_cpu=True, allow_multi_block=True)

    # TODO(@qing): Data range is limited to [0, 10], otherwise the rusult will be mismatched.
    x = np.random.randint(0, 10, shape).astype("int32")
    sub_value = np.random.randint(0, 10, [1, shape[1], shape[2], 1]).astype("int32")
    add_value = np.random.randint(0, 10, [shape[3]]).astype("int32")
    check_edgex_tir_build(
        "fused_subtract_nn_bias_add_cast",
        edgex_schedule,
        cpu_prim_func=cpu_schedule,
        check_cpu=True,
        input_data=[x, sub_value, add_value],
    )


def test_fused_round_clip_cast():
    primfunc = fused_round_clip_cast
    edgex_schedule = naive_vu_schedule(primfunc, is_cpu=False, allow_multi_block=True)
    cpu_schedule = naive_vu_schedule(primfunc, is_cpu=True, allow_multi_block=True)
    check_edgex_tir_build(
        "fused_round_clip_cast",
        edgex_schedule,
        cpu_prim_func=cpu_schedule,
        check_cpu=True,
    )


def test_fused_multiply_cast_multiply_round_right_shift_clip_cast():
    shape = [1, 16, 56, 56]
    channels = shape[1]
    primfunc = fused_multiply_cast_multiply_round_right_shift_clip_cast
    edgex_schedule = naive_vu_schedule(primfunc, is_cpu=False, allow_multi_block=True)
    cpu_schedule = naive_vu_schedule(primfunc, is_cpu=True, allow_multi_block=True)
    x = np.random.randint(-10000, 10000, [1, channels, 1, 1]).astype("int32")
    y = np.random.randint(-10000, 10000, shape).astype("int32")
    m = np.random.randint(0, 5, [channels, 1, 1]).astype("uint8")
    s = np.random.randint(0, 9, [channels, 1, 1]).astype("uint8")
    check_edgex_tir_build(
        "fused_multiply_cast_multiply_round_right_shift_clip_cast",
        edgex_schedule,
        cpu_prim_func=cpu_schedule,
        input_data=[x, y, m, s],
        check_cpu=True,
    )


if __name__ == "__main__":
    test_fused_reduce_sum_multiply()
    test_fused_subtract_nn_bias_add_cast()
    test_fused_round_clip_cast()
    test_fused_multiply_cast_multiply_round_right_shift_clip_cast()
