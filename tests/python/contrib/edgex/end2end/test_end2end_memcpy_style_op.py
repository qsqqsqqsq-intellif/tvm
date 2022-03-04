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
import numpy as np
import tvm.testing
import tvm.script.tir as T
from tvm.contrib.edgex.tir.schedule import EdgexSchedule
from tvm.contrib.edgex.topi import naive_vu_schedule
from tvm.contrib.edgex.testing import check_edgex_tir_build
from tvm.contrib.edgex.topi import schedule_memcpy_style_edgex_impl


@T.prim_func
def split_func(
    placeholder_0: T.Buffer[(1, 8, 10, 512), "float16"],
    T_split_sections: T.Buffer[(1, 8, 10, 256), "float16"],
    T_split_sections_1: T.Buffer[(1, 8, 10, 256), "float16"],
) -> None:
    # body
    # with T.block("root")
    for i0, i1, i2, i3 in T.grid(1, 8, 10, 256):
        with T.block("T_split_sections"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(placeholder_0[ax0, ax1, ax2, ax3])
            T.writes(T_split_sections[ax0, ax1, ax2, ax3])
            T.block_attr({"relay_op_attrs.axis": -1, "relay_op_name": "split"})
            T_split_sections[ax0, ax1, ax2, ax3] = placeholder_0[ax0, ax1, ax2, ax3]
    for i0, i1, i2, i3 in T.grid(1, 8, 10, 256):
        with T.block("T_split_sections_1"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(placeholder_0[ax0, ax1, ax2, ax3 + 256])
            T.writes(T_split_sections_1[ax0, ax1, ax2, ax3])
            T.block_attr({"relay_op_attrs.axis": -1, "relay_op_name": "split"})
            T_split_sections_1[ax0, ax1, ax2, ax3] = placeholder_0[ax0, ax1, ax2, ax3 + 256]


@T.prim_func
def transpose_func(
    placeholder_0: T.Buffer[(1, 8, 10, 2, 256), "float16"],
    T_transpose: T.Buffer[(1, 8, 10, 256, 2), "float16"],
) -> None:
    # body
    # with T.block("root")
    for i0, i1, i2, i3, i4 in T.grid(1, 8, 10, 256, 2):
        with T.block("T_transpose"):
            ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
            T.reads(placeholder_0[ax0, ax1, ax2, ax4, ax3])
            T.writes(T_transpose[ax0, ax1, ax2, ax3, ax4])
            T.block_attr({"relay_op_attrs.axes": [0, 1, 2, 4, 3], "relay_op_name": "transpose"})
            T_transpose[ax0, ax1, ax2, ax3, ax4] = placeholder_0[ax0, ax1, ax2, ax4, ax3]


@T.prim_func
def reshape_func(
    placeholder_0: T.Buffer[(1, 8, 10, 48), "int8"], T_reshape: T.Buffer[(1, 8, 10, 3, 16), "int8"]
) -> None:
    # body
    # with T.block("root")
    for i0, i1, i2, i3, i4 in T.grid(1, 8, 10, 3, 16):
        with T.block("T_reshape"):
            ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
            T.reads(
                placeholder_0[
                    0,
                    ((ax2 * 48 + ax3 * 16 + ax4) // 480 + ax1) % 8,
                    ((ax3 * 16 + ax4) // 48 + ax2) % 10,
                    (ax3 * 16 + ax4) % 48,
                ]
            )
            T.writes(T_reshape[ax0, ax1, ax2, ax3, ax4])
            T.block_attr({"relay_op_attrs.newshape": [1, 8, 10, 3, 16], "relay_op_name": "reshape"})
            T_reshape[ax0, ax1, ax2, ax3, ax4] = placeholder_0[
                0,
                ((ax2 * 48 + ax3 * 16 + ax4) // 480 + ax1) % 8,
                ((ax3 * 16 + ax4) // 48 + ax2) % 10,
                (ax3 * 16 + ax4) % 48,
            ]


@T.prim_func
def concat_func(
    input_0: T.handle,
    input_1: T.handle,
    input_2: T.handle,
    input_3: T.handle,
    output: T.handle,
) -> None:
    # function attr dict
    T.func_attr({"tir.noalias": True})
    # body
    # with T.block("root")
    w = T.var("int32")
    h = T.var("int32")
    c0 = T.var("int32")
    c1 = T.var("int32")
    c2 = T.var("int32")
    c3 = T.var("int32")
    c_out = T.var("int32")

    placeholder = T.match_buffer(input_0, [1, c0, w, h], dtype="int8")
    placeholder_1 = T.match_buffer(input_1, [1, c1, w, h], dtype="int8")
    placeholder_2 = T.match_buffer(input_2, [1, c2, w, h], dtype="int8")
    placeholder_3 = T.match_buffer(input_3, [1, c3, w, h], dtype="int8")
    T_concat = T.match_buffer(output, [1, c_out, w, h], dtype="int8")
    for i0, i1, i2, i3 in T.grid(1, c_out, w, h):
        with T.block("T_concat"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(
                placeholder_3[ax0, ax1 - (c0 + c1 + c2), ax2, ax3],
                placeholder_2[ax0, ax1 - (c0 + c1), ax2, ax3],
                placeholder_1[ax0, ax1 - c0, ax2, ax3],
                placeholder[ax0, ax1, ax2, ax3],
            )
            T.writes(T_concat[ax0, ax1, ax2, ax3])
            T_concat[ax0, ax1, ax2, ax3] = T.if_then_else(
                (c0 + c1 + c2) <= ax1,
                placeholder_3[ax0, ax1 - (c0 + c1 + c2), ax2, ax3],
                T.if_then_else(
                    192 <= ax1,
                    placeholder_2[ax0, ax1 - (c0 + c1), ax2, ax3],
                    T.if_then_else(
                        64 <= ax1,
                        placeholder_1[ax0, ax1 - c0, ax2, ax3],
                        placeholder[ax0, ax1, ax2, ax3],
                        dtype="int8",
                    ),
                    dtype="int8",
                ),
                dtype="int8",
            )


def do_test_concat(shapes, use_auto_vu_strategy):
    primfunc = concat_func
    input_param_0, input_param_1, input_param_2, input_param_3, output_param = primfunc.params
    primfunc = primfunc.specialize(
        {
            input_param_0: tir.decl_buffer(shapes[0]),
            input_param_1: tir.decl_buffer(shapes[1]),
            input_param_2: tir.decl_buffer(shapes[2]),
            input_param_3: tir.decl_buffer(shapes[3]),
            output_param: tir.decl_buffer(shapes[4]),
        }
    )
    target = tvm.target.Target("edgex")
    if use_auto_vu_strategy:
        edgex_schedule = naive_vu_schedule(primfunc, is_cpu=False, allow_multi_block=True)
        cpu_schedule = naive_vu_schedule(primfunc, is_cpu=True, allow_multi_block=True)
    else:
        edgex_schedule = schedule_memcpy_style_edgex_impl(primfunc, target)
        cpu_schedule = schedule_memcpy_style_edgex_impl(primfunc, target)
    check_edgex_tir_build("concat", edgex_schedule, cpu_prim_func=cpu_schedule, check_cpu=True)


def do_test_reshape(shapes, use_auto_vu_strategy):
    primfunc = reshape_func
    input_param, output_param = primfunc.params
    primfunc = primfunc.specialize(
        {
            input_param: tir.decl_buffer(shapes[0]),
            output_param: tir.decl_buffer(shapes[1]),
        }
    )
    target = tvm.target.Target("edgex")
    if use_auto_vu_strategy:
        edgex_schedule = naive_vu_schedule(primfunc, is_cpu=False, allow_multi_block=True)
        cpu_schedule = naive_vu_schedule(primfunc, is_cpu=True, allow_multi_block=True)
    else:
        edgex_schedule = schedule_memcpy_style_edgex_impl(primfunc, target)
        cpu_schedule = schedule_memcpy_style_edgex_impl(primfunc, target)
    check_edgex_tir_build("concat", edgex_schedule, cpu_prim_func=cpu_schedule, check_cpu=True)


def do_test_transpose(shapes, use_auto_vu_strategy):
    primfunc = transpose_func
    input_param, output_param = primfunc.params
    primfunc = primfunc.specialize(
        {
            input_param: tir.decl_buffer(shapes[0]),
            output_param: tir.decl_buffer(shapes[1]),
        }
    )
    target = tvm.target.Target("edgex")
    if use_auto_vu_strategy:
        edgex_schedule = naive_vu_schedule(primfunc, is_cpu=False, allow_multi_block=True)
        cpu_schedule = naive_vu_schedule(primfunc, is_cpu=True, allow_multi_block=True)
    else:
        edgex_schedule = schedule_memcpy_style_edgex_impl(primfunc, target)
        cpu_schedule = schedule_memcpy_style_edgex_impl(primfunc, target)
    check_edgex_tir_build("concat", edgex_schedule, cpu_prim_func=cpu_schedule, check_cpu=True)


def do_test_split(shapes, use_auto_vu_strategy):
    primfunc = split_func
    input_param, output_param, output_param_1 = primfunc.params
    primfunc = primfunc.specialize(
        {
            input_param: tir.decl_buffer(shapes[0]),
            output_param: tir.decl_buffer(shapes[1]),
            output_param_1: tir.decl_buffer(shapes[2]),
        }
    )
    target = tvm.target.Target("edgex")
    if use_auto_vu_strategy:
        edgex_schedule = naive_vu_schedule(primfunc, is_cpu=False, allow_multi_block=True)
        cpu_schedule = naive_vu_schedule(primfunc, is_cpu=True, allow_multi_block=True)
    else:
        edgex_schedule = schedule_memcpy_style_edgex_impl(primfunc, target)
        cpu_schedule = schedule_memcpy_style_edgex_impl(primfunc, target)
    check_edgex_tir_build("concat", edgex_schedule, cpu_prim_func=cpu_schedule, check_cpu=True)


def test_concat():
    h = 28
    w = 28
    shapes = [(1, 64, h, w), (1, 128, h, w), (1, 32, h, w), (1, 32, h, w), (1, 256, h, w)]

    # c0 + c1 + c2 + c3 == c_out
    assert shapes[0][1] + shapes[1][1] + shapes[2][1] + shapes[3][1] == shapes[4][1]
    do_test_concat(shapes, use_auto_vu_strategy=True)
    do_test_concat(shapes, use_auto_vu_strategy=False)


def test_reshape():
    shapes = [(1, 8, 10, 48), (1, 8, 10, 3, 16)]
    do_test_reshape(shapes, use_auto_vu_strategy=False)


def test_transpose():
    shapes = [(1, 8, 10, 2, 256), (1, 8, 10, 256, 2)]
    do_test_transpose(shapes, use_auto_vu_strategy=False)


def test_split():
    shapes = [(1, 8, 10, 512), (1, 8, 10, 256), (1, 8, 10, 256)]
    do_test_split(shapes, use_auto_vu_strategy=False)


if __name__ == "__main__":
    test_concat()
    test_reshape()
    test_transpose()
    test_split()
