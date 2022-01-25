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


def schedule_concat(func, is_cpu):
    s = EdgexSchedule(func, debug_mode=False)

    concat = s.get_child_blocks(s.get_block("root"))[0]

    n, c, h, w = s.get_loops(concat)

    s.loop_partition(c)

    eidma_blocks = []

    concat_stmt = s.get_sref(concat).stmt

    for i in range(len(concat_stmt.reads)):
        eidma = s.cache_read(concat, i, "dm")
        eidma_blocks.append(eidma)

    # tensorize dma intrin
    if not is_cpu:
        for blk in eidma_blocks:
            s.pragma(s.get_loops(blk)[0], "nnp_dma_scope", "eidma")
        # tensorize concat block
        s.pragma(s.get_loops(concat)[0], "nnp_dma_scope", "eodma")

    return s.mod["main"]


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
    if use_auto_vu_strategy:
        s = naive_vu_schedule(primfunc, is_cpu=False, allow_multi_block=True)
    else:
        s = schedule_concat(primfunc, False)
    check_edgex_tir_build("concat", s, check_cpu=True)


def test_concat():
    h = 28
    w = 28
    shapes = [(1, 64, h, w), (1, 128, h, w), (1, 32, h, w), (1, 32, h, w), (1, 256, h, w)]

    # c0 + c1 + c2 + c3 == c_out
    assert shapes[0][1] + shapes[1][1] + shapes[2][1] + shapes[3][1] == shapes[4][1]
    do_test_concat(shapes, use_auto_vu_strategy=False)


if __name__ == "__main__":
    test_concat()
