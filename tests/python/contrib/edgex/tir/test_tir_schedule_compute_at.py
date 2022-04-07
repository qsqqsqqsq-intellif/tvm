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
# pylint: disable=missing-function-docstring,missing-module-docstring
import pytest
import tvm
import tvm.testing
import tvm.script.tir as T
from tvm.contrib.edgex.tir.schedule import EdgexSchedule


@T.prim_func
def reshape(
    placeholder_0: T.Buffer[(1, 3, 80, 80, 85), "float16"],
    T_reshape: T.Buffer[(1, 19200, 85), "float16"],
) -> None:
    placeholder_0_dm = T.alloc_buffer([1, 3, 80, 80, 85], dtype="float16", scope="dm")
    for ax0, ax1, ax2, ax3, ax4 in T.grid(1, 3, 80, 80, 85):
        with T.block("placeholder_0_dm"):
            v0, v1, v2, v3, v4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(placeholder_0[v0, v1, v2, v3, v4])
            T.writes(placeholder_0_dm[v0, v1, v2, v3, v4])
            placeholder_0_dm[v0, v1, v2, v3, v4] = placeholder_0[v0, v1, v2, v3, v4]
    for i1_0, i0, i1_1, i2 in T.grid(2, 1, 9600, 85):
        with T.block("T_reshape"):
            ax0 = T.axis.spatial(1, 0)
            ax1 = T.axis.spatial(19200, i1_0 * 9600 + i1_1)
            ax2 = T.axis.spatial(85, i2)
            T.reads(
                placeholder_0_dm[
                    0,
                    (ax2 // 85 + ax1) % 19200 // 6400,
                    (ax2 // 85 + ax1) % 6400 // 80,
                    (ax2 // 85 + ax1) % 80,
                    ax2 % 85,
                ]
            )
            T.writes(T_reshape[ax0, ax1, ax2])
            T_reshape[ax0, ax1, ax2] = placeholder_0_dm[
                0,
                (ax2 // 85 + ax1) % 19200 // 6400,
                (ax2 // 85 + ax1) % 6400 // 80,
                (ax2 // 85 + ax1) % 80,
                ax2 % 85,
            ]


@T.prim_func
def expected_compute_at_reshape(
    placeholder_0: T.Buffer[(1, 3, 80, 80, 85), "float16"],
    T_reshape: T.Buffer[(1, 19200, 85), "float16"],
) -> None:
    placeholder_0_dm = T.alloc_buffer([1, 3, 80, 80, 85], dtype="float16", scope="dm")
    for i1_0 in T.serial(2):
        for ax0, ax1, ax2, ax3 in T.grid(2, 80, 80, 85):
            with T.block("placeholder_0_dm"):
                v0 = T.axis.spatial(1, 0)
                v1 = T.axis.spatial(3, i1_0 * 9600 // 6400 + ax0)
                v2, v3, v4 = T.axis.remap("SSS", [ax1, ax2, ax3])
                T.reads(placeholder_0[v0, v1, v2, v3, v4])
                T.writes(placeholder_0_dm[v0, v1, v2, v3, v4])
                placeholder_0_dm[v0, v1, v2, v3, v4] = placeholder_0[v0, v1, v2, v3, v4]
        for i0, i1_1, i2 in T.grid(1, 9600, 85):
            with T.block("T_reshape"):
                ax0 = T.axis.spatial(1, 0)
                ax1 = T.axis.spatial(19200, i1_0 * 9600 + i1_1)
                ax2 = T.axis.spatial(85, i2)
                T.reads(
                    placeholder_0_dm[
                        0,
                        (ax2 // 85 + ax1) % 19200 // 6400,
                        (ax2 // 85 + ax1) % 6400 // 80,
                        (ax2 // 85 + ax1) % 80,
                        ax2 % 85,
                    ]
                )
                T.writes(T_reshape[ax0, ax1, ax2])
                T_reshape[ax0, ax1, ax2] = placeholder_0_dm[
                    0,
                    (ax2 // 85 + ax1) % 19200 // 6400,
                    (ax2 // 85 + ax1) % 6400 // 80,
                    (ax2 // 85 + ax1) % 80,
                    ax2 % 85,
                ]


@pytest.mark.skip("skip because compute_at not support in this case.")
def test_compute_at():
    func = reshape
    s = EdgexSchedule(func, debug_mode=False)

    main_blocks = s.get_child_blocks(s.get_block("root"))
    cache_read_blk = main_blocks[0]
    compute_blk = main_blocks[1]
    loops = s.get_loops(compute_blk)
    s.compute_at(cache_read_blk, loops[0])
    print(s.mod.script())

    tvm.ir.assert_structural_equal(expected_compute_at_reshape, s.mod["main"])


if __name__ == "__main__":
    test_compute_at()
