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
from tvm import tir
import tvm.script.tir as T
import numpy as np
from tvm.contrib.edgex.tir.schedule import EdgexSchedule


@T.prim_func
def func_before_blockize(data: T.handle, result: T.handle) -> None:
    X = T.match_buffer(
        data, [1, 64, 32, 32], dtype="int32", elem_offset=0, align=128, offset_factor=1
    )
    Y = T.match_buffer(
        result, [1, 64, 16, 16], dtype="int32", elem_offset=0, align=128, offset_factor=1
    )
    # body
    X_dm = T.alloc_buffer(
        [1, 8, 32, 32, 8], dtype="int32", elem_offset=0, scope="dm", align=128, offset_factor=1
    )
    for nn, cc_outer_outer, hh_outer, ww_outer in T.grid(1, 4, 2, 2):
        for ax1_outer, ax2, ax3, ax1_inner in T.grid(2, 16, 16, 8):
            with T.block("X_dm"):
                v0 = T.axis.spatial(1, 0)
                v1 = T.axis.spatial(64, (((cc_outer_outer * 16) + (ax1_outer * 8)) + ax1_inner))
                v2 = T.axis.spatial(32, (((hh_outer * 16) + ax2) + -1))
                v3 = T.axis.spatial(32, (((ww_outer * 16) + ax3) + -1))

                T.where(((1 <= ((hh_outer * 16) + ax2)) and (1 <= ((ww_outer * 16) + ax3))))
                X_dm[v0, T.floordiv(v1, 8), v2, v3, T.floormod(v1, 8)] = X[v0, v1, v2, v3]


@T.prim_func
def func_after_blockize(data: T.handle, result: T.handle) -> None:
    Y = T.match_buffer(
        result, [1, 64, 16, 16], dtype="int32", elem_offset=0, align=128, offset_factor=1
    )
    X = T.match_buffer(
        data, [1, 64, 32, 32], dtype="int32", elem_offset=0, align=128, offset_factor=1
    )
    # body
    X_dm = T.alloc_buffer(
        [1, 8, 32, 32, 8], dtype="int32", elem_offset=0, scope="dm", align=128, offset_factor=1
    )
    for nn, cc_outer_outer, hh_outer, ww_outer in T.grid(1, 4, 2, 2):
        with T.block("blockized_X_dm"):
            v0, v1o, v2o, v3o = T.axis.remap("SSSS", [nn, cc_outer_outer, hh_outer, ww_outer])
            for ax1_outer, ax2, ax3, ax1_inner in T.grid(2, 16, 16, 8):
                with T.block("X_dm"):
                    T.where(((0 <= (((v2o * 16) + -1) + ax2)) and (0 <= (((v3o * 16) + -1) + ax3))))
                    v1 = T.axis.spatial(64, ((v1o * 16) + ((ax1_outer * 8) + ax1_inner)))
                    v2 = T.axis.spatial(32, (((v2o * 16) + -1) + ax2))
                    v3 = T.axis.spatial(32, (((v3o * 16) + -1) + ax3))

                    X_dm[v0, T.floordiv(v1, 8), v2, v3, T.floormod(v1, 8)] = X[v0, v1, v2, v3]


@pytest.mark.skip("skip because blockize not support yet.")
def test_blockize():
    func = func_before_blockize
    s = EdgexSchedule(func, debug_mode=False)

    X_dm = s.get_block("X_dm")
    s.blockize(s.get_axes(X_dm)[4])

    tvm.ir.assert_structural_equal(func_after_blockize, s.mod["main"])


if __name__ == "__main__":
    test_blockize()
