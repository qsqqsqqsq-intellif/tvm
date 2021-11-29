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
"""EdgeX loop partition primitive tests."""

import tvm
import tvm.testing
import numpy as np
from tvm.contrib.edgex.tir.schedule import EdgexSchedule
from tvm.script import tir as T


def check_result(mod, partitioned_mod, input_shapes, output_shape):
    input_arrays = []
    for shp in input_shapes:
        arr = tvm.nd.array(np.random.uniform(0, 10, shp).astype("float32"))
        input_arrays.append(arr)
    output = tvm.nd.array(np.random.uniform(0, 10, output_shape).astype("float32"))
    output_expected = tvm.nd.array(np.random.uniform(0, 10, output_shape).astype("float32"))
    f_expected = tvm.build(mod, [], "llvm")
    f = tvm.build(partitioned_mod, [], "llvm")
    f_expected(*input_arrays + [output_expected])
    f(*input_arrays + [output])
    tvm.testing.assert_allclose(output_expected.numpy(), output.numpy(), rtol=1e-5)


@T.prim_func
def add(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (10, 10, 10), "float32")
    B = T.match_buffer(b, (10, 10, 10), "float32")
    C = T.match_buffer(c, (10, 14, 10), "float32")
    for i0, i1, i2 in T.grid(10, 14, 10):
        with T.block("block"):
            ii0, ii1, ii2 = T.axis.remap("SSS", [i0, i1, i2])
            C[ii0, ii1, ii2] = T.if_then_else(
                T.likely(2 <= ii1, dtype="bool") and T.likely(ii1 < 12, dtype="bool"),
                A[ii0, ii1 - 2, ii2] + B[ii0, ii1 - 2, ii2],
                0.0,
                dtype="float32",
            )


@T.prim_func
def expected_partitioned_add(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [10, 10, 10])
    B = T.match_buffer(b, [10, 10, 10])
    C = T.match_buffer(c, [10, 14, 10])
    for i0 in T.serial(0, 10):
        for i1, i2 in T.grid(2, 10):
            with T.block("block"):
                ii0, ii1, ii2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads([])
                T.writes([C[ii0, ii1, ii2]])
                C[ii0, ii1, ii2] = 0.0
        for i1_1, i2_1 in T.grid(10, 10):
            with T.block("block_1"):
                ii0_1 = T.axis.spatial(10, i0)
                ii1_1 = T.axis.spatial((2, 12), i1_1 + 2)
                ii2_1 = T.axis.spatial(10, i2_1)
                T.reads([A[ii0_1, (ii1_1 - 2), ii2_1], B[ii0_1, (ii1_1 - 2), ii2_1]])
                T.writes([C[ii0_1, ii1_1, ii2_1]])
                C[ii0_1, ii1_1, ii2_1] = A[ii0_1, (ii1_1 - 2), ii2_1] + B[ii0_1, (ii1_1 - 2), ii2_1]
        for i1_2, i2_2 in T.grid(2, 10):
            with T.block("block_2"):
                ii0_2 = T.axis.spatial(10, i0)
                ii1_2 = T.axis.spatial((12, 14), i1_2 + 12)
                ii2_2 = T.axis.spatial(10, i2_2)
                T.reads([])
                T.writes([C[ii0_2, ii1_2, ii2_2]])
                C[ii0_2, ii1_2, ii2_2] = 0.0


@T.prim_func
def padded_pool2d(data: T.handle, result: T.handle) -> None:
    X = T.match_buffer(data, [224, 224])
    Y = T.match_buffer(result, [112, 112])
    pad = T.alloc_buffer([225, 225])
    for hh, ww in T.grid(225, 225):
        with T.block("pad"):
            h, w = T.axis.remap("SS", [hh, ww])
            pad[h, w] = T.if_then_else(
                (
                    T.likely(1 <= h, dtype="bool")
                    and T.likely(h < 225, dtype="bool")
                    and T.likely(1 <= w, dtype="bool")
                    and T.likely(w < 225, dtype="bool")
                ),
                X[(h - 1), (w - 1)],
                0.0,
                dtype="float32",
            )
    for hh, ww, khh, kww in T.grid(112, 112, 3, 3):
        with T.block("compute"):
            h, w, kh, kw = T.axis.remap("SSRR", [hh, ww, khh, kww])
            with T.init():
                Y[h, w] = 0.0
            Y[h, w] = T.max(Y[h, w], pad[h * 2 + kh, w * 2 + kw])


@T.prim_func
def expected_partitioned_padded_pool2d_partial(data: T.handle, result: T.handle) -> None:
    Y = T.match_buffer(result, [112, 112])
    X = T.match_buffer(data, [224, 224])
    # h = 0
    for hh in T.serial(0, 1):
        # h = 0, w = 0
        for ww, khh, kww in T.grid(1, 3, 3):
            with T.block():
                h, w, kh, kw = T.axis.remap("SSRR", [hh, ww, khh, kww])
                with T.init():
                    Y[h, w] = 0.0
                Y[h, w] = T.max(
                    Y[h, w],
                    T.if_then_else(
                        T.likely(1 <= kh, dtype="bool") and T.likely(1 <= kw, dtype="bool"),
                        X[h * 2 + kh - 1, w * 2 + kw - 1],
                        0.0,
                        dtype="float32",
                    ),
                )
        # h = 0, w >= 1
        for ww_1, khh_1, kww_1 in T.grid(111, 3, 3):
            with T.block():
                h_1 = T.axis.spatial(1, hh)
                w_1 = T.axis.spatial((1, 112), ww_1 + 1)
                kh_1, kw_1 = T.axis.remap("RR", [khh_1, kww_1])
                with T.init():
                    Y[h_1, w_1] = 0.0
                Y[h_1, w_1] = T.max(
                    Y[h_1, w_1],
                    T.if_then_else(
                        T.likely(1 <= kh_1, dtype="bool"),
                        X[h_1 * 2 + kh_1 - 1, w_1 * 2 + kw_1 - 1],
                        0.0,
                        dtype="float32",
                    ),
                )
    # h > 0
    for hh_1 in T.serial(0, 111):
        # h > 0, w = 0
        for ww_2, khh_2, kww_2 in T.grid(1, 3, 3):
            with T.block():
                h_2 = T.axis.spatial((1, 112), hh_1 + 1)
                w_2, kh_2, kw_2 = T.axis.remap("SRR", [ww_2, khh_2, kww_2])
                with T.init():
                    Y[h_2, w_2] = 0.0
                Y[h_2, w_2] = T.max(
                    Y[h_2, w_2],
                    T.if_then_else(
                        T.likely(1 <= kw_2, dtype="bool"),
                        X[h_2 * 2 + kh_2 - 1, w_2 * 2 + kw_2 - 1],
                        0.0,
                        dtype="float32",
                    ),
                )
        # h > 0, w > 0
        for ww_3, khh_3, kww_3 in T.grid(111, 3, 3):
            with T.block():
                h_3 = T.axis.spatial((1, 112), hh_1 + 1)
                w_3 = T.axis.spatial((1, 112), ww_3 + 1)
                kh_3, kw_3 = T.axis.remap("RR", [khh_3, kww_3])
                with T.init():
                    Y[h_3, w_3] = 0.0
                Y[h_3, w_3] = T.max(Y[h_3, w_3], X[h_3 * 2 + kh_3 - 1, w_3 * 2 + kw_3 - 1])


@T.prim_func
def expected_partitioned_padded_pool2d_full(data: T.handle, result: T.handle) -> None:
    Y = T.match_buffer(result, [112, 112], elem_offset=0, align=128, offset_factor=1)
    X = T.match_buffer(data, [224, 224], elem_offset=0, align=128, offset_factor=1)
    for hh in T.serial(0, 1):
        for ww in T.serial(0, 1):
            for khh in T.serial(0, 1):
                for kww in T.serial(0, 1):
                    # h = 0, w = 0, kh = 0, kw = 0
                    with T.block():
                        h, w, kh, kw = T.axis.remap("SSRR", [hh, ww, khh, kww])
                        with T.init():
                            Y[h, w] = 0.0
                        Y[h, w] = T.max(Y[h, w], 0.0)
                for kww_1 in T.serial(0, 2):
                    # h = 0, w = 0, kh = 0, kw >= 1
                    with T.block():
                        h_1, w_1, kh_1 = T.axis.remap("SSR", [hh, ww, khh])
                        kw_1 = T.axis.reduce((1, 3), kww_1 + 1)
                        Y[h_1, w_1] = T.max(Y[h_1, w_1], 0.0)
            for khh_1 in T.serial(0, 2):
                for kww_2 in T.serial(0, 1):
                    # h = 0, w = 0, kh >= 1, kw = 0
                    with T.block():
                        h_2, w_2 = T.axis.remap("SS", [hh, ww])
                        kh_2 = T.axis.reduce((1, 3), khh_1 + 1)
                        kw_2 = T.axis.reduce(1, kww_2)
                        Y[h_2, w_2] = T.max(Y[h_2, w_2], 0.0)
                for kww_3 in T.serial(0, 2):
                    # h = 0, w = 0, kh >= 1, kw >= 1
                    with T.block():
                        h_3, w_3 = T.axis.remap("SS", [hh, ww])
                        kh_3 = T.axis.reduce((1, 3), khh_1 + 1)
                        kw_3 = T.axis.reduce((1, 3), kww_3 + 1)
                        Y[h_3, w_3] = T.max(Y[h_3, w_3], X[h_3 * 2 + kh_3 - 1, w_3 * 2 + kw_3 - 1])
        for ww_1 in T.serial(0, 111):
            for khh_2, kww_4 in T.grid(1, 3):
                # h = 0, w >= 1, kh = 0
                with T.block():
                    h_4 = T.axis.spatial(1, hh)
                    w_4 = T.axis.spatial((1, 112), ww_1 + 1)
                    kh_4, kw_4 = T.axis.remap("RR", [khh_2, kww_4])
                    with T.init():
                        Y[h_4, w_4] = 0.0
                    Y[h_4, w_4] = T.max(Y[h_4, w_4], 0.0)
            for khh_3, kww_5 in T.grid(2, 3):
                # h = 0, w >= 1, kh >= 1
                with T.block():
                    h_5 = T.axis.spatial(1, hh)
                    w_5 = T.axis.spatial((1, 112), ww_1 + 1)
                    kh_5 = T.axis.reduce((1, 3), khh_3 + 1)
                    kw_5 = T.axis.reduce(3, kww_5)
                    Y[h_5, w_5] = T.max(Y[h_5, w_5], X[h_5 * 2 + kh_5 - 1, w_5 * 2 + kw_5 - 1])
    for hh_1 in T.serial(0, 111):
        for ww_2, khh_4 in T.grid(1, 3):
            for kww_6 in T.serial(0, 1):
                # h >= 1, w = 0, kw = 0
                with T.block():
                    h_6 = T.axis.spatial((1, 112), hh_1 + 1)
                    w_6, kh_6, kw_6 = T.axis.remap("SRR", [ww_2, khh_4, kww_6])
                    with T.init():
                        Y[h_6, w_6] = 0.0
                    Y[h_6, w_6] = T.max(Y[h_6, w_6], 0.0)
            for kww_7 in T.serial(0, 2):
                # h >= 1, w = 0, kw >= 1
                with T.block():
                    h_7 = T.axis.spatial((1, 112), hh_1 + 1)
                    w_7, kh_7 = T.axis.remap("SR", [ww_2, khh_4])
                    kw_7 = T.axis.reduce((1, 3), kww_7 + 1)
                    Y[h_7, w_7] = T.max(Y[h_7, w_7], X[h_7 * 2 + kh_7 - 1, w_7 * 2 + kw_7 - 1])
        for ww_3, khh_5, kww_8 in T.grid(111, 3, 3):
            # h >= 1, w >= 1
            with T.block():
                h_8 = T.axis.spatial((1, 112), hh_1 + 1)
                w_8 = T.axis.spatial((1, 112), ww_3 + 1)
                kh_8, kw_8 = T.axis.remap("RR", [khh_5, kww_8])
                with T.init():
                    Y[h_8, w_8] = 0.0
                Y[h_8, w_8] = T.max(Y[h_8, w_8], X[h_8 * 2 + kh_8 - 1, w_8 * 2 + kw_8 - 1])


def test_schedule_loop_partition_single_axis():
    s = EdgexSchedule(add, debug_mode=True)
    blk = s.get_block("block")
    axes = s.get_loops(blk)
    s.loop_partition([axes[1]], lazy=False)
    tvm.ir.assert_structural_equal(expected_partitioned_add, s.mod["main"])
    check_result(add, s.mod["main"], [[10, 10, 10], [10, 10, 10]], [10, 14, 10])


def test_schedule_loop_partition_padding():
    def do_test(partition_axes, expected_partitioned):
        s = EdgexSchedule(padded_pool2d, debug_mode=True)
        pad = s.get_block("pad")
        s.compute_inline(pad)
        block = s.get_block("compute")
        axes = s.get_loops(block)
        s.loop_partition([axes[i] for i in partition_axes], lazy=False)
        tvm.ir.assert_structural_equal(expected_partitioned, s.mod["main"])
        check_result(padded_pool2d, s.mod["main"], [[224, 224]], [112, 112])

    # partition [h, w]
    do_test([0, 1], expected_partitioned_padded_pool2d_partial)

    # partition [h, w, kh, kw]
    do_test([0, 1, 2, 3], expected_partitioned_padded_pool2d_full)


if __name__ == "__main__":
    test_schedule_loop_partition_single_axis()
    test_schedule_loop_partition_padding()
