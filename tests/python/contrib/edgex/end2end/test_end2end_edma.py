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
"""unittest for edma functionalities"""
import pytest
import tvm
import numpy as np
import tvm.testing
from tvm.script import tir as T
from tvm.contrib.edgex.tir.schedule import EdgexSchedule
from tvm.contrib.edgex.testing import check_edgex_tir_build


@T.prim_func
def eidma_load_transpose_nchw_to_nhwc(a: T.handle, b: T.handle) -> None:
    X = T.match_buffer(a, [1, 64, 24, 24], dtype="int8")
    Y = T.match_buffer(b, [1, 24, 24, 64], dtype="int8")
    for n, h, w, c in T.grid(1, 24, 24, 64):
        with T.block("block"):
            nn, hh, ww, cc = T.axis.remap("SSSS", [n, h, w, c])
            Y[nn, hh, ww, cc] = X[nn, cc, hh, ww]


@T.prim_func
def eidma_load_2d_transpose(a: T.handle, b: T.handle) -> None:
    X = T.match_buffer(a, [48, 64], dtype="int32")
    Y = T.match_buffer(b, [64, 48], dtype="int32")
    for ii, jj in T.grid(64, 48):
        with T.block("block"):
            i, j = T.axis.remap("SS", [ii, jj])
            Y[i, j] = X[j, i]


@T.prim_func
def eidma_load_nu_cdhw_to_c1dhwc0_int8(a: T.handle, b: T.handle) -> None:
    X = T.match_buffer(a, [128, 1, 10, 10], dtype="int8")
    Y = T.match_buffer(b, [8, 1, 10, 10, 16], dtype="int8")
    for cc_o, dd, hh, ww, cc_i in T.grid(8, 1, 10, 10, 16):
        with T.block("block"):
            c_o, d, h, w, c_i = T.axis.remap("SSSSS", [cc_o, dd, hh, ww, cc_i])
            Y[c_o, d, h, w, c_i] = X[c_o * 16 + c_i, d, h, w]


@T.prim_func
def eidma_load_nu_cdhw_to_c1dhwc0_int8_bubble(a: T.handle, b: T.handle) -> None:
    X = T.match_buffer(a, [125, 1, 10, 10], dtype="int8")
    Y = T.match_buffer(b, [8, 1, 10, 10, 16], dtype="int8")
    for cc_o, dd, hh, ww, cc_i in T.grid(8, 1, 10, 10, 16):
        with T.block("block"):
            c_o, d, h, w, c_i = T.axis.remap("SSSSS", [cc_o, dd, hh, ww, cc_i])
            Y[c_o, d, h, w, c_i] = T.if_then_else(
                c_o * 16 + c_i < 125, X[c_o * 16 + c_i, d, h, w], T.int8(0), dtype="int8"
            )


@T.prim_func
def eidma_load_nu_cdhw_to_c1dhwc0_float16(a: T.handle, b: T.handle) -> None:
    X = T.match_buffer(a, [128, 1, 10, 10], dtype="float16")
    Y = T.match_buffer(b, [16, 1, 10, 10, 8], dtype="float16")
    for cc_o, dd, hh, ww, cc_i in T.grid(16, 1, 10, 10, 8):
        with T.block("block"):
            c_o, d, h, w, c_i = T.axis.remap("SSSSS", [cc_o, dd, hh, ww, cc_i])
            Y[c_o, d, h, w, c_i] = X[c_o * 8 + c_i, d, h, w]


@T.prim_func
def eodma_store_nu_c1dhwc0_to_dhwc_int8(a: T.handle, b: T.handle) -> None:
    X = T.match_buffer(a, [8, 1, 10, 10, 16], dtype="int8")
    Y = T.match_buffer(b, [1, 10, 10, 128], dtype="int8")
    for dd, hh, ww, cc in T.grid(1, 10, 10, 128):
        with T.block("block"):
            d, h, w, c = T.axis.remap("SSSS", [dd, hh, ww, cc])
            Y[d, h, w, c] = X[c // 16, d, h, w, c % 16]


@T.prim_func
def eodma_store_nu_c1dhwc0_to_dhwc_int8_bubble(a: T.handle, b: T.handle) -> None:
    X = T.match_buffer(a, [8, 1, 1, 1, 16], dtype="int8")
    Y = T.match_buffer(b, [1, 1, 1, 125], dtype="int8")
    for dd, hh, ww, cc in T.grid(1, 1, 1, 125):
        with T.block("block"):
            d, h, w, c = T.axis.remap("SSSS", [dd, hh, ww, cc])
            Y[d, h, w, c] = X[c // 16, d, h, w, c % 16]


@T.prim_func
def eodma_store_nu_c1dhwc0_to_dhwc_float16(a: T.handle, b: T.handle) -> None:
    X = T.match_buffer(a, [16, 1, 10, 10, 8], dtype="float16")
    Y = T.match_buffer(b, [1, 10, 10, 128], dtype="float16")
    for dd, hh, ww, cc in T.grid(1, 10, 10, 128):
        with T.block("block"):
            d, h, w, c = T.axis.remap("SSSS", [dd, hh, ww, cc])
            Y[d, h, w, c] = X[c // 8, d, h, w, c % 8]


@T.prim_func
def eidma_reshape_and_transpose(a: T.handle, b: T.handle) -> None:
    X = T.match_buffer(a, [1, 8, 15, 15, 8], dtype="int32")
    Y = T.match_buffer(b, [1, 64, 15, 15], dtype="int32")
    for ii0, ii1, ii2, ii3 in T.grid(1, 64, 15, 15):
        with T.block("block"):
            i0, i1, i2, i3 = T.axis.remap("SSSS", [ii0, ii1, ii2, ii3])
            Y[i0, i1, i2, i3] = X[i0, i1 // 8, i2, i3, i1 % 8]


@T.prim_func
def ewdma_load_7d(a: T.handle, b: T.handle) -> None:
    X = T.match_buffer(a, [2, 3, 4, 5, 6, 7, 8], dtype="int8")
    Y = T.match_buffer(b, [2, 3, 4, 5, 6, 7, 8], dtype="int8")
    for ii0, ii1, ii2, ii3, ii4, ii5, ii6 in T.grid(2, 3, 4, 5, 6, 7, 8):
        with T.block("block"):
            i0, i1, i2, i3, i4, i5, i6 = T.axis.remap(
                "SSSSSSS", [ii0, ii1, ii2, ii3, ii4, ii5, ii6]
            )
            Y[i0, i1, i2, i3, i4, i5, i6] = X[i0, i1, i2, i3, i4, i5, i6]


def check_edma_result(name, func, tag):
    s = EdgexSchedule(func)
    load_tag = "eidma"
    store_tag = "eodma"
    if tag in {"eidma", "ewdma"}:
        load_tag = tag
        load = s.get_block("block")
        store = s.cache_write(load, 0, "dm")
    else:
        store = s.get_block("block")
        load = s.cache_read(store, 0, "dm")
    s.pragma(s.get_loops(load)[0], "nnp_dma_scope", load_tag)
    s.pragma(s.get_loops(store)[0], "nnp_dma_scope", store_tag)
    check_edgex_tir_build(name, s.mod["main"])


def test_eidma_load_transpose_nchw_to_nhwc():
    check_edma_result(
        "eidma_load_transpose_nchw_to_nhwc", eidma_load_transpose_nchw_to_nhwc, tag="eidma"
    )


def test_eidma_load_2d_transpose():
    check_edma_result("eidma_load_2d_transpose", eidma_load_2d_transpose, tag="eidma")


def test_eodma_store_2d_transpose():
    check_edma_result("eodma_store_2d_transpose", eidma_load_2d_transpose, tag="eodma")


@pytest.mark.skip("tensorize not supported yet")
def test_eidma_load_nu_cdhw_to_c1dhwc0_int8():
    s = tvm.tir.Schedule(eidma_load_nu_cdhw_to_c1dhwc0_int8)
    store = s.get_block("block")
    load = s.cache_write(store, 0, "dm")
    s.tensorize(s.get_axes(load)[0], EidmaLoadIntrin([128, 1, 10, 10], "int8", is_nu_c1dhwc0=True))
    s.tensorize(s.get_axes(store)[0], EodmaStoreIntrin([8, 1, 10, 10, 16], "int8"))
    check_edma_result(
        "eidma_load_nu_cdhw_to_c1dhwc0_int8",
        s,
        [128, 1, 10, 10],
        "int8",
        lambda x: np.transpose(np.reshape(x, [8, 16, 1, 10, 10]), [0, 2, 3, 4, 1]),
    )


@pytest.mark.skip("tensorize not supported yet")
def test_eidma_load_nu_cdhw_to_c1dhwc0_float16():
    s = tvm.tir.Schedule(eidma_load_nu_cdhw_to_c1dhwc0_float16)
    store = s.get_block("block")
    load = s.cache_write(store, 0, "dm")
    s.tensorize(
        s.get_axes(load)[0], EidmaLoadIntrin([128, 1, 10, 10], "float16", is_nu_c1dhwc0=True)
    )
    s.tensorize(s.get_axes(store)[0], EodmaStoreIntrin([16, 1, 10, 10, 8], "float16"))
    check_edma_result(
        "eidma_load_nu_cdhw_to_c1dhwc0_float16",
        s,
        [128, 1, 10, 10],
        "float16",
        lambda x: np.transpose(np.reshape(x, [16, 8, 1, 10, 10]), [0, 2, 3, 4, 1]),
    )


@pytest.mark.skip("tensorize not supported yet")
def test_eidma_load_nu_cdhw_to_c1dhwc0_int8_bubble():
    s = tvm.tir.Schedule(eidma_load_nu_cdhw_to_c1dhwc0_int8_bubble)
    store = s.get_block("block")
    load = s.cache_write(store, 0, "dm")
    s.tensorize(s.get_axes(load)[0], EidmaLoadIntrin([125, 1, 10, 10], "int8", is_nu_c1dhwc0=True))
    s.tensorize(s.get_axes(store)[0], EodmaStoreIntrin([8, 1, 10, 10, 16], "int8"))
    check_edma_result(
        "eidma_load_nu_cdhw_to_c1dhwc0_int8_bubble",
        s,
        [125, 1, 10, 10],
        "int8",
        lambda x: np.transpose(
            np.reshape(np.pad(x, [(0, 3), (0, 0), (0, 0), (0, 0)], "constant"), [8, 16, 1, 10, 10]),
            [0, 2, 3, 4, 1],
        ),
    )


@pytest.mark.skip("tensorize not supported yet")
def test_eodma_store_nu_c1dhwc0_to_dhwc_int8():
    s = tvm.tir.Schedule(eodma_store_nu_c1dhwc0_to_dhwc_int8)
    store = s.get_block("block")
    load = s.cache_read(store, 0, "dm")
    s.tensorize(s.get_axes(load)[0], EidmaLoadIntrin([8, 1, 10, 10, 16], "int8"))
    s.tensorize(
        s.get_axes(store)[0], EodmaStoreIntrin([8, 1, 10, 10, 16], "int8", is_nu_c1dhwc0=True)
    )
    check_edma_result(
        "eodma_store_nu_c1dhwc0_to_dhwc_int8",
        s,
        [8, 1, 10, 10, 16],
        "int8",
        lambda x: np.reshape(np.transpose(x, [1, 2, 3, 0, 4]), [1, 10, 10, 128]),
    )


@pytest.mark.skip("tensorize not supported yet")
def test_eodma_store_nu_c1dhwc0_to_dhwc_float16():
    s = tvm.tir.Schedule(eodma_store_nu_c1dhwc0_to_dhwc_float16)
    store = s.get_block("block")
    load = s.cache_read(store, 0, "dm")
    s.tensorize(s.get_axes(load)[0], EidmaLoadIntrin([16, 1, 10, 10, 8], "float16"))
    s.tensorize(
        s.get_axes(store)[0], EodmaStoreIntrin([16, 1, 10, 10, 8], "float16", is_nu_c1dhwc0=True)
    )
    check_edma_result(
        "eodma_store_nu_c1dhwc0_to_dhwc_float16",
        s,
        [16, 1, 10, 10, 8],
        "float16",
        lambda x: np.reshape(np.transpose(x, [1, 2, 3, 0, 4]), [1, 10, 10, 128]),
    )


@pytest.mark.skip("tensorize not supported yet")
def test_eodma_store_nu_c1dhwc0_to_dhwc_int8_bubble():
    s = tvm.tir.Schedule(eodma_store_nu_c1dhwc0_to_dhwc_int8_bubble)
    store = s.get_block("block")
    load = s.cache_read(store, 0, "dm")
    s.tensorize(s.get_axes(load)[0], EidmaLoadIntrin([8, 1, 1, 1, 16], "int8"))
    s.tensorize(s.get_axes(store)[0], EodmaStoreIntrin([8, 1, 1, 1, 16], "int8", is_nu_c1dhwc0=125))
    input_arr = np.random.randint(0, 128, [8, 1, 1, 1, 16]).astype("int8")
    check_edma_result(
        "eodma_store_nu_c1dhwc0_to_dhwc_int8_bubble",
        s,
        input_arr,
        "int8",
        lambda x: np.reshape(np.transpose(x, [1, 2, 3, 0, 4]), [1, 1, 1, 128])[:, :, :, 0:125],
    )


def test_eidma_load_reshape_and_transpose():
    check_edma_result("eidma_load_reshape_and_transpose", eidma_reshape_and_transpose, tag="eidma")


def test_ewdma_load_7d():
    check_edma_result("ewdma_load_7d_simple", ewdma_load_7d, tag="ewdma")


if __name__ == "__main__":
    test_eidma_load_transpose_nchw_to_nhwc()
    test_eidma_load_2d_transpose()
    test_eodma_store_2d_transpose()
    # todo: tensorize not supported
    # test_eidma_load_nu_cdhw_to_c1dhwc0_int8()
    # test_eidma_load_nu_cdhw_to_c1dhwc0_float16()
    # todo: dirty dm space
    # test_eidma_load_nu_cdhw_to_c1dhwc0_int8_bubble()
    # test_eodma_store_nu_c1dhwc0_to_dhwc_int8()
    # test_eodma_store_nu_c1dhwc0_to_dhwc_float16()
    # test_eodma_store_nu_c1dhwc0_to_dhwc_int8_bubble()
    test_eidma_load_reshape_and_transpose()
    test_ewdma_load_7d()
