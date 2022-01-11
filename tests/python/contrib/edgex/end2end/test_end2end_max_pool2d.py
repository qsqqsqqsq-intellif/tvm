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
from tvm import topi
from tvm import relay
from tvm.contrib.edgex.tir.schedule import EdgexSchedule
from tvm.contrib.edgex.topi import naive_vu_schedule
from tvm.contrib.edgex.testing import TempOpStrategy, check_edgex_relay_build
from tvm.contrib.edgex.base.edgexlog import EdgexLog as el


@T.prim_func
def max_pool2d_s2_p1_3_3(data: T.handle, result: T.handle) -> None:
    m = T.var("int32")
    X = T.match_buffer(data, [1, 64, m, m], dtype="int32")
    Y = T.match_buffer(result, [1, 64, m // 2, m // 2], dtype="int32")
    pad = T.alloc_buffer([1, 64, m + 1, m + 1], dtype="int32")
    for nn, cc, hh, ww in T.grid(1, 64, m + 1, m + 1):
        with T.block("pad"):
            n, c, h, w = T.axis.remap("SSSS", [nn, cc, hh, ww])
            pad[n, c, h, w] = T.if_then_else(
                (
                    T.likely(1 <= h, dtype="bool")
                    and T.likely(h < m + 1, dtype="bool")
                    and T.likely(1 <= w, dtype="bool")
                    and T.likely(w < m + 1, dtype="bool")
                ),
                X[n, c, (h - 1), (w - 1)],
                -2147483648,
                dtype="int32",
            )
    for nn, cc, hh, ww, khh, kww in T.grid(1, 64, m // 2, m // 2, 3, 3):
        with T.block("compute"):
            n, c, h, w, kh, kw = T.axis.remap("SSSSRR", [nn, cc, hh, ww, khh, kww])
            with T.init():
                Y[n, c, h, w] = -2147483648
            Y[n, c, h, w] = T.max(Y[n, c, h, w], pad[n, c, h * 2 + kh, w * 2 + kw])


@T.prim_func
def max_pool2d_i8_s2_p1_3_3(data: T.handle, result: T.handle) -> None:
    m = T.var("int32")
    X = T.match_buffer(data, [1, 64, m, m], dtype="int8")
    Y = T.match_buffer(result, [1, 64, m // 2, m // 2], dtype="int8")
    pad = T.alloc_buffer([1, 64, m + 1, m + 1], dtype="int8")
    for nn, cc, hh, ww in T.grid(1, 64, m + 1, m + 1):
        with T.block("pad"):
            n, c, h, w = T.axis.remap("SSSS", [nn, cc, hh, ww])
            pad[n, c, h, w] = T.if_then_else(
                (
                    T.likely(1 <= h, dtype="bool")
                    and T.likely(h < m + 1, dtype="bool")
                    and T.likely(1 <= w, dtype="bool")
                    and T.likely(w < m + 1, dtype="bool")
                ),
                X[n, c, (h - 1), (w - 1)],
                T.int8(-128),
                dtype="int8",
            )
    for nn, cc, hh, ww, khh, kww in T.grid(1, 64, m // 2, m // 2, 3, 3):
        with T.block("compute"):
            n, c, h, w, kh, kw = T.axis.remap("SSSSRR", [nn, cc, hh, ww, khh, kww])
            with T.init():
                Y[n, c, h, w] = T.int8(-128)
            Y[n, c, h, w] = T.max(Y[n, c, h, w], pad[n, c, h * 2 + kh, w * 2 + kw])


def schedule_max_pool2d_s2_p1_3_3(func, is_cpu):
    s = EdgexSchedule(func, debug_mode=False)

    # inline padding
    pad = s.get_block("pad")
    s.compute_inline(pad)

    block = s.get_block("compute")
    X_dm = s.cache_read(block, 1, "dm")
    X_vm = s.cache_read(block, 1, "vm")
    Y_vm = s.cache_write(block, 0, "vm")
    Y_dm = s.cache_write(Y_vm, 0, "dm")

    # schedule loop and buffer order
    def schedule_packed(blks):
        for blk in blks:
            n, c, h, w = s.get_loops(blk)[:4]
            c_o, c_i = s.split(c, factors=[None, 8])
            s.reorder(n, c_o, h, w, c_i)
        for blk in blks:
            bn, bc, bh, bw = s.get_write_buffer_axes(blk, 0)[:4]
            bc_o, bc_i = s.split_buffer(bc, factor=8)
            s.reorder_buffer(bn, bc_o, bh, bw, bc_i)

    schedule_packed([X_dm, X_vm, block, Y_vm])

    # schedule computation
    n, c_o, h, w, c_i, kh, kw = s.get_loops(block)
    s.reorder(n, c_o, h, w, kh, kw, c_i)
    s.unroll(kh)
    s.unroll(kw)
    s.vectorize(c_i)
    s.loop_partition([h, w])

    # tensorize dma intrin
    if not is_cpu:
        s.pragma(s.get_loops(X_dm)[0], "nnp_dma_scope", "eidma")
        s.pragma(s.get_loops(X_vm)[0], "nnp_dma_scope", "vidma")
        s.pragma(s.get_loops(Y_vm)[0], "nnp_dma_scope", "vodma")
        s.pragma(s.get_loops(Y_dm)[0], "nnp_dma_scope", "eodma")
    return s.mod["main"]


def schedule_max_pool2d_s2_p1_3_3_tiling(func, is_cpu):
    s = EdgexSchedule(func, debug_mode=False)

    # inline padding
    pad = s.get_block("pad")
    s.compute_inline(pad)

    block = s.get_block("compute")
    X_dm = s.cache_read(block, 1, "dm")
    X_vm = s.cache_read(block, 1, "vm")
    Y_vm = s.cache_write(block, 0, "vm")
    Y_dm = s.cache_write(Y_vm, 0, "dm")

    # schedule computation
    n, c, h, w, kh, kw = s.get_loops(block)
    c_oo, c_oi, c_i = s.split(c, factors=[None, 2, 8])
    h_o, h_i = s.split(h, factors=[None, 8])
    w_o, w_i = s.split(w, factors=[None, 8])
    s.reorder(n, c_oo, h_o, w_o, c_oi, h_i, w_i, kh, kw, c_i)
    s.compute_at(X_vm, w_o)
    s.compute_at(X_dm, w_o)
    s.reverse_compute_at(Y_vm, w_o)
    s.reverse_compute_at(Y_dm, w_o)
    s.unroll(kh)
    s.unroll(kw)
    s.vectorize(c_i)

    # schedule cache loop and buffer order
    def schedule_packed(blks):
        for blk in blks:
            if blk == block:
                continue
            c, h, w = s.get_loops(blk)[-3:]
            c_o, c_i = s.split(c, factors=[None, 8])
            s.reorder(c_o, h, w, c_i)
        for blk in blks:
            bn, bc, bh, bw = s.get_write_buffer_axes(blk, 0)[:4]
            bc_o, bc_i = s.split_buffer(bc, factor=8)
            s.reorder_buffer(bn, bc_o, bh, bw, bc_i)

    schedule_packed([X_dm, X_vm, block, Y_vm])

    # tensorize dma intrin
    if not is_cpu:
        s.loop_partition([h_o, w_o, h_i, w_i], True)  # cpu llvm is too slow for partitioned loop
        s.pragma(s.get_loops(X_dm)[4], "nnp_dma_scope", "eidma")
        s.pragma(s.get_loops(X_vm)[4], "nnp_dma_scope", "vidma")
        s.pragma(s.get_loops(Y_vm)[4], "nnp_dma_scope", "vodma")
        s.pragma(s.get_loops(Y_dm)[4], "nnp_dma_scope", "eodma")
    return s.mod["main"]


def dispatch_schedule_max_pool2d_s2_p1_3_3(input_shape, use_auto_vu_strategy, dtype: str = "int32"):
    def fschedule(attrs, primfunc, target):
        # create tir function from pattern
        is_cpu = target.kind.name == "llvm"
        if dtype == "int32":
            primfunc = max_pool2d_s2_p1_3_3
        elif dtype == "int8":
            primfunc = max_pool2d_i8_s2_p1_3_3
        else:
            el.e("Not support dtype: %s" % dtype)
        input_param = primfunc.params[0]
        primfunc = primfunc.specialize({input_param: tir.decl_buffer(input_shape)})
        volume = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
        if use_auto_vu_strategy:
            return naive_vu_schedule(primfunc, is_cpu)
        elif volume <= 1024:
            return schedule_max_pool2d_s2_p1_3_3(primfunc, is_cpu)
        else:
            return schedule_max_pool2d_s2_p1_3_3_tiling(primfunc, is_cpu)

    return fschedule


def do_test_vu_maxpool(shape, use_auto_vu_strategy, dtype: str = "int32"):
    x = relay.var("x", dtype=dtype, shape=shape)
    y = relay.nn.max_pool2d(x, pool_size=(3, 3), strides=(2, 2), padding=(1, 1, 1, 1))
    relay_func = relay.Function([x], y)

    # compute from raw te
    def get_raw_te_output(x_np):
        x_te = tvm.te.placeholder(shape, dtype)
        y_te = topi.nn.pool2d(x_te, [3, 3], [2, 2], [1, 1], [1, 1, 1, 1], "max")
        s = tvm.te.create_schedule([y_te.op])
        f = tvm.build(s, [x_te, y_te], "llvm")
        x_nd = tvm.nd.array(x_np)
        y_nd = tvm.nd.array(np.zeros([d.value for d in y_te.shape], dtype))
        f(x_nd, y_nd)
        return y_nd.asnumpy()

    with TempOpStrategy(
        "nn.max_pool2d",
        ["llvm", "edgex"],
        fschedule=dispatch_schedule_max_pool2d_s2_p1_3_3(shape, use_auto_vu_strategy, dtype),
    ):
        check_edgex_relay_build(relay_func, numpy_func=get_raw_te_output)


def test_vu_maxpool_small():
    do_test_vu_maxpool([1, 64, 4, 4], use_auto_vu_strategy=True)
    do_test_vu_maxpool([1, 64, 4, 4], use_auto_vu_strategy=False)


@pytest.mark.edgex_slow
def test_vu_maxpool_tiling():
    do_test_vu_maxpool([1, 64, 224, 224], use_auto_vu_strategy=True)
    do_test_vu_maxpool([1, 64, 224, 224], use_auto_vu_strategy=False)
    do_test_vu_maxpool([1, 64, 112, 112], use_auto_vu_strategy=True, dtype="int8")


if __name__ == "__main__":
    test_vu_maxpool_small()
    test_vu_maxpool_tiling()
