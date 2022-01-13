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
def elewise_after_pool2d_s2_p1_2_2(data: T.handle, result: T.handle) -> None:
    m = T.var("int32")
    X = T.match_buffer(data, [1, 64, m, m], dtype="int32")
    Y = T.match_buffer(result, [1, 64, (m + 1) // 2, (m + 1) // 2], dtype="int32")

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
                0,
                dtype="int32",
            )
    tmp_buffer = T.alloc_buffer([1, 64, (m + 1) // 2, (m + 1) // 2], dtype="int32")
    for nn, cc, hh, ww, khh, kww in T.grid(1, 64, (m + 1) // 2, (m + 1) // 2, 2, 2):
        with T.block("compute"):
            n, c, h, w, kh, kw = T.axis.remap("SSSSRR", [nn, cc, hh, ww, khh, kww])
            with T.init():
                tmp_buffer[n, c, h, w] = 0
            tmp_buffer[n, c, h, w] = tmp_buffer[n, c, h, w] + pad[n, c, h * 2 + kh, w * 2 + kw]
    for nn, cc, hh, ww in T.grid(1, 64, (m + 1) // 2, (m + 1) // 2):
        with T.block("elewise_add"):
            n, c, h, w = T.axis.remap("SSSS", [nn, cc, hh, ww])
            Y[n, c, h, w] = tmp_buffer[n, c, h, w] + 0xF


def schedule_elewise_after_pool2d_s2_p1_2_2(func, is_cpu):
    s = EdgexSchedule(func, debug_mode=True)

    # inline padding
    pad = s.get_block("pad")
    s.compute_inline(pad)

    block = s.get_block("compute")
    elewise_add = s.get_block("elewise_add")

    # compute block split reoder
    n, c, h, w = s.get_loops(block)[:4]
    c_o, c_i = s.split(c, factors=[None, 8])
    s.reorder(n, c_o, h, w, c_i)
    n, c_o, h, w, c_i, kh, kw = s.get_loops(block)
    s.reorder(n, c_o, h, w, kh, kw, c_i)

    n, c_o, h, w, kh, kw, c_i = s.get_loops(block)
    s.unroll(kh)
    s.unroll(kw)
    s.vectorize(c_i)
    s.loop_partition([h, w])

    # compute_at elewise_add at compute block's w axes
    n, c_o, h, w, c_i, kh, kw = s.get_loops(block)
    s.reverse_compute_at(elewise_add, w)

    buf = s.get_sref(block).stmt.writes[0].buffer
    tmp_buf = tvm.tir.decl_buffer(buf.shape, buf.dtype, buf.name, scope="vm")
    s.replace_buffer(block, buf, tmp_buf)

    X_dm = s.cache_read(block, 1, "dm")
    X_vm = s.cache_read(block, 1, "vm")
    Y_vm = s.cache_write(elewise_add, 0, "vm")
    Y_dm = s.cache_write(Y_vm, 0, "dm")

    # schedule loop and buffer order
    def schedule_loop_order_packed(blks):
        for blk in blks:
            n, c, h, w = s.get_loops(blk)[:4]
            c_o, c_i = s.split(c, factors=[None, 8])
            s.reorder(n, c_o, h, w, c_i)

    def schedule_buffer_order_packed(blks):
        for blk in blks:
            bn, bc, bh, bw = s.get_write_buffer_axes(blk, 0)[:4]
            bc_o, bc_i = s.split_buffer(bc, factor=8)
            s.reorder_buffer(bn, bc_o, bh, bw, bc_i)

    schedule_loop_order_packed([X_dm, X_vm, Y_vm])
    schedule_buffer_order_packed([X_dm, X_vm, Y_vm, elewise_add, block])

    # tensorize dma intrin
    if not is_cpu:
        s.pragma(s.get_loops(X_dm)[0], "nnp_dma_scope", "eidma")
        s.pragma(s.get_loops(X_vm)[0], "nnp_dma_scope", "vidma")
        s.pragma(s.get_loops(Y_vm)[0], "nnp_dma_scope", "vodma")
        s.pragma(s.get_loops(Y_dm)[0], "nnp_dma_scope", "eodma")

    print(s.mod["main"].script())
    return s.mod["main"]


def schedule_elewise_after_pool2d_s2_p1_2_2_tiling(func, is_cpu):
    s = EdgexSchedule(func, debug_mode=False)

    # inline padding
    pad = s.get_block("pad")
    s.compute_inline(pad)

    block = s.get_block("compute")
    elewise_add = s.get_block("elewise_add")

    n, c, h, w, kh, kw = s.get_loops(block)
    c_oo, c_oi, c_i = s.split(c, factors=[None, 2, 8])
    h_o, h_i = s.split(h, factors=[None, 8])
    w_o, w_i = s.split(w, factors=[None, 8])
    s.reorder(n, c_oo, h_o, w_o, c_oi, h_i, w_i, kh, kw, c_i)
    s.vectorize(c_i)

    s.reverse_compute_at(elewise_add, c_i)

    buf = s.get_sref(block).stmt.writes[0].buffer
    tmp_buf = tvm.tir.decl_buffer(buf.shape, buf.dtype, buf.name, scope="vm")
    s.replace_buffer(block, buf, tmp_buf)

    X_dm = s.cache_read(block, 1, "dm")
    X_vm = s.cache_read(block, 1, "vm")
    Y_vm = s.cache_write(elewise_add, 0, "vm")
    Y_dm = s.cache_write(Y_vm, 0, "dm")

    s.compute_at(X_vm, w_o)
    s.compute_at(X_dm, w_o)
    s.reverse_compute_at(Y_vm, w_o)
    s.reverse_compute_at(Y_dm, w_o)
    s.unroll(kh)
    s.unroll(kw)

    # schedule cache loop and buffer order
    def schedule_packed(blks):
        for blk in blks:
            if blk == elewise_add:
                continue
            c, h, w = s.get_loops(blk)[-3:]
            c_o, c_i = s.split(c, factors=[None, 8])
            s.reorder(c_o, h, w, c_i)
        for blk in blks:
            bn, bc, bh, bw = s.get_write_buffer_axes(blk, 0)[:4]
            bc_o, bc_i = s.split_buffer(bc, factor=8)
            s.reorder_buffer(bn, bc_o, bh, bw, bc_i)

    schedule_packed([X_dm, X_vm, elewise_add, Y_vm])
    bn, bc, bh, bw = s.get_write_buffer_axes(block, 0)[:4]
    bc_o, bc_i = s.split_buffer(bc, factor=8)
    s.reorder_buffer(bn, bc_o, bh, bw, bc_i)

    # tensorize dma intrin
    if not is_cpu:
        s.loop_partition([h_o, w_o, h_i, w_i], True)  # cpu llvm is too slow for partitioned loop
        s.pragma(s.get_loops(X_dm)[4], "nnp_dma_scope", "eidma")
        s.pragma(s.get_loops(X_vm)[4], "nnp_dma_scope", "vidma")
        s.pragma(s.get_loops(Y_vm)[4], "nnp_dma_scope", "vodma")
        s.pragma(s.get_loops(Y_dm)[4], "nnp_dma_scope", "eodma")
    return s.mod["main"]


def do_test_vu_elewise_after_pool2d(shape, use_auto_vu_strategy):
    # compute with numpy
    def get_numpy_output(x_np):
        pad_np = np.pad(x_np, ((0, 0), (0, 0), (1, 0), (1, 0)), "constant")
        x_shape = x_np.shape
        out_shape = [x_shape[0], x_shape[1], (x_shape[2] + 1) // 2, (x_shape[3] + 1) // 2]
        y_np = np.zeros(out_shape, dtype=np.int32)
        for n in range(out_shape[0]):
            for c in range(out_shape[1]):
                for h in range(out_shape[2]):
                    for w in range(out_shape[3]):
                        for kh in range(2):
                            for kw in range(2):
                                y_np[n, c, h, w] = (
                                    y_np[n, c, h, w] + pad_np[n, c, h * 2 + kh, w * 2 + kw]
                                )
        return y_np + 0xF

    primfunc = elewise_after_pool2d_s2_p1_2_2
    input_param = primfunc.params[0]
    primfunc = primfunc.specialize({input_param: tir.decl_buffer(shape)})
    volume = shape[0] * shape[1] * shape[2] * shape[3]
    if use_auto_vu_strategy:
        s = naive_vu_schedule(primfunc, allow_multi_block=True)
    elif volume <= 1024:
        s = schedule_elewise_after_pool2d_s2_p1_2_2(primfunc, False)
    else:
        s = schedule_elewise_after_pool2d_s2_p1_2_2_tiling(primfunc, False)
    check_edgex_tir_build("elewise_after_pool2d", s, get_numpy_output, check_cpu=True)


@pytest.mark.edgex_slow
def test_vu_elewise_after_pool2d_small():
    do_test_vu_elewise_after_pool2d([1, 64, 3, 3], use_auto_vu_strategy=True)
    do_test_vu_elewise_after_pool2d([1, 64, 3, 3], use_auto_vu_strategy=False)


@pytest.mark.edgex_slow
def test_vu_elewise_after_pool2d_tiling():
    do_test_vu_elewise_after_pool2d([1, 64, 32, 32], use_auto_vu_strategy=True)
    do_test_vu_elewise_after_pool2d([1, 64, 32, 32], use_auto_vu_strategy=False)


if __name__ == "__main__":
    test_vu_elewise_after_pool2d_small()
    test_vu_elewise_after_pool2d_tiling()
