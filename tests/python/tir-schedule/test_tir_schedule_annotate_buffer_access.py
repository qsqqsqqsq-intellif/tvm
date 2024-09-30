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
import tvm
import tvm.testing
from tvm import tir
from tvm.script import tir as T
from tvm.tir.schedule.testing import (
    verify_trace_roundtrip,
    assert_structural_equal_ignore_global_symbol,
)
from tvm.tir.stmt_functor import post_order_visit


def test_annotate_read_buffer_access():
    @T.prim_func
    def before(A: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")):
        B = T.alloc_buffer((128, 128), "float32")
        for i, j in T.grid(128, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 2.0
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + 1.0

    @T.prim_func
    def expected(A: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")):
        B = T.alloc_buffer((128, 128), "float32")
        for i, j in T.grid(128, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi - 1 : vi - 1 + 2, vj - 1 : vj - 1 + 2])
                T.writes(B[vi, vj])
                T.block_attr({"explicit_read_region": [0]})
                B[vi, vj] = A[vi, vj] * 2.0
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + 1.0

    sch = tir.Schedule(before, debug_mask="all")
    block = sch.get_block("B")
    sch.annotate_buffer_access(
        block, 0, "read", lambda vi, vj: ((vi - 1, vi + 1), (vj - 1, vj + 1))
    )
    assert_structural_equal_ignore_global_symbol(sch.mod["main"], expected)
    verify_trace_roundtrip(sch=sch, mod=before)


def test_annotate_write_buffer_access():
    @T.prim_func
    def before(A: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")):
        B = T.alloc_buffer((128, 128), "float32")
        for i, j in T.grid(128, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 2.0
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + 1.0

    @T.prim_func
    def expected(A: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")):
        B = T.alloc_buffer((128, 128), "float32")
        for i, j in T.grid(128, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi, vj])
                T.writes(B[vi : vi + 2, vj : vj + 2])
                T.block_attr({"explicit_write_region": [0]})
                B[vi, vj] = A[vi, vj] * 2.0
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + 1.0

    sch = tir.Schedule(before, debug_mask="all")
    block = sch.get_block("B")
    sch.annotate_buffer_access(block, 0, "write", lambda vi, vj: ((vi, vi + 2), (vj, vj + 2)))
    assert_structural_equal_ignore_global_symbol(sch.mod["main"], expected)
    verify_trace_roundtrip(sch=sch, mod=before)


def test_annotate_buffer_access_for_resize():
    # fmt: off
    @T.prim_func
    def resize_before(x: T.Buffer((1, 1, 32, 32), "float16"), resize: T.Buffer((1, 1, 16, 16), "float16")):
        for i0, i1, i2, i3 in T.grid(1, 1, 16, 16):
            with T.block("resize"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(x[v_i0, v_i1, 0:32, 0:32])
                T.writes(resize[v_i0, v_i1, v_i2, v_i3])
                resize[v_i0, v_i1, v_i2, v_i3] = T.Cast("float16", T.Cast("float32", x[v_i0, v_i1, T.max(T.min(T.Cast("int32", T.floor((T.Cast("float32", v_i2) + T.float32(0.5)) * T.float32(2) - T.float32(0.5) + T.float32(1.0000000000000001e-05))), 31), 0), T.max(T.min(T.Cast("int32", T.floor((T.Cast("float32", v_i3) + T.float32(0.5)) * T.float32(2) - T.float32(0.5) + T.float32(1.0000000000000001e-05))), 31), 0)]))

    @T.prim_func
    def resize_expected(x: T.Buffer((1, 1, 32, 32), "float16"), resize: T.Buffer((1, 1, 16, 16), "float16")):
        for i0, i1, i2, i3 in T.grid(1, 1, 16, 16):
            with T.block("resize"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(x[v_i0, v_i1, v_i2 * 2 - 3:v_i2 * 2 + 3, v_i3 * 2 - 3:v_i3 * 2 + 3])
                T.writes(resize[v_i0, v_i1, v_i2, v_i3])
                T.block_attr({"explicit_read_region": [0]})
                resize[v_i0, v_i1, v_i2, v_i3] = T.Cast("float16", T.Cast("float32", x[v_i0, v_i1, T.max(T.min(T.Cast("int32", T.floor((T.Cast("float32", v_i2) + T.float32(0.5)) * T.float32(2) - T.float32(0.5) + T.float32(1.0000000000000001e-05))), 31), 0), T.max(T.min(T.Cast("int32", T.floor((T.Cast("float32", v_i3) + T.float32(0.5)) * T.float32(2) - T.float32(0.5) + T.float32(1.0000000000000001e-05))), 31), 0)]))
    # fmt: on
    sch = tir.Schedule(resize_before, debug_mask="all")
    block = sch.get_block("resize")
    sch.annotate_buffer_access(
        block,
        0,
        "read",
        gen_new_ranges=lambda v_i0, v_i1, v_i2, v_i3: [
            v_i0,
            v_i1,
            (v_i2 * 2 - 3, v_i2 * 2 + 3),
            (v_i3 * 2 - 3, v_i3 * 2 + 3),
        ],
    )
    assert_structural_equal_ignore_global_symbol(sch.mod["main"], resize_expected)
    verify_trace_roundtrip(sch=sch, mod=resize_before)


def test_annotate_buffer_access_read_and_write():
    @T.prim_func
    def before(A: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")):
        B = T.alloc_buffer((128, 128), "float32")
        for i, j in T.grid(128, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi, vj])
                T.writes(B[vi, vj])
                B[vi, vj] = A[vi, vj] * 2.0
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(B[vi, vj])
                T.writes(C[vi, vj])
                C[vi, vj] = B[vi, vj] + 1.0

    @T.prim_func
    def expected(A: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")):
        B = T.alloc_buffer((128, 128), "float32")
        for i, j in T.grid(128, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi - 1 : vi + 2, vj - 1 : vj + 2])
                T.writes(B[vi : vi + 2, vj : vj + 2])
                T.block_attr({"explicit_read_region": [0], "explicit_write_region": [0]})
                B[vi, vj] = A[vi, vj] * 2.0
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(B[vi, vj])
                T.writes(C[vi, vj])
                C[vi, vj] = B[vi, vj] + 1.0

    sch = tir.Schedule(before, debug_mask="all")
    block = sch.get_block("B")

    sch.annotate_buffer_access(
        block, 0, "read", lambda vi, vj: ((vi - 1, vi + 2), (vj - 1, vj + 2))
    )

    sch.annotate_buffer_access(block, 0, "write", lambda vi, vj: ((vi, vi + 2), (vj, vj + 2)))

    assert_structural_equal_ignore_global_symbol(sch.mod["main"], expected)
    verify_trace_roundtrip(sch=sch, mod=before)


def test_double_annotate_buffer_access_read():
    @T.prim_func
    def before(A: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")):
        B = T.alloc_buffer((128, 128), "float32")
        for i, j in T.grid(128, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi, vj])
                T.writes(B[vi, vj])
                B[vi, vj] = A[vi, vj] * 2.0
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(B[vi, vj])
                T.writes(C[vi, vj])
                C[vi, vj] = B[vi, vj] + 1.0

    @T.prim_func
    def expected(A: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")):
        B = T.alloc_buffer((128, 128), "float32")
        for i, j in T.grid(128, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi - 2 : vi + 3, vj - 2 : vj + 3])
                T.writes(B[vi, vj])
                T.block_attr({"explicit_read_region": [0]})
                B[vi, vj] = A[vi, vj] * 2.0
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(B[vi, vj])
                T.writes(C[vi, vj])
                C[vi, vj] = B[vi, vj] + 1.0

    sch = tir.Schedule(before, debug_mask="all")
    block = sch.get_block("B")

    sch.annotate_buffer_access(
        block, 0, "read", lambda vi, vj: ((vi - 1, vi + 2), (vj - 1, vj + 2))
    )

    sch.annotate_buffer_access(
        block, 0, "read", lambda vi, vj: ((vi - 2, vi + 3), (vj - 2, vj + 3))
    )

    assert_structural_equal_ignore_global_symbol(sch.mod["main"], expected)
    verify_trace_roundtrip(sch=sch, mod=before)

# @T.prim_func
# def resize_before(x: T.Buffer((1, 3, 200, 200), "float32"), y: T.Buffer((1, 3, 100, 100), "float32")):
#     x_global = T.alloc_buffer([1, 3, 200, 200], dtype="float32")
#     for ax0, ax1, ax2, ax3 in T.grid(1, 3, 200, 200):
#         with T.block("cache"):
#             v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
#             x_global[v0, v1, v2, v3] = x[v0, v1, v2, v3]
#     for n, c, h, w in T.grid(1, 3, 100, 100):
#         with T.block("resize"):
#             vn, vc, vh, vw = T.axis.remap("SSSS", [n, c, h, w])
#             T.reads(x_global[vn, vc, 0:200, 0:200])
#             T.writes(y[vn, vc, vh, vw])
#             y[vn, vc, vh, vw] = x_global[vn, vc, 
#                 T.cast(T.floor(T.Cast("float32", vh) * T.float32(2) + T.float32(0.5)), "int32"),
#                 T.cast(T.floor(T.Cast("float32", vw) * T.float32(2) + T.float32(0.5)), "int32")]


@T.prim_func
def resize_before(x: T.Buffer((1, 3, 200, 200), "float32"), y: T.Buffer((1, 3, 100, 100), "float32")):
    x_global = T.alloc_buffer([1, 3, 200, 200], dtype="float32")
    for ax0, ax1, ax2, ax3 in T.grid(1, 3, 200, 200):
        with T.block("cache"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            x_global[v0, v1, v2, v3] = x[v0, v1, v2, v3]
    for i0, i1, i2, i3 in T.grid(1, 3, 100, 100):
        with T.block("resize"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(x_global[v_i0, v_i1, 0:200, 0:200])
            T.writes(y[v_i0, v_i1, v_i2, v_i3])
            y[v_i0, v_i1, v_i2, v_i3] = T.Cast("float32", T.Cast("float32", x_global[v_i0, v_i1, T.max(T.min(T.Cast("int32", T.floor((T.Cast("float32", v_i2) + T.float32(0.5)) * T.float32(2) - T.float32(0.5) + T.float32(1.0000000000000001e-05))), 199), 0), T.max(T.min(T.Cast("int32", T.floor((T.Cast("float32", v_i3) + T.float32(0.5)) * T.float32(2) - T.float32(0.5) + T.float32(1.0000000000000001e-05))), 199), 0)]))
            
@T.prim_func
def resize_tir(x: T.Buffer((1, 16, 512, 512), "float16"), resize: T.Buffer((1, 16, 256, 256), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i0, i1, i2, i3 in T.grid(1, 16, 256, 256):
        with T.block("resize"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(x[v_i0, v_i1, v_i2 * 2 - 1:v_i2 * 2 + 1, v_i3 * 2 - 1 : v_i3 * 2 + 1])
            T.writes(resize[v_i0, v_i1, v_i2, v_i3])
            resize[v_i0, v_i1, v_i2, v_i3] = T.Cast("float16", T.Cast("float32", x[v_i0, v_i1, T.max(T.min(T.Cast("int32", T.floor((T.Cast("float32", v_i2) + T.float32(0.5)) * T.float32(2) - T.float32(0.5) + T.float32(1.0000000000000001e-05))), 511), 0), T.max(T.min(T.Cast("int32", T.floor((T.Cast("float32", v_i3) + T.float32(0.5)) * T.float32(2) - T.float32(0.5) + T.float32(1.0000000000000001e-05))), 511), 0)]))            

@T.prim_func
def resize_after(x: T.Buffer((1, 3, 200, 200), "float32"), y: T.Buffer((1, 3, 100, 100), "float32")):
    x_global = T.alloc_buffer([1, 3, 200, 200], dtype="float32")
    for h_o, w_o in T.grid(7, 7):
        for ax0, ax1, ax2, ax3 in T.grid(1, 3, 30, 30):
            with T.block("cache"):
                v0 = T.axis.spatial(1, ax0)
                v1 = T.axis.spatial(3, ax1)
                v2 = T.axis.spatial(200, h_o * 30 + ax2)
                v3 = T.axis.spatial(200, w_o * 30 + ax3)
                T.where(h_o * 30 + ax2 < 200 and w_o * 30 + ax3 < 200)
                x_global[v0, v1, v2, v3] = x[v0, v1, v2, v3]
        for n, c, h_i, w_i in T.grid(1, 3, 15, 15):
            with T.block("resize"):
                vn = T.axis.spatial(1, n)
                vc = T.axis.spatial(3, c)
                vh = T.axis.spatial(100, h_o * 15 + h_i)
                vw = T.axis.spatial(100, w_o * 15 + w_i)
                T.where(h_o * 15 + h_i < 100 and w_o * 15 + w_i < 100)
                T.reads(x_global[vn, vc, vh * 2 - 1 : vh * 2 + 3, vw * 2 - 1 : vw * 2 + 3])
                T.writes(y[vn, vc, vh, vw])
                T.block_attr({"explicit_read_region": [0]})
                y[vn, vc, vh, vw] = x_global[vn, vc, 
                    T.cast(T.floor(vh * 2 + 0.5), "int32"),
                    T.cast(T.floor(vw * 2 + 0.5), "int32")]

def test_annotate_buffer_access_enables_compute_at_for_resize():
    # sch = tir.Schedule(resize_before, debug_mask="all")
    
    # block = sch.get_block("resize")
    # cache_block = sch.get_block("cache")
    
    # # Try to tile and compute_at without annotate_buffer_access (should fail)
    # try:
    #     h, w = sch.get_loops(block)[-2:]
    #     ho, hi = sch.split(h, factors=[7, 15])
    #     wo, wi = sch.split(w, factors=[7, 15])
    #     sch.reorder(ho, wo, hi, wi)
    #     sch.compute_at(cache_block, wo)
    #     print(sch.mod["main"].script())
    #     assert False, "compute_at should have failed without annotate_buffer_access"
    # except tvm.tir.ScheduleError:
    #     print("compute_at failed without annotate_buffer_access, as expected.")
    
    # Reset schedule
    sch = tir.Schedule(resize_before, debug_mask="all")
    block = sch.get_block("resize")
    cache_block = sch.get_block("cache")
    
    def _get_block(s: tir.ScheduleState, name_hint: str) -> tir.StmtSRef:
        result = None

        def f_visit(node):
            nonlocal result
            if isinstance(node, tvm.tir.Block) and node.name_hint == name_hint:
                result = node

        func = s.mod["main"]
        post_order_visit(func.body, f_visit)
        assert result is not None and isinstance(result, tvm.tir.Block)
        return s.get_sref(result)
    
    
    # s = tir.ScheduleState(resize_before, debug_mask="all")
    # resize_flags = s._get_cached_flags(_get_block(s, "resize"))
    # print(resize_flags)
    
    # Annotate buffer access
    sch.annotate_buffer_access(block, 0, "read", 
        lambda vn, vc, vh, vw: (vn, vc, 
            (vh * 2 - 3, vh * 2 + 3),
            (vw * 2 - 3, vw * 2 + 3)))
    
    # Tile and compute_at with annotate_buffer_access (should succeed)
    h, w = sch.get_loops(block)[-2:]
    ho, hi = sch.split(h, factors=[20, 10])
    wo, wi = sch.split(w, factors=[20, 10])
    sch.reorder(ho, wo, hi, wi)
    sch.compute_at(cache_block, wo)
    
    print(sch.mod["main"].script())
    assert_structural_equal_ignore_global_symbol(sch.mod["main"], resize_after)
    verify_trace_roundtrip(sch=sch, mod=resize_before)


if __name__ == "__main__":
    test_annotate_buffer_access_enables_compute_at_for_resize()
    tvm.testing.main()
