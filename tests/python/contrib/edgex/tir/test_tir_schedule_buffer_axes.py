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
from tvm.contrib.edgex.tir.schedule import EdgexSchedule
from tvm.script import tir as T


@T.prim_func
def with_intermediate_buffer(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    C = T.match_buffer(c, (128, 128))
    B = T.alloc_buffer((128, 128))
    for i in range(0, 128):
        for j in range(0, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 2.0
    for i in range(0, 128):
        for j in range(0, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def with_intermediate_buffer_split_first_dim(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    C = T.match_buffer(c, [128, 128])
    B = T.alloc_buffer([2, 64, 128])
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads([A[vi, vj]])
            T.writes([B[T.floordiv(vi, 64), T.floormod(vi, 64), vj]])
            B[T.floordiv(vi, 64), T.floormod(vi, 64), vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads([B[T.floordiv(vi, 64), T.floormod(vi, 64), vj]])
            T.writes([C[vi, vj]])
            C[vi, vj] = B[T.floordiv(vi, 64), T.floormod(vi, 64), vj] + 1.0


@T.prim_func
def with_intermediate_buffer_fused(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    C = T.match_buffer(c, [128, 128])
    B = T.alloc_buffer([16384])
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads([A[vi, vj]])
            T.writes([B[vi * 128 + vj]])
            B[vi * 128 + vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads([B[vi * 128 + vj]])
            T.writes([C[vi, vj]])
            C[vi, vj] = B[vi * 128 + vj] + 1.0


@T.prim_func
def with_intermediate_buffer_reorder(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    C = T.match_buffer(c, (128, 128))
    B = T.alloc_buffer((128, 128))
    for i in range(0, 128):
        for j in range(0, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vj, vi] = A[vi, vj] * 2.0
    for i in range(0, 128):
        for j in range(0, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = B[vj, vi] + 1.0


@T.prim_func
def with_function_input_buffer_fused(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [16384])
    C = T.match_buffer(c, [128, 128])
    B = T.alloc_buffer([128, 128])
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads([A[vi * 128 + vj]])
            T.writes([B[vi, vj]])
            B[vi, vj] = A[vi * 128 + vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads([B[vi, vj]])
            T.writes([C[vi, vj]])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def with_function_input_buffer_stacked(a: T.handle) -> None:
    A_C = T.match_buffer(a, [256, 128])
    B = T.alloc_buffer([128, 128])
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads([A_C[vi, vj]])
            T.writes([B[vi, vj]])
            B[vi, vj] = A_C[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads([B[vi, vj]])
            T.writes([A_C[vi + 128, vj]])
            A_C[vi + 128, vj] = B[vi, vj] + 1.0


def test_buffer_split():
    s = EdgexSchedule(with_intermediate_buffer)
    B = s.get_block("B")
    n, m = s.get_write_buffer_axes(B, 0)
    s.split_buffer(n, factor=64)
    tvm.ir.assert_structural_equal(with_intermediate_buffer_split_first_dim, s.mod["main"])


def test_buffer_fuse():
    s = EdgexSchedule(with_intermediate_buffer)
    B = s.get_block("B")
    n, m = s.get_write_buffer_axes(B, 0)
    s.fuse_buffer(n, m)
    tvm.ir.assert_structural_equal(with_intermediate_buffer_fused, s.mod["main"])


def test_buffer_reorder():
    s = EdgexSchedule(with_intermediate_buffer)
    B = s.get_block("B")
    n, m = s.get_write_buffer_axes(B, 0)
    s.reorder_buffer(m, n)
    tvm.ir.assert_structural_equal(with_intermediate_buffer_reorder, s.mod["main"])


def test_fuse_function_buffer():
    s = EdgexSchedule(with_intermediate_buffer)
    B = s.get_block("B")
    n, m = s.get_read_buffer_axes(B, 0)
    s.fuse_buffer(n, m)
    tvm.ir.assert_structural_equal(with_function_input_buffer_fused, s.mod["main"])


def test_stack_function_buffer():
    s = EdgexSchedule(with_intermediate_buffer)
    B = s.get_block("B")
    n1, m1 = s.get_read_buffer_axes(B, 0)
    C = s.get_block("C")
    n2, m2 = s.get_write_buffer_axes(C, 0)
    s.stack_buffer(n1, n2)
    tvm.ir.assert_structural_equal(with_function_input_buffer_stacked, s.mod["main"])


if __name__ == "__main__":
    test_buffer_split()
    test_buffer_fuse()
    test_buffer_reorder()
    test_fuse_function_buffer()
    test_stack_function_buffer()
