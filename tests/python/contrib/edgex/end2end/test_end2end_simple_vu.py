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
import inspect
import itertools

import pytest
import tvm
from tvm import tir
from tvm.script import tir as T
import numpy as np
import tvm.testing
from tvm.contrib.edgex.testing import check_edgex_tir_build
from tvm.contrib.edgex.tir.schedule import EdgexSchedule


@T.prim_func
def myadd(a: T.handle, b: T.handle, c: T.handle) -> None:
    n = T.var("int32")
    A = T.match_buffer(a, [n], "int32")
    B = T.match_buffer(b, [n], "int32")
    C = T.match_buffer(c, [n], "int32")
    for i in range(n):
        with T.block("myadd"):
            vi = T.axis.remap("S", [i])
            C[vi] = A[vi] + B[vi]


def simple_vu_tensorize_dma(s, dm_inputs, dm_output, vm_inputs, vm_output, n_dm, n_vm, dtype):
    """tensorize helper"""
    for block in dm_inputs:
        s.pragma(s.get_loops(block)[-1], "nnp_dma_scope", "eidma")
    for block in vm_inputs:
        s.pragma(s.get_loops(block)[-1], "nnp_dma_scope", "vidma")
    s.pragma(s.get_loops(vm_output)[-1], "nnp_dma_scope", "vodma")
    s.pragma(s.get_loops(dm_output)[-1], "nnp_dma_scope", "eodma")


def simple_vu_schedule_without_iter(tir_func, n, dtype):
    """schedule simple elemwise vu op where edma and vdma are both not iterative"""
    n_args = len(tir_func.params) - 1
    s = EdgexSchedule(tir_func, debug_mode=True)
    block = s.get_child_blocks(s.get_block("root"))[0]
    dm_inputs = []
    vm_inputs = []
    for i in range(n_args):
        dm_inputs.append(s.cache_read(block, i, "dm"))
    for i in range(n_args):
        vm_inputs.append(s.cache_read(block, i, "vm"))
    Yvm = s.cache_write(block, 0, "vm")
    Ydm = s.cache_write(Yvm, 0, "dm")
    s.vectorize(s.get_loops(block)[0])
    simple_vu_tensorize_dma(s, dm_inputs, Ydm, vm_inputs, Yvm, n, n, dtype)
    return s


def simple_vu_te_create_func(te_func, n, dtype):
    """helper function to use te to create prim func"""
    n_args = len(inspect.getfullargspec(te_func).args)
    input_tensors = [tvm.te.placeholder([n], dtype, "X%d" % i) for i in range(n_args)]
    output_tensor = tvm.te.compute([n], lambda i: te_func(*[x[i] for x in input_tensors]), "Y")
    return tvm.te.create_prim_func(input_tensors + [output_tensor])


def do_test_multi_param(*items, failures=None):
    """test simple elemwise vu op with parameter compositions"""
    for (key, desc), n, dtype in itertools.product(*items):
        if failures is not None:
            if key in failures or (key, n) in failures or (key, n, dtype) in failures:
                continue
        if isinstance(desc, tuple):
            te_func, numpy_func = desc
        else:
            te_func, numpy_func = desc, desc
        # print("Testcase (%s, %d, %s) with func:\n%s" % (key, n, dtype, inspect.getsource(te_func)))
        tir_func = simple_vu_te_create_func(te_func, n, dtype)
        s = simple_vu_schedule_without_iter(tir_func, n, dtype)
        name = "simple_vu_case_%s_%d_%s" % (key, n, dtype)
        # use numpy func as expect, do not run cpu, since they are not compatible on saturation
        check_edgex_tir_build(name, s.mod["main"], numpy_func=numpy_func, check_cpu=False)


def numpy_saturate_add(x, y):
    if isinstance(x, np.ndarray):
        dtype = x.dtype
        i64_add = x.astype("int64") + y
    else:
        dtype = y.dtype
        i64_add = x + y.astype("int64")
    tinfo = np.iinfo(dtype)
    saturate = np.maximum(np.minimum(i64_add, tinfo.max), tinfo.min)
    return saturate.astype(dtype)


def numpy_saturate_sub(x, y):
    if isinstance(x, np.ndarray):
        dtype = x.dtype
        i64_add = x.astype("int64") - y
    else:
        dtype = y.dtype
        i64_add = x - y.astype("int64")
    tinfo = np.iinfo(dtype)
    saturate = np.maximum(np.minimum(i64_add, tinfo.max), tinfo.min)
    return saturate.astype(dtype)


def numpy_saturate_mul(x, y):
    if isinstance(x, np.ndarray):
        dtype = x.dtype
        i64_add = x.astype("int64") * y
    else:
        dtype = y.dtype
        i64_add = x * y.astype("int64")
    tinfo = np.iinfo(dtype)
    saturate = np.maximum(np.minimum(i64_add, tinfo.max), tinfo.min)
    return saturate.astype(dtype)


@pytest.mark.edgex_slow
def test_simple_vu_ops_int32():
    """This test case is used to test i32 basic vu end2end functionality
    where edma and vdma are both not iterative."""
    funcs = {
        "add": (lambda x, y: x + y, lambda x, y: numpy_saturate_add(x, y)),
        "sub": (lambda x, y: x - y, lambda x, y: numpy_saturate_sub(x, y)),
        "mul": (lambda x, y: x * y, lambda x, y: numpy_saturate_mul(x, y)),
        "neg": (lambda x: -x, lambda x: numpy_saturate_sub(np.zeros_like(x), x)),
        "addc": (lambda x: x + 10, lambda x: numpy_saturate_add(x, 10)),
        "cadd": (lambda x: 10 + x, lambda x: numpy_saturate_add(10, x)),
        "subc": (lambda x: x - 10, lambda x: numpy_saturate_sub(x, 10)),
        "csub": (
            lambda x: 10 - x,
            lambda x: numpy_saturate_sub(
                10,
                x,
            ),
        ),
        "mulc": (lambda x: x * 5, lambda x: numpy_saturate_mul(x, 5)),
        "cmul": (lambda x: 5 * x, lambda x: numpy_saturate_mul(5, x)),
        "madd": (lambda x, y, z: x + y * z, lambda x, y, z: numpy_saturate_add(x, y * z)),
        "msub": (lambda x, y, z: x - y * z, lambda x, y, z: numpy_saturate_sub(x, y * z)),
        "logical_shl": lambda x: x >> 10,
        "arith_shr": lambda x: x << 15,
        "bitwise_and": (
            lambda x, y: tir.call_intrin("int32", "tir.bitwise_and", x, y),
            lambda x, y: np.bitwise_and(x, y),
        ),
        "bitwise_or": (
            lambda x, y: tir.call_intrin("int32", "tir.bitwise_or", x, y),
            lambda x, y: np.bitwise_or(x, y),
        ),
        "bitwise_xor": (
            lambda x, y: tir.call_intrin("int32", "tir.bitwise_xor", x, y),
            lambda x, y: np.bitwise_xor(x, y),
        ),
    }
    failures = {}
    lengths = [128, 29, 17]
    do_test_multi_param(funcs.items(), lengths, ["int32"], failures=failures)


@pytest.mark.edgex_slow
def test_simple_vu_ops_int8():
    """This test case is used to test basic i8 vu end2end functionality
    where edma and vdma are both not iterative."""
    funcs = {
        "add": (lambda x, y: x + y, lambda x, y: numpy_saturate_add(x, y)),
        "sub": (lambda x, y: x - y, lambda x, y: numpy_saturate_sub(x, y)),
        "mul": (lambda x, y: x * y, lambda x, y: numpy_saturate_mul(x, y)),
        "neg": (lambda x: -x, lambda x: numpy_saturate_sub(np.zeros_like(x), x)),
        "addc": (lambda x: x + tir.const(10, "int8"), lambda x: numpy_saturate_add(x, 10)),
        "cadd": (lambda x: tir.const(10, "int8") + x, lambda x: numpy_saturate_add(10, x)),
        "subc": (lambda x: x - tir.const(10, "int8"), lambda x: numpy_saturate_sub(x, 10)),
        "csub": (lambda x: tir.const(10, "int8") - x, lambda x: numpy_saturate_sub(10, x)),
        "mulc": (lambda x: x * tir.const(5, "int8"), lambda x: numpy_saturate_mul(x, 5)),
        "cmul": (lambda x: tir.const(5, "int8") * x, lambda x: numpy_saturate_mul(5, x)),
        "logical_shl": lambda x: x >> 5,
        "arith_shr": lambda x: x << 5,
        "bitwise_and": (
            lambda x, y: tir.call_intrin("int8", "tir.bitwise_and", x, y),
            lambda x, y: np.bitwise_and(x, y),
        ),
        "bitwise_or": (
            lambda x, y: tir.call_intrin("int8", "tir.bitwise_or", x, y),
            lambda x, y: np.bitwise_or(x, y),
        ),
        "bitwise_xor": (
            lambda x, y: tir.call_intrin("int8", "tir.bitwise_xor", x, y),
            lambda x, y: np.bitwise_xor(x, y),
        ),
    }
    failures = {}
    lengths = [128, 29, 17]
    do_test_multi_param(funcs.items(), lengths, ["int8"], failures=failures)


@pytest.mark.edgex_slow
def test_simple_vu_ops_float32():
    """This test case is used to test basic f32 vu end2end functionality
    where edma and vdma are both not iterative."""
    funcs = {
        "add": lambda x, y: x + y,
        "sub": lambda x, y: x - y,
        "mul": lambda x, y: x * y,
        "neg": lambda x: -x,
        "addc": lambda x: x + 10.0,
        "cadd": lambda x: 10.0 + x,
        "subc": lambda x: x - 10.0,
        "csub": lambda x: 10.0 - x,
        "mulc": lambda x: x * 5.0,
        "cmul": lambda x: 5.0 * x,
    }
    failures = {}
    lengths = [128, 29, 17]
    do_test_multi_param(funcs.items(), lengths, ["float32"], failures=failures)


@pytest.mark.edgex_slow
def test_simple_vu_ops_float16():
    """This test case is used to test basic f16 vu end2end functionality
    where edma and vdma are both not iterative."""
    funcs = {
        "add": lambda x, y: x + y,
        "sub": lambda x, y: x - y,
        "mul": lambda x, y: x * y,
        "neg": lambda x: -x,
        "addc": (lambda x: x + tir.const(10, "float16"), lambda x: x + np.half(10)),
        "cadd": (lambda x: tir.const(10, "float16") + x, lambda x: np.half(10) + x),
        "subc": (lambda x: x - tir.const(10, "float16"), lambda x: x - np.half(10)),
        "csub": (lambda x: tir.const(10, "float16") - x, lambda x: np.half(10) - x),
        "mulc": (lambda x: x * tir.const(5, "float16"), lambda x: x * np.half(5)),
        "cmul": (lambda x: tir.const(5, "float16") * x, lambda x: np.half(5) * x),
    }
    failures = {}
    lengths = [128, 29, 17]
    do_test_multi_param(funcs.items(), lengths, ["float16"], failures=failures)


@pytest.mark.edgex_slow
def test_simple_vu_ops_float_to_int_cast():
    """This test case is used to test basic f16 vu casting functionality"""
    funcs = {
        "cast_int8": (lambda x: tir.Cast("int8", x), lambda x: x.astype("int8")),
        "cast_int32": (lambda x: tir.Cast("int32", x), lambda x: x.astype("int32")),
    }
    failures = {}
    lengths = [128, 29, 17]
    do_test_multi_param(funcs.items(), lengths, ["float16", "float32"], failures=failures)


@pytest.mark.edgex_slow
def test_simple_vu_add_dm_in_iter():
    """This test case is used to test basic vu end2end functionality.
    where edma and vdma are both iterative in outer loop"""
    n = 1024
    factor = 128
    dtype = "int32"
    a = myadd.params[0]
    func = myadd.specialize({a: tir.decl_buffer([n])})
    s = EdgexSchedule(func, debug_mode=True)
    add = s.get_block("myadd")
    Adm = s.cache_read(add, 0, "dm")
    Bdm = s.cache_read(add, 1, "dm")
    Avm = s.cache_read(add, 0, "vm")
    Bvm = s.cache_read(add, 1, "vm")
    Cvm = s.cache_write(add, 0, "vm")
    Cdm = s.cache_write(Cvm, 0, "dm")
    (axis,) = s.get_loops(Cdm)
    outer_i, _ = s.split(axis, factors=[None, factor])
    s.compute_at(Cvm, outer_i)
    s.compute_at(add, outer_i)
    s.compute_at(Bvm, outer_i)
    s.compute_at(Avm, outer_i)
    s.compute_at(Bdm, outer_i)
    s.compute_at(Adm, outer_i)
    s.vectorize(s.get_loops(add)[1])
    simple_vu_tensorize_dma(s, [Adm, Bdm], Cdm, [Avm, Bvm], Cvm, factor, factor, dtype)
    check_edgex_tir_build("simple_vu_add_dm_in_iter", s.mod["main"], check_cpu=True)


@pytest.mark.edgex_slow
def test_simple_vu_add_vm_in_iter():
    """This test case is used to test basic vu end2end functionality.
    where edma is global and vdma is iterative in outer loop"""
    n = 1024
    factor = 128
    dtype = "int32"
    a = myadd.params[0]
    func = myadd.specialize({a: tir.decl_buffer([n])})
    s = EdgexSchedule(func, debug_mode=True)
    add = s.get_block("myadd")
    Adm = s.cache_read(add, 0, "dm")
    Bdm = s.cache_read(add, 1, "dm")
    Avm = s.cache_read(add, 0, "vm")
    Bvm = s.cache_read(add, 1, "vm")
    Cvm = s.cache_write(add, 0, "vm")
    Cdm = s.cache_write(Cvm, 0, "dm")
    (axis,) = s.get_loops(Cvm)
    outer_i, _ = s.split(axis, factors=[None, factor])
    s.compute_at(add, outer_i)
    s.compute_at(Bvm, outer_i)
    s.compute_at(Avm, outer_i)
    s.vectorize(s.get_loops(add)[1])
    simple_vu_tensorize_dma(s, [Adm, Bdm], Cdm, [Avm, Bvm], Cvm, n, factor, dtype)
    check_edgex_tir_build("simple_vu_add_vm_in_iter", s.mod["main"], check_cpu=True)


@pytest.mark.edgex_slow
def test_simple_vu_add_dm_in_iter0_vm_in_iter1():
    """This test case is used to test basic vu end2end functionality.
    where edma is in outer loop and vdma is in inner loop"""
    n = 4096
    dm_factor = 1024
    vm_factor = 128
    dtype = "int32"
    a = myadd.params[0]
    func = myadd.specialize({a: tir.decl_buffer([n])})
    s = EdgexSchedule(func, debug_mode=True)
    add = s.get_block("myadd")
    Adm = s.cache_read(add, 0, "dm")
    Bdm = s.cache_read(add, 1, "dm")
    Avm = s.cache_read(add, 0, "vm")
    Bvm = s.cache_read(add, 1, "vm")
    Cvm = s.cache_write(add, 0, "vm")
    Cdm = s.cache_write(Cvm, 0, "dm")
    (axis,) = s.get_loops(add)
    dm_i, inner_i = s.split(axis, factors=[None, dm_factor])
    vm_i, _ = s.split(inner_i, factors=[None, vm_factor])
    s.compute_at(Avm, vm_i)
    s.compute_at(Bvm, vm_i)
    s.compute_at(Adm, dm_i)
    s.compute_at(Bdm, dm_i)
    s.reverse_compute_at(Cvm, vm_i)
    s.reverse_compute_at(Cdm, dm_i)
    s.vectorize(s.get_loops(add)[2])
    simple_vu_tensorize_dma(s, [Adm, Bdm], Cdm, [Avm, Bvm], Cvm, dm_factor, vm_factor, dtype)
    check_edgex_tir_build("simple_vu_add_dm_in_iter0_vm_in_iter1", s.mod["main"], check_cpu=True)


if __name__ == "__main__":
    test_simple_vu_add_dm_in_iter()
    test_simple_vu_add_vm_in_iter()
    test_simple_vu_add_dm_in_iter0_vm_in_iter1()
    test_simple_vu_ops_int8()
    test_simple_vu_ops_int32()
    test_simple_vu_ops_float16()
    test_simple_vu_ops_float32()
    test_simple_vu_ops_float_to_int_cast()
