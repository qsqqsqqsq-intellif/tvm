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
import subprocess
import os
import tempfile
import tvm
from tvm import te
from tvm.contrib.edgex.runtime import edgex_runtime
import numpy as np
from tvm.contrib.edgex.edgex import build_config_nnp
import tvm.topi.testing
import tvm.testing
import tvm._ffi


def test_add():
    shape = [1024]
    A = te.placeholder(shape, dtype="int8", name="A")
    B = tvm.te.extern(
        (A.shape[0] // 2,),
        [A],
        lambda ins, outs: tvm.tir.call_packed("tvm.contrib.edgex.add_example", ins[0], outs[0]),
        name="B",
    )
    s = te.create_schedule([B.op])
    ctx = tvm.edgex(0)
    base_dir = os.environ.get("EDGEX_ROOT_DIR", "./")
    a_np = np.fromfile(base_dir + "/tests/drv_case0_ncore_vcore0/data_in.bin", dtype="int8")
    c_np = np.fromfile(base_dir + "/tests/drv_case0_ncore_vcore0/ref.dat.bin", dtype="int8")
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.empty((shape[0] // 2,), "int8", ctx)
    f = tvm.build(s, [A, B], "edgex", name="add")
    f(a, b)
    tvm.testing.assert_allclose(b.numpy(), c_np, rtol=1e-3)


def test_ndarray():
    device = tvm.edgex()

    # allocation and copy host to device
    numpy_arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], "int32")
    device_arr = tvm.nd.array(numpy_arr, device)
    assert np.array_equal(numpy_arr, device_arr.numpy())

    # copy device to device
    device_arr2 = tvm.nd.empty([3, 3], "int32", device)
    device_arr2.copyfrom(device_arr)
    assert np.array_equal(numpy_arr, device_arr2.numpy())

    # device to host
    host_arr = tvm.nd.empty([3, 3], "int32", tvm.cpu())
    host_arr.copyfrom(device_arr)
    assert np.array_equal(numpy_arr, host_arr.numpy())


def test_iss():
    edgex_dir = os.environ["EDGEX_ROOT_DIR"]
    a_np = np.ones(1024, dtype="int8")
    a = tvm.nd.array(a_np, tvm.edgex(0))
    b = tvm.nd.array(np.ones(512, dtype="int8"), tvm.edgex(0))
    edgex_runtime.edgex_launch_iss(
        edgex_dir + "/tests/drv_case0_ncore_vcore0/drv_case0_ncore_vcore0.bin",
        edgex_dir + "/tests/drv_case0_ncore_vcore0/drv_case0_ncore_vcore0_cpp.lst",
        [a, b],
        False,
    )
    assert np.array_equal(b.numpy(), a_np[0:512] + a_np[512:])


def test_create_llvm_module():
    x = tvm.te.placeholder([128], "int32")
    y = tvm.te.compute([128], lambda i: x[i] + 1)
    s = tvm.contrib.edgex.tir.schedule.EdgexSchedule(tvm.te.create_prim_func([x, y]))
    block = s.get_child_blocks(s.get_block("root"))[0]
    x_dm = s.cache_read(block, 0, "dm")
    x_vm = s.cache_read(block, 0, "vm")
    y_vm = s.cache_write(block, 0, "vm")
    y_dm = s.cache_write(y_vm, 0, "dm")
    s.vectorize(s.get_loops(block)[0])
    s.pragma(s.get_loops(x_dm)[-1], "nnp_dma_scope", "eidma")
    s.pragma(s.get_loops(x_vm)[-1], "nnp_dma_scope", "vidma")
    s.pragma(s.get_loops(y_vm)[-1], "nnp_dma_scope", "vodma")
    s.pragma(s.get_loops(y_dm)[-1], "nnp_dma_scope", "eodma")
    with build_config_nnp():
        mod = tvm.lower(s.mod)
    llvm_mod = edgex_runtime.create_llvm_module(mod, tvm.target.edgex())
    llvm_bin_dir = os.environ.get("EDGEX_LLVM_TOOLCHAIN_DIR")
    assert llvm_bin_dir, "EDGEX_LLVM_TOOLCHAIN_DIR not configured"
    with tempfile.NamedTemporaryFile("w") as ll_file:
        ll_file.write(llvm_mod)
        ll_file.flush()
        status = subprocess.call([f"{llvm_bin_dir}/llc", "-mtriple=nnp", ll_file.name])
        assert status == 0


if __name__ == "__main__":
    test_ndarray()
    test_add()
    test_iss()
    test_create_llvm_module()
