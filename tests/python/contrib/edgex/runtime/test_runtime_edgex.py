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
import os
import tvm
from tvm import te
from tvm.contrib.edgex import edgex_runtime
import numpy as np
import tvm.topi.testing
import tvm.testing
import tvm._ffi


def verify_matmul(m, l, n, dtype="int8"):
    """matmul: m*l X l*n"""
    A = te.placeholder((m, l), dtype=dtype, name="A")
    B = te.placeholder((l, n), dtype=dtype, name="B")
    C = edgex_runtime.matmul(A, B)
    s = te.create_schedule([C.op])

    ctx = tvm.edgex(0)
    # data in host, from model
    # two input tensors in one file
    base_dir = os.environ.get("EDGEX_ROOT_DIR", "./")
    a_np = np.fromfile(base_dir + "/tests/matmul_case/ddr_in_MatMul1.bin", dtype=dtype).reshape(
        (m, l)
    )
    b_np = np.fromfile(base_dir + "/tests/matmul_case/ddr_in_MatMul2.bin", dtype=dtype).reshape(
        (l, n)
    )
    c_np = np.fromfile(base_dir + "/tests/matmul_case/ddr_out_MatMul.bin", dtype=dtype).reshape(
        (m, n)
    )
    # copy data from host to device, in CreateTVMOp
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(b_np, ctx)
    c = tvm.nd.empty((m, n), dtype, ctx)
    f = tvm.build(s, [A, B, C], "edgex", target_host="llvm", name="add")
    f(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)


def test_matmul():
    verify_matmul(384, 256, 384)


def verify_add(shape, dtype="int8"):
    A = te.placeholder(shape, dtype=dtype, name="A")
    B = edgex_runtime.add(A)
    s = te.create_schedule([B.op])

    ctx = tvm.edgex(0)
    # data in host, from model
    # two input tensors in one file
    base_dir = os.environ.get("EDGEX_ROOT_DIR", "./")
    a_np = np.fromfile(base_dir + "/tests/add_case/data_in.bin", dtype=dtype)
    c_np = np.fromfile(base_dir + "/tests/add_case/ref.dat.bin", dtype=dtype)
    # copy data from host to device, in CreateTVMOp
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.empty((shape[0] // 2,), dtype, ctx)
    f = tvm.build(s, [A, B], "edgex", target_host="llvm", name="add")
    f(a, b)
    tvm.testing.assert_allclose(b.numpy(), c_np, rtol=1e-3)


def test_add():
    verify_add((1024,))


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
        edgex_dir + "/tests/add_case/drv_case0.bin",
        edgex_dir + "/tests/add_case/drv_case0_cpp.lst",
        [a, b],
        False,
    )
    assert np.array_equal(b.numpy(), a_np[0:512] + a_np[512:])


if __name__ == "__main__":
    test_ndarray()
    test_add()
    test_matmul()
    test_iss()
