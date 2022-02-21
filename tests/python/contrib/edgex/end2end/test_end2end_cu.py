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
import tvm
import pytest
import tvm.testing
from tvm.script import tir as T
from tvm.contrib.edgex.testing import check_edgex_tir_build


@T.prim_func
def myadd(a: T.handle, b: T.handle, c: T.handle) -> None:
    n = T.var("int32")
    A = T.match_buffer(a, [n], "int32")
    B = T.match_buffer(b, [n], "int32")
    C = T.match_buffer(c, [n], "int32")
    for i in range(0, n):
        with T.block("myadd"):
            (vi,) = T.axis.remap("S", [i])
            C[vi] = A[vi] + B[vi]


@T.prim_func
def mysqrt(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [1024], "float32")
    B = T.match_buffer(b, [1024], "float32")
    for i in range(0, 1024):
        with T.block("mysqrt"):
            (vi,) = T.axis.remap("S", [i])
            B[vi] = T.sqrt(A[vi], dtype="float32")


@T.prim_func
def add_with_global_temp_workspace(a: T.handle, b: T.handle, c: T.handle) -> None:
    n = T.var("int32")
    A = T.match_buffer(a, [n], "int32")
    B = T.match_buffer(b, [n], "int32")
    C = T.match_buffer(c, [n], "int32")
    C_temp = T.alloc_buffer([n], dtype="int32", scope="global")
    for i in range(0, n):
        C_temp[i] = A[i] + B[i]
    for i in range(0, n):
        C[i] = C_temp[i]


def test_cu_add():
    n = 1024
    func = myadd
    func = func.specialize({func.params[0]: tvm.tir.decl_buffer([n])})
    mod = tvm.tir.transform.DecorateDeviceScope()(tvm.IRModule.from_expr(func))
    check_edgex_tir_build("simple_cu_add", mod["main"], check_cpu=False)


@pytest.mark.skip
def test_cu_sqrt():
    n = 1024
    func = mysqrt
    func = func.specialize({func.params[0]: tvm.tir.decl_buffer([n])})
    check_edgex_tir_build("simple_cu_add", func, check_cpu=True)


def test_temp_global_workspace():
    func = add_with_global_temp_workspace
    func = func.specialize({func.params[0]: tvm.tir.decl_buffer([128])})
    check_edgex_tir_build("add_with_global_temp_workspace", func, check_cpu=True)


if __name__ == "__main__":
    # test_cu_sqrt()
    test_cu_add()
    test_temp_global_workspace()
