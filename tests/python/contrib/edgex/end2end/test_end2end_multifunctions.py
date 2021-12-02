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
import numpy as np
import tvm.testing
import tvm.contrib.edgex.edgex as edgex
from tvm.contrib import graph_runtime
from tvm.script import tir as T
from tvm.autotvm import DispatchContext


@T.prim_func
def myadd(a: T.handle, b: T.handle, c: T.handle) -> None:
    T.func_attr({"global_symbol": "myadd"})
    A = T.match_buffer(a, [16], "int32")
    B = T.match_buffer(b, [16], "int32")
    C = T.match_buffer(c, [16], "int32")
    for i in range(16):
        with T.block("block"):
            vi = T.axis.remap("S", [i])
            C[i] = A[vi] + B[vi]


@T.prim_func
def mysub(a: T.handle, b: T.handle, c: T.handle) -> None:
    T.func_attr({"global_symbol": "mysub"})
    A = T.match_buffer(a, [16], "int32")
    B = T.match_buffer(b, [16], "int32")
    C = T.match_buffer(c, [16], "int32")
    for i in range(16):
        with T.block("block"):
            vi = T.axis.remap("S", [i])
            C[i] = A[vi] - B[vi]


def test_build_multi_functions():
    ctx = tvm.edgex()
    x = tvm.nd.array(np.random.randint(-10, 10, [16]).astype("int32"), ctx)
    y = tvm.nd.array(np.random.randint(-10, 10, [16]).astype("int32"), ctx)
    z = tvm.nd.array(np.zeros([16]).astype("int32"), ctx)
    mod = tvm.IRModule(functions={"myadd": myadd, "mysub": mysub})
    with edgex.build_config_nnp():
        f = tvm.build(mod, target="edgex")
        f["myadd"](x, y, z)
        tvm.testing.assert_allclose(z.asnumpy(), x.asnumpy() + y.asnumpy())
        f["mysub"](x, y, z)
        tvm.testing.assert_allclose(z.asnumpy(), x.asnumpy() - y.asnumpy())


if __name__ == "__main__":
    test_build_multi_functions()
