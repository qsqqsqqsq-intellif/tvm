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
from tvm.script import tir as T
from tvm.contrib.edgex.tir.transform import EliminateDynamicAllocation


@T.prim_func
def func_with_allocations() -> None:
    A0 = T.allocate([1024], "float32", scope="global")
    for i in range(64):
        A1 = T.allocate([100 + i], "float32", scope="global")
        A2 = T.allocate([100 + i], "float32", scope="dm")
        for j in range(8):
            A3 = T.allocate([T.max(i * 8 + j + 1, 200)], "float32", scope="vm")
            A4 = T.allocate([T.min(10 - 8 * i - j, 1)], "float32", scope="vm")
            T.evaluate(0)


@T.prim_func
def func_with_allocations_rewritten() -> None:
    A0 = T.allocate([1024], "float32", "global")
    for i in T.serial(64):
        A1 = T.allocate([100 + i], "float32", "global")
        A2 = T.allocate([163], "float32", "dm")
        for j in T.serial(8):
            A3 = T.allocate([512], "float32", "vm")
            A4 = T.allocate([1], "float32", "vm")
            T.evaluate(0)


def test_eliminate_dynamic_allocation():
    mod = tvm.IRModule.from_expr(func_with_allocations)
    mod = EliminateDynamicAllocation()(mod)
    tvm.ir.assert_structural_equal(mod["main"], func_with_allocations_rewritten, True)


if __name__ == "__main__":
    test_eliminate_dynamic_allocation()
