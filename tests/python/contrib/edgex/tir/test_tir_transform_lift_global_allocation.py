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
from tvm.ir.module import IRModule
from tvm.script import tir as T
from tvm.contrib.edgex.tir.transform import LiftGlobalAllocation


# fmt: off
@T.prim_func
def intermediate_ddr_allocation(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [64], dtype="int32")
    B = T.match_buffer(b, [64], dtype="int32")
    T.attr(0, "device_scope", 0)
    T0 = T.allocate([64], dtype="int32", scope="global")
    T1 = T.allocate([64], dtype="int32", scope="global")
    for i in T.serial(0, 64):
        T.store(T0, i, T.load("int32", A.data, i) + 1)
    for i in T.serial(0, 64):
        T.store(T1, i, T.load("int32", T0, i) + 1)
    for i in T.serial(0, 64):
        T.store(B.data, i, T.load("int32", T1, i) + 1)


@T.prim_func
def intermediate_ddr_allocation_lifted(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [64], dtype="int32")
    B = T.match_buffer(b, [64], dtype="int32")
    T0 = T.allocate([64], "int32", "global")
    T1 = T.allocate([64], "int32", "global")
    T.attr(0, "device_scope", 0)
    for i in T.serial(0, 64):
        T.store(T0, i, (T.load("int32", A.data, i) + 1), 1)
    for i in T.serial(0, 64):
        T.store(T1, i, (T.load("int32", T0, i) + 1), 1)
    for i in T.serial(0, 64):
        T.store(B.data, i, (T.load("int32", T1, i) + 1), 1)
# fmt: on


def test_lift_intermediate_ddr_allocation():
    mod = IRModule.from_expr(intermediate_ddr_allocation)
    mod = LiftGlobalAllocation()(mod)
    tvm.ir.assert_structural_equal(mod["main"], intermediate_ddr_allocation_lifted, True)


if __name__ == "__main__":
    test_lift_intermediate_ddr_allocation()
