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
from tvm.contrib.edgex.tir.transform import SplitVcuControlFlow


# fmt: off
@T.prim_func
def vcu_example(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [256], dtype="int32")
    A_dm = T.allocate([256], "int32", "dm")
    A_vm = T.allocate([128], "int32", "vm")
    T.evaluate(T.nnp_eidma_load(dtype="handle"))
    T.evaluate(T.nnp_eidma_load(dtype="handle"))
    T.evaluate(T.nnp_sync("eidma", "ub", "vidma", dtype="handle"))
    for i in T.serial(2, 1):
        T.evaluate(T.nnp_sync("cu", "ub", "vcu", dtype="handle"))
        T.evaluate(T.nnp_sync("vcu", "wo", "cu", dtype="handle"))
        if (i == 0):
            T.evaluate(T.nnp_sync("vidma", "wo", "eidma", dtype="handle"))
        if (i == 1):
            T.evaluate(T.nnp_sync("vidma", "wo", "vodma", dtype="handle"))
        T.evaluate(T.nnp_vidma_load(dtype="handle"))
        T.evaluate(T.nnp_vidma_load(dtype="handle"))
        T.evaluate(T.nnp_sync("vidma", "wo", "eidma", dtype="handle"))
        for j in T.serial(64, 1):
            A_vm[j] = (T.load("int32", A_vm, j) + T.load("int32", A_vm, (64 + j)))
        T.evaluate(T.nnp_vodma_store(dtype="handle"))
        if (i == 0):
            T.evaluate(T.nnp_sync("vodma", "ub", "vidma", dtype="handle"))
        if (i == 1):
            T.evaluate(T.nnp_sync("vodma", "ub", "eodma", dtype="handle"))
    T.evaluate(T.nnp_sync("eodma", "wo", "vodma", dtype="handle"))
    T.evaluate(T.nnp_eodma_store(dtype="handle"))


@T.prim_func
def vcu_splitted(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [256], dtype="int32")
    A_dm = T.allocate([256], "int32", "dm")
    A_vm = T.allocate([128], "int32", "vm")
    if T.nnp_cuid(dtype="int32") >= 4:
        T.evaluate(T.nnp_lock_vcu(dtype=""))
        T.evaluate(T.nnp_iss_bind_input_buffer(dtype=""))
        T.evaluate(T.nnp_eidma_load(dtype="handle"))
        T.evaluate(T.nnp_eidma_load(dtype="handle"))
        T.evaluate(T.nnp_sync("eidma", "ub", "vidma", dtype="handle"))
        for i in T.serial(2, 1):
            T.evaluate(T.nnp_sync("cu", "ub", "vcu", dtype="handle"))
        T.evaluate(T.nnp_sync("eodma", "wo", "vodma", dtype="handle"))
        T.evaluate(T.nnp_eodma_store(dtype="handle"))
    else:
        for i_1 in T.serial(2, 1):
            T.evaluate(T.nnp_sync("vcu", "wo", "cu", dtype="handle"))
            if (i_1 == 0):
                T.evaluate(T.nnp_sync("vidma", "wo", "eidma", dtype="handle"))
            if (i_1 == 1):
                T.evaluate(T.nnp_sync("vidma", "wo", "vodma", dtype="handle"))
            T.evaluate(T.nnp_vidma_load(dtype="handle"))
            T.evaluate(T.nnp_vidma_load(dtype="handle"))
            T.evaluate(T.nnp_sync("vidma", "wo", "eidma", dtype="handle"))
            for j in T.serial(64, 1):
                A_vm[j] = (T.load("int32", A_vm, j) + T.load("int32", A_vm, (64 + j)))
            T.evaluate(T.nnp_vodma_store(dtype="handle"))
            if (i_1 == 0):
                T.evaluate(T.nnp_sync("vodma", "ub", "vidma", dtype="handle"))
            if (i_1 == 1):
                T.evaluate(T.nnp_sync("vodma", "ub", "eodma", dtype="handle"))
        T.evaluate(T.nnp_unlock_vcu(dtype=""))
# fmt: on


def test_split_vcu_control_flow():
    mod = tvm.IRModule.from_expr(vcu_example)
    mod = SplitVcuControlFlow()(mod)
    assert mod["main"].body.attr_key == "vcore_resource"
    tvm.ir.assert_structural_equal(mod["main"].body.body, vcu_splitted.body, True)


if __name__ == "__main__":
    test_split_vcu_control_flow()
