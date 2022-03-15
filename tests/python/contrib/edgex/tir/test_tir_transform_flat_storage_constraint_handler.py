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
from tvm.contrib.edgex.tir.transform import FlatStorageConstraintHandler


# fmt: off
@T.prim_func
def idma_example() -> None:
    x_dm = T.allocate([128], "float16", "dm")
    x = T.allocate([128], "float16", "iobuf")
    T.evaluate(T.nnp_idma_load(T.type_annotation(dtype="float16"),
        T.tvm_access_ptr(T.type_annotation(dtype="float16"), x.data, 0, 128, 2, dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="float16"), x_dm.data, 0, 128, 1, dtype="handle"),
        "ci_w_idma=5", dtype=""))


@T.prim_func
def idma_example_rewritten() -> None:
    x_dm = T.allocate([160], "float16", "dm")
    T.attr(x_dm.data, "storage_alignment", 80)
    x = T.allocate([128], "float16", "iobuf")
    T.evaluate(T.nnp_idma_load(T.type_annotation(dtype="float16"),
        T.tvm_access_ptr(T.type_annotation(dtype="float16"), x.data, 0, 128, 2, dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="float16"), x_dm.data, 0, 160, 1, dtype="handle"),
        "ci_w_idma=5", dtype=""))  # align=5*16


@T.prim_func
def wdma_example() -> None:
    w_dm = T.allocate([200], "float16", "dm")
    w = T.allocate([200], "float16", "iobuf")
    T.evaluate(T.nnp_wdma_load(T.type_annotation(dtype="float16"),
        T.tvm_access_ptr(T.type_annotation(dtype="float16"), w.data, 0, 200, 2, dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="float16"), w_dm.data, 0, 200, 1, dtype="handle"),
        "cube_enable_wdma=1", "rotate_en_wdma=0", dtype=""))


@T.prim_func
def wdma_example_rewritten() -> None:
    w_dm = T.allocate([256], "float16", "dm")
    T.attr(w_dm.data, "storage_alignment", 512)
    w = T.allocate([200], "float16", "iobuf")
    T.evaluate(T.nnp_wdma_load(T.type_annotation(dtype="float16"),
        T.tvm_access_ptr(T.type_annotation(dtype="float16"), w.data, 0, 200, 2, dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="float16"), w_dm.data, 0, 256, 1, dtype="handle"),
        "cube_enable_wdma=1", "rotate_en_wdma=0", dtype=""))  # align=(1+1)*256


@T.prim_func
def bdma_example() -> None:
    b_dm = T.allocate([7], "float16", "dm")
    b = T.allocate([7], "float16", "iobuf")
    T.evaluate(T.nnp_bdma_load(T.type_annotation(dtype="float16"),
        T.tvm_access_ptr(T.type_annotation(dtype="float16"), b.data, 0, 7, 2, dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="float16"), b_dm.data, 0, 7, 1, dtype="handle"),
        dtype=""))


@T.prim_func
def bdma_example_rewritten() -> None:
    b_dm = T.allocate([8], "float16", "dm")
    T.attr(b_dm.data, "storage_alignment", 16)
    b = T.allocate([7], "float16", "iobuf")
    T.evaluate(T.nnp_bdma_load(T.type_annotation(dtype="float16"),
        T.tvm_access_ptr(T.type_annotation(dtype="float16"), b.data, 0, 7, 2, dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="float16"), b_dm.data, 0, 8, 1, dtype="handle"),
        dtype=""))


@T.prim_func
def odma_example() -> None:
    y_dm = T.allocate([100], "float16", "dm")
    y = T.allocate([100], "float16", "iobuf")
    T.evaluate(T.nnp_odma_store(T.type_annotation(dtype="float16"),
        T.tvm_access_ptr(T.type_annotation(dtype="float16"), y_dm.data, 0, 100, 2, dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="float16"), y.data, 0, 100, 1, dtype="handle"),
        dtype=""))

@T.prim_func
def odma_example_rewritten() -> None:
    y_dm = T.allocate([128], "float16", "dm")
    T.attr(y_dm.data, "storage_alignment", 128)
    y = T.allocate([100], "float16", "iobuf")
    T.evaluate(T.nnp_odma_store(T.type_annotation(dtype="float16"),
        T.tvm_access_ptr(T.type_annotation(dtype="float16"), y_dm.data, 0, 128, 2, dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="float16"), y.data, 0, 100, 1, dtype="handle"),
        dtype=""))  # algin=128

# fmt: on


def do_test(func, expected):
    mod = IRModule.from_expr(func)
    mod = FlatStorageConstraintHandler()(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected, True)


def test_flat_storage_constraint_handler():
    do_test(idma_example, idma_example_rewritten)
    do_test(wdma_example, wdma_example_rewritten)
    do_test(bdma_example, bdma_example_rewritten)
    do_test(odma_example, odma_example_rewritten)


if __name__ == "__main__":
    test_flat_storage_constraint_handler()
