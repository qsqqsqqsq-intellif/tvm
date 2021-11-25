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
from tvm.contrib.edgex.tir.transform import InjectHandShakeIntrin


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func)
    mod = InjectHandShakeIntrin()(mod)
    # To flatten the seqstmt.
    mod = tvm.tir.transform.RemoveNoOp()(mod)
    tvm.ir.assert_structural_equal(mod["main"], transformed, True)


@T.prim_func
def eidma_vidma_load(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (6, 6), "float32")
    C = T.match_buffer(c, (6, 6), "float32")
    for i in range(0, 6):
        T.evaluate(
            T.nnp_eidma_load(
                "float32",
                A.access_ptr("r", offset=0),
                C.access_ptr("w", offset=0),
                dtype="handle",
            )
        )
        T.evaluate(
            T.nnp_vidma_load(
                "float32",
                A.access_ptr("r", offset=0),
                C.access_ptr("w", offset=0),
                dtype="handle",
            )
        )


@T.prim_func
def eidma_vidma_load_sync(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (6, 6), "float32")
    C = T.match_buffer(c, (6, 6), "float32")
    for i in T.serial(0, 6):
        T.evaluate(
            T.nnp_eidma_load(
                "float32",
                T.tvm_access_ptr(
                    T.type_annotation(dtype="float32"), A.data, 0, 36, 1, dtype="handle"
                ),
                T.tvm_access_ptr(
                    T.type_annotation(dtype="float32"), C.data, 0, 36, 2, dtype="handle"
                ),
                dtype="handle",
            )
        )
        T.evaluate(T.nnp_sync("eidma", "ub", "cu", dtype=""))
        T.evaluate(T.nnp_sync("cu", "wo", "eidma", dtype=""))
        T.evaluate(T.nnp_sync("eidma", "ub", "vidma0", dtype=""))
        T.evaluate(T.nnp_sync("vidma", "wo", "eidma", dtype=""))
        T.evaluate(T.nnp_sync("eidma", "wo", "vidma0", dtype=""))
        T.evaluate(T.nnp_sync("vidma", "ub", "eidma", dtype=""))
        T.evaluate(
            T.nnp_vidma_load(
                "float32",
                T.tvm_access_ptr(
                    T.type_annotation(dtype="float32"), A.data, 0, 36, 1, dtype="handle"
                ),
                T.tvm_access_ptr(
                    T.type_annotation(dtype="float32"), C.data, 0, 36, 2, dtype="handle"
                ),
                dtype="handle",
            )
        )
        T.evaluate(T.nnp_sync("vidma", "ub", "vcu", dtype=""))
        T.evaluate(T.nnp_sync("vcu", "wo", "vidma", dtype=""))


@T.prim_func
def vodma_eodma_store(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (6, 6), "float32")
    C = T.match_buffer(c, (6, 6), "float32")
    for i in range(0, 6):
        T.evaluate(
            T.nnp_vodma_store(
                "float32",
                A.access_ptr("r", offset=0),
                C.access_ptr("w", offset=0),
                dtype="handle",
            )
        )
        T.evaluate(
            T.nnp_eodma_store(
                "float32",
                A.access_ptr("r", offset=0),
                C.access_ptr("w", offset=0),
                dtype="handle",
            )
        )


@T.prim_func
def vodma_eodma_store_sync(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (6, 6), "float32")
    C = T.match_buffer(c, (6, 6), "float32")
    for i in range(0, 6):
        T.evaluate(
            T.nnp_vodma_store(
                "float32",
                T.tvm_access_ptr(
                    T.type_annotation(dtype="float32"), A.data, 0, 36, 1, dtype="handle"
                ),
                T.tvm_access_ptr(
                    T.type_annotation(dtype="float32"), C.data, 0, 36, 2, dtype="handle"
                ),
                dtype="handle",
            )
        )
        T.evaluate(T.nnp_sync("vodma", "ub", "vcu", dtype=""))
        T.evaluate(T.nnp_sync("vcu", "wo", "vodma", dtype=""))
        T.evaluate(T.nnp_sync("vodma", "ub", "eodma", dtype=""))
        T.evaluate(T.nnp_sync("eodma", "wo", "vodma0", dtype=""))
        T.evaluate(
            T.nnp_eodma_store(
                "float32",
                T.tvm_access_ptr(
                    T.type_annotation(dtype="float32"), A.data, 0, 36, 1, dtype="handle"
                ),
                T.tvm_access_ptr(
                    T.type_annotation(dtype="float32"), C.data, 0, 36, 2, dtype="handle"
                ),
                dtype="handle",
            )
        )
        T.evaluate(T.nnp_sync("eodma", "ub", "cu", dtype=""))
        T.evaluate(T.nnp_sync("cu", "wo", "eodma", dtype=""))


@T.prim_func
def eidma_idma_load(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (6, 6), "float32")
    C = T.match_buffer(c, (6, 6), "float32")
    for i in range(0, 6):
        T.evaluate(
            T.nnp_eidma_load(
                "float32",
                A.access_ptr("r", offset=0),
                C.access_ptr("w", offset=0),
                dtype="handle",
            )
        )
        T.evaluate(
            T.nnp_idma_load(
                "float32",
                A.access_ptr("r", offset=0),
                C.access_ptr("w", offset=0),
                dtype="handle",
            )
        )


@T.prim_func
def eidma_idma_load_sync(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (6, 6), "float32")
    C = T.match_buffer(c, (6, 6), "float32")
    for i in range(0, 6):
        T.evaluate(
            T.nnp_eidma_load(
                "float32",
                T.tvm_access_ptr(
                    T.type_annotation(dtype="float32"), A.data, 0, 36, 1, dtype="handle"
                ),
                T.tvm_access_ptr(
                    T.type_annotation(dtype="float32"), C.data, 0, 36, 2, dtype="handle"
                ),
                dtype="handle",
            )
        )
        T.evaluate(T.nnp_sync("eidma", "ub", "cu", dtype=""))
        T.evaluate(T.nnp_sync("cu", "wo", "eidma", dtype=""))
        T.evaluate(
            T.nnp_idma_load(
                "float32",
                T.tvm_access_ptr(
                    T.type_annotation(dtype="float32"), A.data, 0, 36, 1, dtype="handle"
                ),
                T.tvm_access_ptr(
                    T.type_annotation(dtype="float32"), C.data, 0, 36, 2, dtype="handle"
                ),
                dtype="handle",
            )
        )


@T.prim_func
def odma_eodma_store(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (6, 6), "float32")
    C = T.match_buffer(c, (6, 6), "float32")
    for i in range(0, 6):
        T.evaluate(
            T.nnp_odma_store(
                "float32",
                A.access_ptr("r", offset=0),
                C.access_ptr("w", offset=0),
                dtype="handle",
            )
        )
        T.evaluate(
            T.nnp_eodma_store(
                "float32",
                A.access_ptr("r", offset=0),
                C.access_ptr("w", offset=0),
                dtype="handle",
            )
        )


@T.prim_func
def odma_eodma_store_sync(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (6, 6), "float32")
    C = T.match_buffer(c, (6, 6), "float32")
    for i in range(0, 6):
        T.evaluate(
            T.nnp_odma_store(
                "float32",
                T.tvm_access_ptr(
                    T.type_annotation(dtype="float32"), A.data, 0, 36, 1, dtype="handle"
                ),
                T.tvm_access_ptr(
                    T.type_annotation(dtype="float32"), C.data, 0, 36, 2, dtype="handle"
                ),
                dtype="handle",
            )
        )
        T.evaluate(T.nnp_sync("odma", "ub", "eodma", dtype=""))
        T.evaluate(T.nnp_sync("eodma", "wo", "odma", dtype=""))
        T.evaluate(
            T.nnp_eodma_store(
                "float32",
                T.tvm_access_ptr(
                    T.type_annotation(dtype="float32"), A.data, 0, 36, 1, dtype="handle"
                ),
                T.tvm_access_ptr(
                    T.type_annotation(dtype="float32"), C.data, 0, 36, 2, dtype="handle"
                ),
                dtype="handle",
            )
        )
        T.evaluate(T.nnp_sync("eodma", "ub", "cu", dtype=""))
        T.evaluate(T.nnp_sync("cu", "wo", "eodma", dtype=""))


def test_eidma_vidma_load_sync():
    _check(eidma_vidma_load, eidma_vidma_load_sync)


def test_vodma_eodma_store_sync():
    _check(vodma_eodma_store, vodma_eodma_store_sync)


def test_eidma_idma_load_sync():
    _check(eidma_idma_load, eidma_idma_load_sync)


def test_odma_eodma_store_sync():
    _check(odma_eodma_store, odma_eodma_store_sync)


if __name__ == "__main__":
    test_eidma_vidma_load_sync()
    test_vodma_eodma_store_sync()
    test_eidma_idma_load_sync()
    test_odma_eodma_store_sync()
