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
from tvm.contrib.edgex.tir.transform import InlinePrimFuncCalls


@T.prim_func
def myadd(a: T.handle, b: T.handle):
    A = T.match_buffer(a, [128], "float32")
    B = T.match_buffer(b, [128], "float32")
    with T.block("root"):
        A2 = T.alloc_buffer([128], "float32")
        for i in range(128):
            A2[i] = A[i]
        for i in range(128):
            B[i] = A2[i] + 1.0


@T.prim_func
def call_multiple_primfunc_with_extern(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128], "float32")
    B = T.match_buffer(b, [128], "float32")
    C = T.match_buffer(c, [128], "float32")
    T.evaluate(T.call_extern("f1", A.data, B.data, dtype=""))
    T.evaluate(T.call_extern("f2", B.data, C.data, dtype=""))


@T.prim_func
def call_multiple_primfunc_with_extern_inlined(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128], dtype="float32")
    B = T.match_buffer(b, [128], dtype="float32")
    C = T.match_buffer(c, [128], dtype="float32")
    A2 = T.alloc_buffer([128], dtype="float32")
    A2_1 = T.alloc_buffer([128], dtype="float32")
    for i in T.serial(0, 128):
        A2[i] = A[i]
    for i in T.serial(0, 128):
        B[i] = A2[i] + 1.0
    for i in T.serial(0, 128):
        A2_1[i] = B[i]
    for i in T.serial(0, 128):
        C[i] = A2_1[i] + 1.0


def do_test_inline_primfunc_calls(func, expect, extern_primfuncs=None):
    mod = tvm.IRModule.from_expr(func)
    mod = InlinePrimFuncCalls(extern_primfuncs)(mod)
    tvm.ir.assert_structural_equal(mod["main"], expect, True)


def test_inline_primfunc_call_with_extern():
    do_test_inline_primfunc_calls(
        call_multiple_primfunc_with_extern,
        call_multiple_primfunc_with_extern_inlined,
        {"f1": myadd, "f2": myadd},
    )


if __name__ == "__main__":
    test_inline_primfunc_call_with_extern()
