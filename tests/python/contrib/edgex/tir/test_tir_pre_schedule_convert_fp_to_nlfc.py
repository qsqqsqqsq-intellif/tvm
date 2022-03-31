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
# pylint: disable=missing-function-docstring,missing-module-docstring
import tvm
import tvm.testing
import tvm.script.tir as T
from tvm import IRModule
from tvm.contrib.edgex.tir.transform import ConvertFpToNlfc
from tvm.contrib.edgex.arith import extract_nlfc_params


@T.prim_func
def simple_fp_sigmoid(X: T.Buffer[(16,), "float16"], Y: T.Buffer[(16,), "float16"]):
    for i in range(16):
        with T.block():
            vi = T.axis.spatial(16, i)
            Y[vi] = T.sigmoid(X[vi], dtype="float16")


@T.prim_func
def simple_fp_sigmoid_converted(
    tir_nnp_nlfc_sigmoid_non_iter: T.Buffer[(1024,), "int8"],
    X: T.Buffer[(16,), "float16"],
    Y: T.Buffer[(16,), "float16"],
) -> None:
    for i in T.serial(16):
        with T.block():
            vi = T.axis.spatial(16, i)
            T.reads(X[vi], tir_nnp_nlfc_sigmoid_non_iter[0:1024])
            T.writes(Y[vi])
            Y[vi] = T.nnp_nlfc_sigmoid(X[vi], tir_nnp_nlfc_sigmoid_non_iter.data, dtype="float16")


def test_convert_fp_to_nlfc():
    res = ConvertFpToNlfc()(IRModule.from_expr(simple_fp_sigmoid))
    nlfc_params, nlfc_tables, res = extract_nlfc_params(res["main"])
    assert len(nlfc_params) == 1
    assert len(nlfc_tables) == 1
    tvm.ir.assert_structural_equal(res, simple_fp_sigmoid_converted)


if __name__ == "__main__":
    test_convert_fp_to_nlfc()
