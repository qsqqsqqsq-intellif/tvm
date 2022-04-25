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
import sys
import pytest
import tvm
from tvm.contrib.edgex.tir.schedule import EdgexSchedule
from tvm.script import tir as T


def test_inplace_buffer():
    @T.prim_func
    def compute_func(A: T.Buffer[(128,), "int32"], C: T.Buffer[(128,), "int32"]):
        B0 = T.alloc_buffer([128], "int32")
        B1 = T.alloc_buffer([128], "int32")
        for i in range(128):
            with T.block("B0"):
                vi = T.axis.remap("S", [i])
                B0[vi] = A[vi]
        for i in range(128):
            with T.block("B1"):
                vi = T.axis.remap("S", [i])
                B1[vi] = B0[vi] * 16
        for i in range(128):
            with T.block("C"):
                vi = T.axis.remap("S", [i])
                C[vi] = B1[vi]

    @T.prim_func
    def compute_func_after_inplace(A: T.Buffer[(128,), "int32"], C: T.Buffer[(128,), "int32"]):
        B0 = T.alloc_buffer([128], dtype="int32")
        for i in T.serial(128):
            with T.block("B0"):
                vi = T.axis.spatial(128, i)
                B0[vi] = A[vi]
        for i in T.serial(128):
            with T.block("B1"):
                vi = T.axis.spatial(128, i)
                B0[vi] = B0[vi] * 16
        for i in T.serial(128):
            with T.block("C"):
                vi = T.axis.spatial(128, i)
                C[vi] = B0[vi]

    s = EdgexSchedule(compute_func)
    s.inplace_buffer(s.get_block("B1"), 0, 0, unsafe=True)
    tvm.ir.assert_structural_equal(s.mod["main"], compute_func_after_inplace)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
