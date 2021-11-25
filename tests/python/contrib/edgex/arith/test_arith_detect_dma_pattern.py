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
from tvm import tir
from tvm.contrib.edgex.arith import detect_reshape_transpose_seq


def check_detect_shape_op(iter_shape, binding_func, expect):
    vars = []
    dom_map = {}
    for i, dim in enumerate(iter_shape):
        v = tir.Var("i%d" % i, "int32")
        vars.append(v)
        dom_map[v] = tvm.ir.Range.from_min_extent(0, dim)
    bindings = binding_func(*vars)
    if not isinstance(bindings, list) and not isinstance(bindings, tuple):
        bindings = [bindings]
    ops = detect_reshape_transpose_seq(vars, bindings, dom_map)
    assert len(ops) == len(expect)
    for k in range(len(ops)):
        op_type, vec = ops[k]
        expect_op_type, expect_vec = expect[k]
        assert op_type == expect_op_type, "Inconsistent at %d: %s\n%s" % (
            k,
            str(ops[k]),
            str(expect[k]),
        )
        assert vec == expect_vec, "Inconsistent at %d: %s\n%s" % (k, str(ops[k]), str(expect[k]))


def test_simple_fuse():
    expect = [("reshape", [15, 8])]
    check_detect_shape_op([15, 8], lambda i, j: i * 8 + j, expect)


def test_simple_split():
    expect = [("reshape", [120])]
    check_detect_shape_op([120], lambda i: (i // 8, i % 8), expect)


def test_simple_broadcast():
    expect = [("reshape", [9, -5]), ("reshape", [45])]
    check_detect_shape_op([45], lambda i: i // 5, expect)
    expect = [("reshape", [-9, 5]), ("reshape", [45])]
    check_detect_shape_op([45], lambda i: i % 5, expect)


def test_packing_unpacking():
    expect = [
        ("reshape", [15, 8, 576]),
        ("transpose", [0, 2, 1]),
        ("reshape", [1, 15, 24, 24, 8]),
    ]
    check_detect_shape_op(
        [1, 15, 24, 24, 8], lambda n, co, h, w, ci: [n, co * 8 + ci, h, w], expect
    )
    expect = [
        ("reshape", [15, 576, 8]),
        ("transpose", [0, 2, 1]),
        ("reshape", [1, 120, 24, 24]),
    ]
    check_detect_shape_op([1, 120, 24, 24], lambda n, c, h, w: [n, c // 8, h, w, c % 8], expect)


if __name__ == "__main__":
    test_simple_fuse()
    test_simple_split()
    test_simple_broadcast()
    test_packing_unpacking()
