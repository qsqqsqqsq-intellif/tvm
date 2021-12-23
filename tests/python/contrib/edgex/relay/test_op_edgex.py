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
from tvm import relay
from tvm.relay import create_executor
from tvm.relay.testing import run_infer_type
from tvm.contrib.edgex.relay.op import cast_reinterpret, round_right_shift
import numpy as np


def test_round_right_shift_op():
    """Test round right shift relay op"""
    x = relay.var("data", shape=[4], dtype="int32")
    p = relay.var("param", shape=[4], dtype="int32")
    data = relay.Constant(tvm.nd.array(np.asarray([5, 10, 8, -5], dtype="int32")))
    shift = relay.Constant(tvm.nd.array(np.asarray([1, 2, 0, 1], dtype="int32")))
    y = round_right_shift(x, shift)
    intrp = create_executor()
    op_res = intrp.evaluate(y, {x: data, p: shift})
    ref_res = np.asarray([3, 3, 8, -3], dtype="int32")
    np.testing.assert_equal(op_res.numpy(), ref_res)


def test_cast_reinterpret_op():
    """Test numpy.ndarray.view like relay op cast_reinterpret.
    Due to PrimExpr limitation, tvm does not support 64 bits"""

    def verify_cast_reinterpret(otype, ttype, expand):
        a = relay.var("a", relay.TensorType((10, 4, 1000), otype))
        y = cast_reinterpret(a, ttype)
        yy = run_infer_type(y)
        assert yy.checked_type == relay.TensorType((10, 4, 1000 * expand), ttype)
        data = np.random.randint(
            np.iinfo(otype).min, np.iinfo(otype).max, size=(10, 4, 1000), dtype=otype
        )
        intrp = create_executor()
        op_res = intrp.evaluate(y, {a: relay.const(data)})
        ref_res = data.view(ttype)
        np.testing.assert_equal(op_res.numpy(), ref_res)

        b = relay.var("b", yy.checked_type)
        y = cast_reinterpret(b, otype)
        data = op_res.numpy()
        op_res = intrp.evaluate(y, {b: relay.const(data)})
        ref_res = data.view(otype)
        np.testing.assert_equal(op_res.numpy(), ref_res)

    types = ["int32", "int16", "int8"]
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            verify_cast_reinterpret(types[i], types[j], 1 << (j - i))


if __name__ == "__main__":
    test_round_right_shift_op()
    test_cast_reinterpret_op()
