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
import pytest
import tvm
from tvm import relay
from tvm.ir.module import IRModule
from tvm.contrib.edgex.relay.transform import ConvertDepthwiseConv2D
from tvm.relay.build_module import bind_params_by_name
import numpy as np


def do_test(before, expected, params_before):
    mod_before = relay.transform.InferType()(IRModule.from_expr(before))
    mod_expect = relay.transform.InferType()(IRModule.from_expr(expected))
    mod_after, params_after = ConvertDepthwiseConv2D()(mod_before, params_before)
    assert tvm.ir.structural_equal(mod_expect, mod_after)
    executor = relay.create_executor()
    result_before = executor.evaluate(bind_params_by_name(mod_before["main"], params_before))()
    result_after = executor.evaluate(bind_params_by_name(mod_after["main"], params_after))()
    print(mod_after.astext(False))
    tvm.testing.assert_allclose(result_before.numpy(), result_after.numpy())


def test_small_group():
    def before():
        x = relay.var("x", shape=(1, 3, 32, 32), dtype="int8")
        w = relay.var("w", shape=(3, 1, 1, 1), dtype="int8")
        y = relay.nn.conv2d(x, w, padding=(0, 0, 0, 0), strides=[1, 1], out_dtype="int32", groups=3)
        return relay.Function([x, w], y)

    def expected():
        x = relay.var("x", shape=(1, 3, 32, 32), dtype="int8")
        w = relay.var("w", shape=(3, 3, 1, 1), dtype="int8")
        y = relay.nn.conv2d(x, w, padding=(0, 0, 0, 0), strides=[1, 1], out_dtype="int32")
        return relay.Function([x, w], y)

    params_before = {
        "x": tvm.nd.array(np.random.randint(-10, 10, [1, 3, 32, 32]).astype("int8")),
        "w": tvm.nd.array(np.random.randint(-10, 10, [3, 1, 1, 1]).astype("int8")),
    }
    do_test(before(), expected(), params_before)


def test_large_group_divided_by_16():
    def before():
        x = relay.var("x", shape=(1, 960, 7, 7), dtype="int8")
        w = relay.var("w", shape=(960, 1, 3, 3), dtype="int8")
        y = relay.nn.conv2d(
            x, w, padding=(0, 0, 0, 0), strides=[1, 1], out_dtype="int32", groups=960
        )
        return relay.Function([x, w], y)

    def expected():
        x = relay.var("x", shape=(1, 960, 7, 7), dtype="int8")
        w = relay.var("w", shape=(960, 16, 3, 3), dtype="int8")
        y = relay.nn.conv2d(
            x, w, padding=(0, 0, 0, 0), strides=[1, 1], out_dtype="int32", groups=60
        )
        return relay.Function([x, w], y)

    params_before = {
        "x": tvm.nd.array(np.random.randint(-10, 10, [1, 960, 7, 7]).astype("int8")),
        "w": tvm.nd.array(np.random.randint(-10, 10, [960, 1, 3, 3]).astype("int8")),
    }
    do_test(before(), expected(), params_before)


if __name__ == "__main__":
    pytest.main([__file__])
