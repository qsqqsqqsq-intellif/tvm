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
"""Test convert avg pool pass"""
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.quantization.relay_transforms import *


def run_opt_pass(expr, passes):
    passes = passes if isinstance(passes, list) else [passes]
    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def test_convert_avg_pool2d():
    dshape = (1, 3, 28, 28)
    x = relay.var("x", shape=dshape)
    y = relay.nn.avg_pool2d(x, pool_size=(2, 2), strides=(2, 2), padding=(0, 0))
    func = relay.Function([x], y)
    a = run_opt_pass(func, [transform.InferType(), ConvertAvgpoolToSumpool()])
    # a = run_opt_pass(func, transform.InferType())

    y = relay.nn.sum_pool2d(x, pool_size=(2, 2), strides=(2, 2), padding=(0, 0))
    coef = relay.const(1.0 / 2 / 2, dtype="float32")
    m = relay.multiply(y, coef)
    b = relay.Function([x], m)
    b = run_opt_pass(b, transform.InferType())
    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a) + "\nExpect = \n" + str(b)


def test_convert_global_avg_pool2d():
    dshape = (1, 3, 28, 28)
    x = relay.var("x", shape=dshape)
    y = relay.nn.avg_pool2d(x, pool_size=(2, 2), strides=(2, 2), padding=(0, 0))
    func = relay.Function([x], y)
    a = run_opt_pass(func, [transform.InferType(), ConvertAvgpoolToSumpool()])
    # a = run_opt_pass(func, transform.InferType())

    y = relay.nn.sum_pool2d(x, pool_size=(2, 2), strides=(2, 2), padding=(0, 0))
    coef = relay.const(1.0 / 2 / 2, dtype="float32")
    m = relay.multiply(y, coef)
    b = relay.Function([x], m)
    b = run_opt_pass(b, transform.InferType())
    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a) + "\nExpect = \n" + str(b)


if __name__ == "__main__":
    test_convert_avg_pool2d()
    test_convert_global_avg_pool2d()
