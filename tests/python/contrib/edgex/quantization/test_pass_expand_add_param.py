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
"""Test fuse add pass"""
import tvm
from tvm import relay
from tvm.relay import transform
import numpy as np
from tvm.relay.quantization.pre_processes import ExpandAddParam


def run_opt_pass(expr, passes):
    passes = passes if isinstance(passes, list) else [passes]
    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def test_expand_add_param_nhwc():
    weight = relay.const(np.random.rand(3, 3, 64, 64), "float32")
    add_w = relay.const(np.array(3, dtype="float32"))

    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )

        # y = relay.nn.bias_add(y, bias_w, 3)
        y = relay.add(y, add_w)
        y = relay.Function([x], y)
        return y

    a = before()
    a = run_opt_pass(a, [transform.InferType(), ExpandAddParam()])
    print(a)


def test_expand_add_param_nchw():
    weight = relay.const(np.random.rand(64, 64, 3, 3), "float32")
    add_w = relay.const(np.array(3, dtype="float32"))

    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )

        y = relay.add(y, add_w)
        y = relay.Function([x], y)
        return y

    a = before()
    a = run_opt_pass(a, [transform.InferType(), ExpandAddParam()])
    print(a)


if __name__ == "__main__":
    test_expand_add_param_nhwc()
    test_expand_add_param_nchw()
