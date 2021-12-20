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
from tvm import te

from tvm import relay
from tvm.relay.op import register_alter_op_layout
from tvm.relay import transform, analysis
import numpy as np
import torch
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.quantization.relay_transforms import *
from tvm.relay.transform.transform import InferType


def run_opt_pass(expr, passes):
    passes = passes if isinstance(passes, list) else [passes]
    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def test_fuse_single_add():
    add_w = relay.const(np.random.rand(64), "float32")

    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        y = relay.add(x, add_w)
        y = relay.Function([x], y)
        return y

    a = before()
    a = run_opt_pass(a, InferType())
    b = run_opt_pass(a, FuseAdd())
    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_fuse_add_into_biasadd():
    weight = relay.const(np.random.rand(3, 3, 64, 64), "float32")
    bias_w = relay.const(np.random.rand(64), "float32")
    add_w = relay.const(np.random.rand(64), "float32")

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

        y = relay.nn.bias_add(y, bias_w, 3)
        y = relay.add(y, add_w)
        y = relay.Function([x], y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        # weight = relay.const(np.random.rand(3, 3, 64, 64), "float32")
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        bias_wt = relay.add(bias_w, add_w)
        y = relay.nn.bias_add(y, bias_wt, 3)
        y = relay.Function([x], y)
        return y

    a = before()
    a = run_opt_pass(a, FuseAdd())
    b = run_opt_pass(expected(), transform.FoldConstant())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_fuse_conv2d_add_to_biasadd():
    weight = relay.const(np.random.rand(3, 3, 64, 64), "float32")
    bias_w = relay.const(np.random.rand(64), "float32")
    add_w = relay.const(np.random.rand(64), "float32")

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

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        # weight = relay.const(np.random.rand(3, 3, 64, 64), "float32")
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        # bias_wt = relay.add(bias_w, add_w)
        y = relay.nn.bias_add(y, add_w, 3)
        y = relay.Function([x], y)
        return y

    a = before()
    a = run_opt_pass(a, FuseAdd())
    b = run_opt_pass(expected(), transform.FoldConstant())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_fuse_dense_add_to_biasadd():
    weight = relay.const(np.random.rand(48, 64), "float32")
    bias_w = relay.const(np.random.rand(48), "float32")
    add_w = relay.const(np.random.rand(48), "float32")

    def before():
        x = relay.var("x", shape=(32, 64))
        y = relay.nn.dense(x, weight)
        y = relay.add(y, add_w)
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(32, 64))
        y = relay.nn.dense(x, weight)
        y = relay.nn.bias_add(y, add_w, 1)
        y = relay.Function(analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, FuseAdd())
    b = run_opt_pass(expected(), transform.FoldConstant())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_fuse_bn():
    class ConvertMultiplyToConv(ExprMutator):
        def __init__(self):
            super().__init__()

        def visit_call(self, call):
            visited = super().visit_call(call)
            if visited.op == tvm.relay.op.get("multiply") and isinstance(
                visited.args[1], tvm.relay.Constant
            ):

                shape = call.args[1].checked_type.shape
                if len(call._checked_type_.shape) == 4 and len(shape) == 3:
                    channels = call.checked_type.shape[1].value
                    assert channels == shape[0].value
                    conv2d_arg = relay.Constant(
                        tvm.nd.array(
                            visited.args[1].data.asnumpy().reshape([shape[0].value, 1, 1, 1])
                        )
                    )
                    return relay.nn.conv2d(
                        visited.args[0], conv2d_arg, groups=channels, kernel_size=(1, 1)
                    )

            return visited

        def run(self, func):
            self.layout = "NCHW"
            return self.visit(func)

    class dense(torch.nn.Module):
        def __init__(self):
            super(dense, self).__init__()
            self.Conv2d1 = torch.nn.Conv2d(3, 8, 3, 2)
            self.BatchNorm2d1 = torch.nn.BatchNorm2d(8)

            self.Conv2d2 = torch.nn.Conv2d(8, 16, 3, 2)
            self.BatchNorm2d2 = torch.nn.BatchNorm2d(16)

            self.Conv2d31 = torch.nn.Conv2d(16, 32, 3, 2)
            self.BatchNorm2d31 = torch.nn.BatchNorm2d(32)

            self.Conv2d32 = torch.nn.Conv2d(16, 32, 3, 2)
            self.BatchNorm2d32 = torch.nn.BatchNorm2d(32)

            self.Conv2d4 = torch.nn.Conv2d(32, 64, 3, 2)
            self.BatchNorm2d4 = torch.nn.BatchNorm2d(64)

            self.Linear = torch.nn.Linear(10816, 1000, bias=False)

        def forward(self, x):
            x = self.Conv2d1(x)
            x = self.BatchNorm2d1(x)
            x = torch.relu(x)
            x = self.Conv2d2(x)
            x = self.BatchNorm2d2(x)
            x = torch.relu(x)
            x1 = self.Conv2d31(x)
            x1 = self.BatchNorm2d31(x1)
            x2 = self.Conv2d32(x)
            x2 = self.BatchNorm2d32(x2)
            x = x1 + x2
            x = torch.relu(x)
            x = self.Conv2d4(x)
            x = self.BatchNorm2d4(x)
            x = torch.relu(x)
            x = torch.flatten(x, 1)
            x = self.Linear(x)
            x = torch.nn.functional.softmax(x)
            return x

    model = dense()

    def _bind_params(func, params):
        """Bind the params to the expression."""
        name_dict = {}
        for arg in func.params:
            name = arg.name_hint
            if name in name_dict:
                name_dict[name] = None
            else:
                name_dict[name] = arg
        bind_dict = {}
        for k, v in params.items():
            if k not in name_dict:
                continue
            arg = name_dict[k]
            if arg is None:
                raise ValueError("Multiple args in the function have name %s" % k)
            bind_dict[arg] = relay.expr.const(v)
        return relay.expr.bind(func, bind_dict)

    x = torch.randn([1, 3, 224, 224])
    scripted_model = torch.jit.trace(model.eval(), x)
    shape_list = [("input", x.numpy().shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    a = _bind_params(mod["main"], params)
    a = run_opt_pass(a, [transform.SimplifyInference(), transform.FoldConstant(), FuseAdd()])

    a = ConvertMultiplyToConv().run(a)
    a = run_opt_pass(a, [transform.InferType()])
    a = run_opt_pass(a, [FuseAdd(), transform.FoldConstant()])
    print(a)
    pass


if __name__ == "__main__":
    test_fuse_add_into_biasadd()
    test_fuse_conv2d_add_to_biasadd()
    test_fuse_dense_add_to_biasadd()
    test_fuse_bn()
