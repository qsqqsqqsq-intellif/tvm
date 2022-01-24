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
# pylint: disable=unused-argument,inconsistent-return-statements
# pylint: disable=invalid-name
"""pattern match"""

import numpy
from tvm import relay
from tvm.relay.dataflow_pattern import is_op, wildcard, is_constant
from tvm.relay.expr_functor import ExprMutator

PATTERNS = []


class Conv2dBiasAdd(ExprMutator):
    """Conv2dBiasAdd"""

    conv_node = is_op("nn.conv2d")(wildcard(), wildcard())
    bias_node = is_op("nn.bias_add")(conv_node, wildcard())

    def __init__(self, mod):
        super().__init__()
        mod["main"] = self.visit(mod["main"])
        self.new_mod = relay.transform.InferType()(mod)

    def visit_call(self, call):
        new_args = [self.visit(arg) for arg in call.args]
        new_call = relay.Call(call.op, new_args, call.attrs, call.type_args, call.span)

        if self.bias_node.match(new_call):
            a0 = relay.var("arg0_")
            a1 = relay.var("arg1_")
            a2 = relay.var("arg2_")

            conv2d = relay.nn.conv2d(a0, a1, **dict(new_call.args[0].attrs))
            bias_add = relay.nn.bias_add(conv2d, a2, **dict(new_call.attrs))
            new_fn = relay.Function([a0, a1, a2], bias_add)
            new_fn = new_fn.with_attr("Composite", "conv2d_bias_add")
            new_fn = new_fn.with_attr("Primitive", 1)

            arg0 = new_call.args[0].args[0]
            arg1 = new_call.args[0].args[1]
            arg2 = new_call.args[1]
            new_call = relay.Call(new_fn, [arg0, arg1, arg2])
            return new_call

        return new_call


class Conv3dBiasAdd(ExprMutator):
    """Conv3dBiasAdd"""

    conv_node = is_op("nn.conv3d")(wildcard(), wildcard())
    bias_node = is_op("nn.bias_add")(conv_node, wildcard())

    def __init__(self, mod):
        super().__init__()
        mod["main"] = self.visit(mod["main"])
        self.new_mod = relay.transform.InferType()(mod)

    def visit_call(self, call):
        new_args = [self.visit(arg) for arg in call.args]
        new_call = relay.Call(call.op, new_args, call.attrs, call.type_args, call.span)

        if self.bias_node.match(new_call):
            a0 = relay.var("arg0_")
            a1 = relay.var("arg1_")
            a2 = relay.var("arg2_")

            conv3d = relay.nn.conv3d(a0, a1, **dict(new_call.args[0].attrs))
            bias_add = relay.nn.bias_add(conv3d, a2, **dict(new_call.attrs))
            new_fn = relay.Function([a0, a1, a2], bias_add)
            new_fn = new_fn.with_attr("Composite", "conv3d_bias_add")
            new_fn = new_fn.with_attr("Primitive", 1)

            arg0 = new_call.args[0].args[0]
            arg1 = new_call.args[0].args[1]
            arg2 = new_call.args[1]
            new_call = relay.Call(new_fn, [arg0, arg1, arg2])
            return new_call

        return new_call


class DenseBiasAdd(ExprMutator):
    """DenseBiasAdd"""

    dense_node = is_op("nn.dense")(wildcard(), wildcard())
    bias_node = is_op("nn.bias_add")(dense_node, wildcard())

    def __init__(self, mod):
        super().__init__()
        mod["main"] = self.visit(mod["main"])
        self.new_mod = relay.transform.InferType()(mod)

    def visit_call(self, call):
        new_args = [self.visit(arg) for arg in call.args]
        new_call = relay.Call(call.op, new_args, call.attrs, call.type_args, call.span)

        if self.bias_node.match(new_call):
            a0 = relay.var("arg0_")
            a1 = relay.var("arg1_")
            a2 = relay.var("arg2_")

            dense = relay.nn.dense(a0, a1, **dict(new_call.args[0].attrs))
            bias_add = relay.nn.bias_add(dense, a2, **dict(new_call.attrs))
            new_fn = relay.Function([a0, a1, a2], bias_add)
            new_fn = new_fn.with_attr("Composite", "dense_bias_add")
            new_fn = new_fn.with_attr("Primitive", 1)

            arg0 = new_call.args[0].args[0]
            arg1 = new_call.args[0].args[1]
            arg2 = new_call.args[1]
            new_call = relay.Call(new_fn, [arg0, arg1, arg2])
            return new_call

        return new_call


class HardSwish(ExprMutator):
    """HardSwish"""

    x = wildcard()
    add = is_op("add")(x, is_constant())
    clip = is_op("clip")(add)
    divide = is_op("divide")(clip, is_constant())
    multiply = is_op("multiply")(x, divide)

    def __init__(self, mod):
        super().__init__()
        mod["main"] = self.visit(mod["main"])
        self.new_mod = relay.transform.InferType()(mod)

    def visit_call(self, call):
        new_args = [self.visit(arg) for arg in call.args]
        new_call = relay.Call(call.op, new_args, call.attrs, call.type_args, call.span)

        if self.multiply.match(new_call):
            divide_node = new_call.args[1]
            clip_node = divide_node.args[0]
            add_node = clip_node.args[0]
            cond1 = add_node.args[1].data.asnumpy() == 3
            cond2 = clip_node.attrs.a_min == 0 and clip_node.attrs.a_max == 6
            cond3 = divide_node.args[1].data.asnumpy() == 6
            if cond1 and cond2 and cond3:
                a0 = relay.var("arg0_")

                add = relay.add(a0, relay.const(3, "float32"))
                clip = relay.clip(add, **dict(clip_node.attrs))
                divide = relay.multiply(clip, relay.const(1 / 6, "float32"))
                multiply = relay.multiply(a0, divide)
                new_fn = relay.Function([a0], multiply)
                new_fn = new_fn.with_attr("Composite", "hard_swish")
                new_fn = new_fn.with_attr("Primitive", 1)

                arg0 = add_node.args[0]
                new_call = relay.Call(new_fn, [arg0])
                return new_call

        return new_call


class HardSigmoid(ExprMutator):
    """HardSigmoid"""

    x = wildcard()
    add = is_op("add")(x, is_constant())
    clip = is_op("clip")(add)
    divide = is_op("divide")(clip, is_constant())

    def __init__(self, mod):
        super().__init__()
        mod["main"] = self.visit(mod["main"])
        self.new_mod = relay.transform.InferType()(mod)

    def visit_call(self, call):
        new_args = [self.visit(arg) for arg in call.args]
        new_call = relay.Call(call.op, new_args, call.attrs, call.type_args, call.span)

        if self.divide.match(new_call):
            clip_node = new_call.args[0]
            add_node = clip_node.args[0]
            cond1 = add_node.args[1].data.asnumpy() == 3
            cond2 = clip_node.attrs.a_min == 0 and clip_node.attrs.a_max == 6
            cond3 = new_call.args[1].data.asnumpy() == 6
            if cond1 and cond2 and cond3:
                a0 = relay.var("arg0_")

                add = relay.add(a0, relay.const(3, "float32"))
                clip = relay.clip(add, **dict(clip_node.attrs))
                divide = relay.multiply(clip, relay.const(1 / 6, "float32"))
                new_fn = relay.Function([a0], divide)
                new_fn = new_fn.with_attr("Composite", "hard_sigmoid")
                new_fn = new_fn.with_attr("Primitive", 1)

                arg0 = add_node.args[0]
                new_call = relay.Call(new_fn, [arg0])
                return new_call

        return new_call


class LayerNorm(ExprMutator):
    """LayerNorm"""

    x = wildcard()
    mean = is_op("mean")(x)
    subtract = is_op("subtract")(x, mean)
    variance = is_op("variance")(x, mean)
    add1 = is_op("add")(variance, wildcard())
    sqrt = is_op("sqrt")(add1)
    divide = is_op("divide")(subtract, sqrt)
    multiply = is_op("multiply")(divide, wildcard())
    add2 = is_op("add")(multiply, wildcard())

    def __init__(self, mod):
        super().__init__()
        mod["main"] = self.visit(mod["main"])
        self.new_mod = relay.transform.InferType()(mod)

    def visit_call(self, call):
        new_args = [self.visit(arg) for arg in call.args]
        new_call = relay.Call(call.op, new_args, call.attrs, call.type_args, call.span)

        if self.add2.match(new_call):
            multiply_node = new_call.args[0]
            divide_node = multiply_node.args[0]
            sqrt_node = divide_node.args[1]
            add_node = sqrt_node.args[0]
            variance_node = add_node.args[0]
            mean_node = variance_node.args[1]

            # a0 = relay.var("arg0_")

            # mean = relay.mean(a0, **dict(mean_node.attrs))
            # subtract = relay.subtract(a0, mean)
            # variance = relay.variance(a0, **dict(variance_node.attrs))
            # add1 = relay.add(variance, add_node.args[1])
            # sqrt = relay.sqrt(add1)
            # divide = relay.divide(subtract, sqrt)
            # multiply = relay.multiply(divide, multiply_node.args[1])
            # add2 = relay.add(multiply, new_call.args[1])

            # new_fn = relay.Function([a0], add2)
            # new_fn = new_fn.with_attr("Composite", "layer_norm")
            # new_fn = new_fn.with_attr("Primitive", 1)

            # new_call = relay.Call(new_fn, [variance_node.args[0]])

            data = variance_node.args[0]
            gamma = multiply_node.args[1]
            beta = new_call.args[1]
            axis = [i.value for i in mean_node.attrs["axis"]][0]
            epsilon = add_node.args[1].data.asnumpy().item()
            new_call = relay.nn.layer_norm(
                data, gamma, beta, axis, epsilon, center=True, scale=True
            )
            return new_call

        return new_call


class GELU(ExprMutator):
    """GELU"""

    x = wildcard()
    multiply1 = is_op("multiply")(x, wildcard())
    erf = is_op("erf")(multiply1)
    multiply2 = is_op("multiply")(erf, wildcard())
    add = is_op("add")(wildcard(), multiply2)
    multiply3 = is_op("multiply")(x, add)

    def __init__(self, mod):
        super().__init__()
        mod["main"] = self.visit(mod["main"])
        self.new_mod = relay.transform.InferType()(mod)

    def visit_call(self, call):
        new_args = [self.visit(arg) for arg in call.args]
        new_call = relay.Call(call.op, new_args, call.attrs, call.type_args, call.span)

        if self.multiply3.match(new_call):
            add_node = new_call.args[1]
            multiply_node1 = add_node.args[1]
            erf_node = multiply_node1.args[0]
            multiply_node2 = erf_node.args[0]

            cond1 = round(multiply_node2.args[1].data.asnumpy().item(), 7) == round(2 ** -0.5, 7)
            cond2 = multiply_node1.args[1].data.asnumpy() == 0.5
            cond3 = add_node.args[0].data.asnumpy() == 0.5
            if cond1 and cond2 and cond3:
                a0 = relay.var("arg0_")

                # multiply1 = relay.multiply(a0, multiply_node2.args[1])
                # erf = relay.erf(multiply1)
                # multiply2 = relay.multiply(erf, multiply_node1.args[1])
                # add = relay.add(add_node.args[0], multiply2)
                # multiply3 = relay.multiply(a0, add)

                # 近似计算1
                # multiply3=a0 * relay.sigmoid(relay.const(1.702, numpy.float32) * a0)
                # 近似计算2
                multiply3 = (
                    relay.const(0.5, numpy.float32)
                    * a0
                    * (
                        relay.const(1, numpy.float32)
                        + relay.tanh(
                            relay.const(numpy.sqrt(2 / numpy.pi), numpy.float32)
                            * (
                                a0
                                + relay.const(0.044715, numpy.float32)
                                * relay.power(a0, relay.const(3, numpy.float32))
                            )
                        )
                    )
                )

                new_fn = relay.Function([a0], multiply3)
                new_fn = new_fn.with_attr("Composite", "GELU")
                new_fn = new_fn.with_attr("Primitive", 1)

                input_node = multiply_node2.args[0]
                new_call = relay.Call(new_fn, [input_node])
                return new_call

        return new_call


class Swish(ExprMutator):
    """Swish"""

    x = wildcard()
    sigmoid = is_op("sigmoid")(x)
    mul = is_op("multiply")(x, sigmoid)

    def __init__(self, mod):
        super().__init__()
        mod["main"] = self.visit(mod["main"])
        self.new_mod = relay.transform.InferType()(mod)

    def visit_call(self, call):
        new_args = [self.visit(arg) for arg in call.args]
        new_call = relay.Call(call.op, new_args, call.attrs, call.type_args, call.span)

        if self.mul.match(new_call):
            arg0 = new_call.args[0]
            a0 = relay.var("arg0_")
            sigmoid = relay.sigmoid(a0)
            mul = relay.multiply(a0, sigmoid)
            new_fn = relay.Function([a0], mul)
            new_fn = new_fn.with_attr("Composite", "swish")
            new_fn = new_fn.with_attr("Primitive", 1)
            new_call = relay.Call(new_fn, [arg0])
            return new_call

        return new_call


class HighDimensionDenseAdd(ExprMutator):
    """HighDimensionDenseAdd"""

    x = wildcard()
    batch_matmul = is_op("nn.batch_matmul")(x, is_constant())
    add = is_op("add")(batch_matmul, is_constant())

    def __init__(self, mod):
        super().__init__()
        mod["main"] = self.visit(mod["main"])
        self.new_mod = relay.transform.InferType()(mod)

    def visit_call(self, call):
        new_args = [self.visit(arg) for arg in call.args]
        new_call = relay.Call(call.op, new_args, call.attrs, call.type_args, call.span)

        if self.add.match(new_call):
            batch_matmul_node = new_call.args[0]

            cond1 = len(call.args[0].args[0].checked_type.shape) > 2
            cond2 = len(new_call.args[1].data.asnumpy().shape) == 1
            if cond1 and cond2:
                a0 = relay.var("arg0_")
                a1 = relay.var("arg1_")
                a2 = relay.var("arg2_")

                batch_matmul = relay.nn.batch_matmul(a0, a1)
                add = relay.add(batch_matmul, a2)

                new_fn = relay.Function([a0, a1, a2], add)
                new_fn = new_fn.with_attr("Composite", "high_dimension_dense_add")
                new_fn = new_fn.with_attr("Primitive", 1)

                arg0 = batch_matmul_node.args[0]
                arg1 = batch_matmul_node.args[1]
                arg2 = new_call.args[1]
                new_call = relay.Call(new_fn, [arg0, arg1, arg2])
                return new_call

        return new_call


class HighDimensionDense(ExprMutator):
    """HighDimensionDense"""

    x = wildcard()
    batch_matmul = is_op("nn.batch_matmul")(x, is_constant())

    def __init__(self, mod):
        super().__init__()
        mod["main"] = self.visit(mod["main"])
        self.new_mod = relay.transform.InferType()(mod)

    def visit_call(self, call):
        new_args = [self.visit(arg) for arg in call.args]
        new_call = relay.Call(call.op, new_args, call.attrs, call.type_args, call.span)

        if self.batch_matmul.match(new_call):
            cond1 = len(call.args[0].checked_type.shape) > 2
            if cond1:
                a0 = relay.var("arg0_")
                a1 = relay.var("arg1_")

                batch_matmul = relay.nn.batch_matmul(a0, a1)

                new_fn = relay.Function([a0, a1], batch_matmul)
                new_fn = new_fn.with_attr("Composite", "high_dimension_dense")
                new_fn = new_fn.with_attr("Primitive", 1)

                arg0 = new_call.args[0]
                arg1 = new_call.args[1]
                new_call = relay.Call(new_fn, [arg0, arg1])
                return new_call

        return new_call


def pattern_match(mod):
    """pattern_match"""
    mod = Conv2dBiasAdd(mod).new_mod
    mod = Conv3dBiasAdd(mod).new_mod
    mod = DenseBiasAdd(mod).new_mod
    mod = HardSwish(mod).new_mod
    mod = HardSigmoid(mod).new_mod
    mod = LayerNorm(mod).new_mod
    mod = GELU(mod).new_mod
    mod = Swish(mod).new_mod
    mod = HighDimensionDenseAdd(mod).new_mod
    mod = HighDimensionDense(mod).new_mod
    return mod
