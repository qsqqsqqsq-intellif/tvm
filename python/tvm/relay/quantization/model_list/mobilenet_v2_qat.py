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

import numpy
import tvm
from tvm import relay
import tvm.relay.quantization
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.quantization.pre_processes import ConvertQnnOps

model_file = "/data/share/demodels-lfs/400T_model_benchmark/mobilenet_v2_1.0_224_quant/mobilenet_v2_1.0_224_quant/mobilenet_v2_1.0_224_quant.tflite"
ctx = tvm.cpu()
target = "llvm"

tflite_model_buf = open(model_file, "rb").read()
try:
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model

    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)


input_tensor = "input"
input_shape = (1, 224, 224, 3)
input_dtype = "uint8"

# parse TFLite model and convert into Relay computation graph
mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict={input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype}
)


def bind_params(func, params):
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


class RemoveCast(ExprMutator):
    """remove cast"""

    def __init__(self, mod):
        super().__init__()
        mod["main"] = self.visit(mod["main"])
        self.new_mod = relay.transform.InferType()(mod)

    def visit_call(self, call):
        visit = super().visit_call(call)

        if visit.op.name == "cast":
            arg0 = visit.args[0]
            dtype = visit.attrs.dtype
            if isinstance(arg0, relay.Constant):
                arg0 = arg0.data.asnumpy()
                arg0 = arg0.astype(dtype)
                arg0 = relay.const(arg0)
                return arg0

            if isinstance(arg0, relay.Call) and arg0.op.name == "cast":
                return relay.cast(arg0.args[0], dtype)

        return visit


class FuseMultiplyAndAvgpool(ExprMutator):
    """FuseMultiplyAndAvgpool"""

    def __init__(self, mod):
        super().__init__()
        mod["main"] = self.visit(mod["main"])
        self.new_mod = relay.transform.InferType()(mod)

    def visit_call(self, call):
        visit = super().visit_call(call)

        if visit.op.name == "nn.avg_pool2d":
            arg0 = visit.args[0]
            if isinstance(arg0, relay.Call) and arg0.op.name == "multiply":
                attrs = visit.attrs
                pool_size = attrs.pool_size
                k_size = pool_size[0] * pool_size[1]
                mul_w = arg0.args[1]
                if isinstance(mul_w, relay.Constant) and mul_w.data.asnumpy() == k_size:
                    return relay.nn.sum_pool2d(
                        arg0.args[0],
                        pool_size=pool_size,
                        strides=attrs.strides,
                        dilation=attrs.dilation,
                        padding=attrs.padding,
                        layout=attrs.layout,
                        out_layout=attrs.out_layout,
                    )

        return visit


class CutGraph(ExprMutator):
    """cut graph"""

    def __init__(self):
        super().__init__()
        self.end = False

    def visit_call(self, call):
        visited = super().visit_call(call)

        if visited.op.name == "qnn.add":
            self.end = True

        if self.end:
            return self.body

        self.body = visited

        return visited

    def visit_function(self, fn):
        params = fn.params
        super().visit_function(fn)

        return relay.Function(params, self.body)

    def run(self, func):
        return self.visit(func)


class FuseAddBias(ExprMutator):
    """fuse add bias"""

    def __init__(self, mod):
        super().__init__()
        mod["main"] = self.visit(mod["main"])
        self.new_mod = relay.transform.FoldConstant()(mod)

    def visit_call(self, call):
        visited = super().visit_call(call)

        if visited.op.name == "nn.bias_add":
            arg0 = call.args[0]

            if (
                isinstance(arg0, relay.Call)
                and arg0.op.name == "add"
                and isinstance(arg0.args[1], relay.Constant)
            ):
                shape = call.checked_type.shape
                channels = shape[call.attrs.axis]

                add_w = arg0.args[1].data.asnumpy()
                if add_w.size == channels:
                    add_w = add_w.reshape(-1)
                    add_w = relay.const(add_w)
                    new_bias_w = relay.add(add_w, call.args[1])

                    return relay.nn.bias_add(
                        visited.args[0].args[0], new_bias_w, axis=call.attrs.axis
                    )

        return visited


func = bind_params(mod["main"], params)
mod = tvm.IRModule.from_expr(func)
mod = relay.transform.InferType()(mod)

mod["main"] = CutGraph().run(mod["main"])

new_mod = ConvertQnnOps()(mod)
new_mod = relay.qnn.transform.CanonicalizeOps()(new_mod)
new_mod = FuseMultiplyAndAvgpool(new_mod).new_mod
new_mod = relay.transform.FoldConstant()(new_mod)

new_mod = FuseAddBias(new_mod).new_mod
new_mod = RemoveCast(new_mod).new_mod


# input_data = numpy.random.uniform(0, 255, (1, 224, 224, 3)).astype(numpy.uint8)
# from tvm.relay.quantization.post_process import extract_module

# extract_module(new_mod, "/data/wangjiuyang/", "mobilenet_v2_qat_clip", {"input": input_data})
