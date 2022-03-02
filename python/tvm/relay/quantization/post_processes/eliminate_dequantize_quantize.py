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
# pylint: disable=unused-argument,inconsistent-return-statements,too-many-function-args
"""eliminate_dequantize_quantize"""

from tvm import relay
from tvm.relay.dataflow_pattern import is_op, wildcard
from tvm.relay.expr_functor import ExprMutator


class EliminateDequantizeQuantize(ExprMutator):
    """eliminate dequantize quantize"""

    dequantize = is_op("qnn.dequantize")(wildcard(), wildcard(), wildcard())
    quantize = is_op("qnn.quantize")(dequantize, wildcard(), wildcard())

    def __init__(self, mod):
        super().__init__()
        mod["main"] = self.visit(mod["main"])
        self.new_mod = relay.transform.InferType()(mod)

    def visit_call(self, call):
        new_args = [self.visit(arg) for arg in call.args]
        new_call = relay.Call(call.op, new_args, call.attrs, call.type_args, call.span)

        if self.quantize.match(new_call):
            pre_arg = new_call.args[0]
            cond1 = pre_arg.attrs.axis == new_call.attrs.axis
            cond2 = pre_arg.args[1].data.asnumpy() == new_call.args[1].data.asnumpy()
            cond3 = pre_arg.args[2].data.asnumpy() == new_call.args[2].data.asnumpy()
            if cond1 and cond2.all() and cond3.all():
                return pre_arg.args[0]

        return new_call


def eliminate_dequantize_quantize(mod):
    mod = EliminateDequantizeQuantize(mod).new_mod
    return mod
