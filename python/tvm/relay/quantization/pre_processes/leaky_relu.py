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
"""LeakyRelu"""

from tvm import relay
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.dataflow_pattern import is_op, is_tuple_get_item, wildcard


class LeakyRelu(ExprMutator):
    """LeakyRelu"""

    x = wildcard()
    multiply1 = is_op("multiply")(x, wildcard())
    maximum1 = is_op("maximum")(multiply1, x)

    tgi1 = is_tuple_get_item(x)
    multiply2 = is_op("multiply")(tgi1, wildcard())
    tgi2 = is_tuple_get_item(x)
    maximum2 = is_op("maximum")(multiply2, tgi2)

    def __init__(self, mod):
        super().__init__()
        mod["main"] = self.visit(mod["main"])
        self.new_mod = relay.transform.InferType()(mod)

    def visit_call(self, call):
        new_args = [self.visit(arg) for arg in call.args]
        new_call = relay.Call(call.op, new_args, call.attrs, call.type_args, call.span)

        if self.maximum1.match(new_call):
            multiply_node = new_call.args[0]
            alpha = multiply_node.args[0].data.asnumpy().tolist()
            new_call = relay.nn.leaky_relu(multiply_node.args[1], alpha)

        if self.maximum2.match(new_call):
            multiply_node = new_call.args[0]
            tgi1 = multiply_node.args[1]
            tgi2 = new_call.args[1]
            if tgi1.index == tgi2.index:
                alpha = multiply_node.args[0].data.asnumpy().tolist()
                new_call = relay.nn.leaky_relu(tgi1, alpha)

        return new_call


def leaky_relu(mod):
    mod = LeakyRelu(mod).new_mod
    return mod
