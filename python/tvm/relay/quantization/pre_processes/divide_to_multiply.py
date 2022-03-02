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
"""divide to multiply"""

from tvm import relay
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.dataflow_pattern import is_op, wildcard, is_constant


class DivideToMultiply(ExprMutator):
    """divide to multiply"""

    divide = is_op("divide")(wildcard(), is_constant())

    def __init__(self, mod):
        super().__init__()
        mod["main"] = self.visit(mod["main"])
        self.new_mod = relay.transform.InferType()(mod)

    def visit_call(self, call):
        new_args = [self.visit(arg) for arg in call.args]
        new_call = relay.Call(call.op, new_args, call.attrs, call.type_args, call.span)

        if self.divide.match(new_call):
            data = new_call.args[1].data.asnumpy()
            if "int" in str(data.dtype):
                return new_call

            new_call = relay.multiply(new_call.args[0], relay.const(1 / data, data.dtype))
            return new_call

        return new_call


def divide_to_multiply(mod):
    mod = DivideToMultiply(mod).new_mod
    return mod
