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
"""Move Reshape"""

from tvm import relay
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.dataflow_pattern import is_op, wildcard, is_constant


@relay.transform.function_pass(opt_level=3)
class MoveReshape(ExprMutator):
    """MoveReshape"""

    dense = is_op("nn.dense")(wildcard(), is_constant())
    reshape = is_op("reshape")(dense)
    bias = is_op("nn.bias_add")(reshape, is_constant())

    def visit_call(self, call):
        visited = super().visit_call(call)

        if self.bias.match(visited):
            bias_op = visited
            reshape_op = bias_op.args[0]
            dense_op = reshape_op.args[0]

            new_bias = relay.nn.bias_add(dense_op, bias_op.args[1], bias_op.attrs.axis)
            new_reshape = relay.reshape(new_bias, reshape_op.attrs.newshape)
            return new_reshape

        return visited

    def transform_function(self, func, mod, ctx):
        return self.visit(func)
