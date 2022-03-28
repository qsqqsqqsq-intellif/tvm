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
# pylint: disable=unused-argument,inconsistent-return-statements,bad-continuation
"""fuse ops"""
from tvm import relay
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.dataflow_pattern import is_op, is_tuple, wildcard, is_constant


@relay.transform.function_pass(opt_level=3)
class FuseOps(ExprMutator):
    """fuse ops"""

    pattern1 = is_op("multiply")(is_op("multiply")(wildcard(), is_constant()), is_constant())
    pattern2 = is_op("concatenate")(is_tuple(None))

    def visit_call(self, call):
        visited = super().visit_call(call)

        if self.pattern1.match(visited):
            arg0 = visited.args[0].args[0]
            const1 = visited.args[0].args[1].data.asnumpy()
            const2 = visited.args[1].data.asnumpy()
            new_const = const1 * const2
            new_const = relay.const(new_const)
            return relay.multiply(arg0, new_const)

        if self.pattern2.match(visited):
            axis = visited.attrs.axis
            tup = visited.args[0]

            replaced = False
            new_fields = []
            for field in tup.fields:
                if isinstance(field, relay.Call) and field.op.name == "concatenate":
                    pre_axis = field.attrs.axis
                    if axis == pre_axis:
                        replaced = True
                        for f in field.args[0].fields:
                            new_fields.append(f)

                        continue

                new_fields.append(field)

            if replaced:
                return relay.concatenate(new_fields, axis)

        return visited

    def transform_function(self, func, mod, ctx):
        return self.visit(func)
