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
"""convert_input"""

import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprMutator


def is_call(the_expr, names):
    op_names = names if isinstance(names, list) else [names]
    return isinstance(the_expr, tvm.relay.expr.Call) and the_expr.op.name in op_names


class RemoveInputQuantize(ExprMutator):
    """RemoveInputQuantize"""

    def __init__(self, mod, net_in_dtype):
        super().__init__()
        self.new_params = []
        self.convert = False
        self.net_in_dtype = net_in_dtype
        mod["main"] = self.visit(mod["main"])
        self.new_mod = relay.transform.InferType()(mod)

    def visit_var(self, var):
        if var in self.ori_params and self.net_in_dtype in ["uint8", "int16"]:
            new_var = relay.var(
                var.name_hint, shape=var.type_annotation.shape, dtype=self.net_in_dtype
            )
            self.new_params.append(new_var)
            return new_var
        return var

    def visit_call(self, call):
        visited = super().visit_call(call)
        if self.convert or self.net_in_dtype not in ["uint8", "int16"]:
            return visited
        if is_call(visited.args[0], "qnn.quantize"):
            self.convert = True
            return relay.Call(
                visited.op,
                [visited.args[0].args[0], visited.args[1]],
                visited.attrs,
                visited.type_args,
            )

        if is_call(visited.args[0], "cast") and visited.args[0].attrs.dtype == self.net_in_dtype:
            the_expr = visited.args[0].args[0]
            if is_call(the_expr, "clip"):
                the_expr = the_expr.args[0]
                if is_call(the_expr, "round"):
                    the_expr = the_expr.args[0]
                    if is_call(the_expr, "divide"):
                        the_expr = the_expr.args[0]
                        if is_call(the_expr, "cast") and isinstance(the_expr.args[0], relay.Var):
                            self.convert = True
                            return relay.Call(
                                visited.op,
                                [the_expr.args[0], *visited.args[1:]],
                                visited.attrs,
                                visited.type_args,
                            )

        return visited

    def visit_function(self, fn):
        self.ori_params = fn.params
        visited = super().visit_function(fn)
        return visited


def remove_input_quantize(mod, net_in_dtype):
    mod = RemoveInputQuantize(mod, net_in_dtype).new_mod
    return mod
