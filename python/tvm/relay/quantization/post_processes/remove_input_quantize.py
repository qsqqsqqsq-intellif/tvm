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
# pylint: disable=all
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
        self.original_params = []
        self.net_in_dtype = net_in_dtype
        self.expr_var_map = {}
        self.new_vars = {}

        if isinstance(mod, relay.Function):
            mod = self.visit(mod)
            self.new_mod = relay.ir_pass.infer_type(mod)
        else:
            mod["main"] = self.visit(mod["main"])
            self.new_mod = relay.transform.InferType()(mod)

    def visit_call(self, call):
        visited = super().visit_call(call)

        if is_call(visited, "cast") and visited.attrs.dtype in ["uint8", "int16"]:
            the_expr = visited.args[0]
            if is_call(the_expr, "clip"):
                the_expr = the_expr.args[0]
                if is_call(the_expr, "round"):
                    the_expr = the_expr.args[0]
                    if is_call(the_expr, "multiply"):
                        the_expr = the_expr.args[0]
                        if is_call(the_expr, "cast") and isinstance(the_expr.args[0], relay.Var):
                            expr = the_expr.args[0]
                            var_name = expr.name_hint

                            if expr in self.original_params:
                                if self.expr_var_map.get(expr) is None:

                                    if (
                                        isinstance(self.net_in_dtype, dict)
                                        and var_name in self.net_in_dtype
                                        and self.net_in_dtype[var_name] == visited.attrs.dtype
                                    ):
                                        dtype = self.net_in_dtype[var_name]
                                    elif (
                                        isinstance(self.net_in_dtype, str)
                                        and self.net_in_dtype == visited.attrs.dtype
                                    ):
                                        dtype = self.net_in_dtype

                                    shape = expr.checked_type.shape

                                    new_var = relay.var(expr.name_hint, shape=shape, dtype=dtype)
                                    self.new_vars[var_name] = new_var
                                    self.expr_var_map[expr] = new_var
                                    return new_var

                                return self.expr_var_map[expr]

        return visited

    def visit_function(self, fn):
        self.original_params = fn.params
        visited = super().visit_function(fn)

        params = list(visited.params)
        new_params = list(self.new_vars.values())
        for param in params:
            if param.name_hint not in list(self.new_vars.keys()):
                new_params.append(param)

        visited = relay.Function(
            new_params, visited.body, visited.ret_type, visited.type_params, visited.attrs
        )

        return visited


def remove_input_quantize(mod, net_in_dtype):
    mod = RemoveInputQuantize(mod, net_in_dtype).new_mod
    return mod
