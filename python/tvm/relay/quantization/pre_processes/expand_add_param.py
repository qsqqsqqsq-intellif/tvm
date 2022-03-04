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
# pylint: disable=unused-argument,inconsistent-return-statements,bad-continuation,arguments-differ
"""expand add pram and then convert to bias_add"""
import numpy as np
import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprMutator


@relay.transform.function_pass(opt_level=3)
class ExpandAddParam(ExprMutator):
    """ExpandAddParam"""

    def visit_call(self, call):
        def find_layout(node):
            """find layout"""
            layout = None
            transposed = False
            while not layout and isinstance(node, relay.Call):
                if isinstance(node.op, relay.Function):
                    layout = find_layout(node.op.body)

                else:
                    if node.op.name in ["transpose", "reshape", "nn.batch_flatten"]:
                        transposed = True
                        break

                if transposed:
                    break

                attrs = dict(node.attrs) if node.attrs is not None else {}
                if "layout" in attrs:
                    layout = attrs["layout"]

                if "data_layout" in attrs:
                    layout = attrs["data_layout"]

                node = node.args[0]

            return layout

        visited = super().visit_call(call)
        if visited.op == tvm.relay.op.get("add") and isinstance(
            visited.args[1], tvm.relay.Constant
        ):
            shape0 = call.args[0].checked_type.shape
            shape1 = call.args[1].checked_type.shape
            if shape1:
                return visited

            if len(shape0) < 4:
                return visited

            layout = find_layout(visited.args[0])

            if layout:
                idx = layout.find("C")
                # dest_shape = [1 for _ in range(len(shape0) - idx)]
                # dest_shape[0] = shape0[idx].value
                dest_shape = [shape0[idx].value]

                full_ones = np.ones(dest_shape, call.args[1].checked_type.dtype)
                new_weight = full_ones * visited.args[1].data.asnumpy()

                return relay.nn.bias_add(visited.args[0], relay.const(new_weight), axis=idx)

        return visited

    def transform_function(self, func, mod, ctx):
        return self.visit(func)
