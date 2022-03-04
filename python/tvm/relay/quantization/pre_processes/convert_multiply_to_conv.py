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
"""convert multiply to conv2d"""
import numpy as np
import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprMutator


@relay.transform.function_pass(opt_level=3)
class ConvertMultiplyToConv(ExprMutator):
    """ConvertMultiplyToConv"""

    def visit_call(self, call):
        visited = super().visit_call(call)
        if visited.op == tvm.relay.op.get("multiply") and isinstance(
            visited.args[1], tvm.relay.Constant
        ):

            shape0 = call.args[0].checked_type.shape
            if shape0:
                shape0 = [x.value for x in shape0]

            if len(shape0) != 4:
                return visited

            shape1 = call.args[1].checked_type.shape
            if shape1:
                shape1 = [x.value for x in shape1]
            weight = visited.args[1].data.asnumpy()
            layout = None

            # find layout
            if not shape1 or not layout:
                pre_expr = call.args[0]

                while isinstance(pre_expr, relay.Call) and pre_expr.op.name not in [
                    "transpose",
                    "reshape",
                    "nn.batch_flatten",
                ]:
                    print(pre_expr)
                    attrs = dict(pre_expr.attrs) if pre_expr.attrs is not None else {}
                    if "layout" in attrs:
                        layout = attrs["layout"]

                    if "data_layout" in attrs:
                        layout = attrs["data_layout"]

                    if layout is not None:
                        break

                    pre_expr = pre_expr.args[0]

                if not shape1 and layout is not None:
                    idx = layout.find("C")
                    shape1 = [1 for _ in range(len(shape0) - idx)]
                    shape1[0] = shape0[idx]

                    full_ones = np.ones(shape1, call.args[1].checked_type.dtype)
                    weight = full_ones * visited.args[1].data.asnumpy()

            if (
                len(shape0) == 5
                and isinstance(visited.args[0], relay.Call)
                and visited.args[0].op.name in ["nn.conv3d"]
            ):
                assert layout == "NCDHW"
                ichannel = shape0[1]
                ochannel = shape1[0]
                assert ichannel == ochannel
                s = 1
                for x in shape1:
                    s = s * x
                assert s == ochannel
                r_shape = [ochannel, 1, 1, 1, 1]
                weight = weight.reshape(r_shape)

                conv3d_w = visited.args[0].args[1].data.asnumpy()
                new_w = conv3d_w * weight
                conv3d_arg = relay.Constant(tvm.nd.array(new_w))
                attrs = dict(visited.args[0].attrs)

                return relay.nn.conv3d(visited.args[0].args[0], conv3d_arg, **attrs)

            if len(shape0) == 5 and (
                not isinstance(visited.args[0], relay.Call)
                or visited.args[0].op.name not in ["nn.conv3d"]
            ):
                assert layout == "NCDHW"
                if len(shape1) != 4:
                    return visited
                ichannel = shape0[1]
                ochannel = shape1[0]
                assert ichannel == ochannel
                s = 1
                for x in shape1:
                    s = s * x
                assert s == ochannel
                r_shape = [ichannel, 1, 1, 1, 1]
                x = np.ones((ochannel, 1, 1, 1), dtype=np.float32)

                zeros = np.zeros((1, 1, 1), dtype=np.float32)

                weight = weight.reshape(r_shape)
                weight = weight * x
                for i in range(0, ochannel):
                    for j in range(0, ichannel):
                        if i != j:
                            weight[i][j] = zeros
                conv3d_arg = relay.Constant(tvm.nd.array(weight))

                return relay.nn.conv3d(
                    visited.args[0], conv3d_arg, channels=ochannel, kernel_size=[1, 1, 1]
                )

            if layout == "NHWC":
                c_idx = 3
                kernel_layout = "HWOI"
                r_shape = [1, 1, shape1[0], 1]
            elif layout == "NCHW":
                c_idx = 1
                kernel_layout = "OIHW"
                r_shape = [shape1[0], 1, 1, 1]
            else:
                return visited

            channels = shape0[c_idx]
            assert channels == shape1[0]
            conv2d_arg = relay.Constant(tvm.nd.array(weight.reshape(r_shape)))
            return relay.nn.conv2d(
                visited.args[0],
                conv2d_arg,
                data_layout=layout,
                kernel_layout=kernel_layout,
                groups=channels,
                kernel_size=(1, 1),
            )

        return visited

    def transform_function(self, func, mod, ctx):
        return self.visit(func)
