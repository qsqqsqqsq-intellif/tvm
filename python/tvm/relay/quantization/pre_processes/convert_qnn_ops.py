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
"""convert qnn requantize"""

import math
import numpy
from tvm import relay
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.dataflow_pattern import is_op, wildcard, is_constant
from tvm._ffi import runtime_ctypes
from ..relay_ops import round_right_shift


def _get_dtype_info(dtype, qmin=None, qmax=None):
    """get_dtype_info"""
    dtype = runtime_ctypes.DataType(dtype)
    dtype_str = dtype.CODE2STR[dtype.type_code]
    dtype_bit = dtype.bits
    if qmin is None and qmax is None:
        if dtype_str == "int":
            int_range = 2 ** (dtype_bit - 1)
            qmin = -int_range
            qmax = int_range - 1
        elif dtype_str == "uint":
            int_range = 2 ** dtype_bit
            qmin = 0
            qmax = int_range - 1
        else:
            raise NotImplementedError
    assert qmin is not None and isinstance(qmin, int)
    assert qmax is not None and isinstance(qmax, int)
    return {"qmin": qmin, "qmax": qmax}


def requantize(
    data, input_scale, input_zero_point, output_scale, output_zero_point, dtype, u15=False
):
    """requantize"""
    assert input_scale.size == 1
    assert input_zero_point.size == 1
    assert output_scale.size == 1
    assert output_zero_point.size == 1

    q_min_max = _get_dtype_info(dtype)

    data = relay.cast(data, "int64")

    if input_zero_point > 0:
        input_zero_point = relay.const(input_zero_point, dtype="int64")
        data = relay.subtract(data, input_zero_point)

    mul_coef_max = 255
    shift_coef_max = 32
    if u15:
        mul_coef_max = 32767
        shift_coef_max = 39

    scale = input_scale / output_scale

    bit = 0
    v = scale

    while v < mul_coef_max and bit <= shift_coef_max:
        v = v * 2
        bit = bit + 1

    bit = bit - 1
    if bit == -1:
        new_a = mul_coef_max
        bit = 0
    else:
        new_a = math.floor(scale * (2 ** bit) + 0.5)

    new_scale = relay.const(new_a, "int64")
    shift = numpy.array(bit, "int64")
    shift = relay.const(shift, "int64")

    data = relay.cast(data, "int64")
    data = relay.multiply(data, new_scale)
    data = round_right_shift(data, shift)

    if output_zero_point > 0:
        output_zero_point = relay.const(output_zero_point, dtype="int64")
        data = relay.add(data, output_zero_point)
    data = relay.clip(data, q_min_max["qmin"], q_min_max["qmax"])
    data = relay.cast(data, dtype)

    return data


@relay.transform.function_pass(opt_level=3)
class ConvertQnnOps(ExprMutator):
    """convert qnn ops"""

    requantize = is_op("qnn.requantize")(
        wildcard(), is_constant(), is_constant(), is_constant(), is_constant()
    )
    add = is_op("qnn.add")(
        wildcard(),
        wildcard(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
    )

    def visit_call(self, call):
        visited = super().visit_call(call)
        if self.requantize.match(visited):
            data = visited.args[0]
            input_scale = visited.args[1].data.asnumpy()
            input_zero_point = visited.args[2].data.asnumpy()
            output_scale = visited.args[3].data.asnumpy()
            output_zero_point = visited.args[4].data.asnumpy()
            q_dtype = visited.attrs.out_dtype

            u15 = False
            if isinstance(visited.args[0], relay.Call) and (
                visited.args[0].op.name
                in ["nn.bias_add", "nn.conv2d", "nn.dense", "nn.batch_matmul"]
            ):
                u15 = True

            data = requantize(
                data, input_scale, input_zero_point, output_scale, output_zero_point, q_dtype, u15
            )
            return data

        if self.add.match(visited):
            lhs_data = visited.args[0]
            rhs_data = visited.args[1]
            lhs_input_scale = visited.args[2].data.asnumpy()
            lhs_input_zero_point = visited.args[3].data.asnumpy()
            rhs_input_scale = visited.args[4].data.asnumpy()
            rhs_input_zero_point = visited.args[5].data.asnumpy()
            output_scale = visited.args[6].data.asnumpy()
            output_zero_point = visited.args[7].data.asnumpy()

            u15 = True

            lhs_data = requantize(
                lhs_data,
                lhs_input_scale,
                lhs_input_zero_point,
                output_scale,
                output_zero_point,
                "int64",
                u15,
            )
            rhs_data = requantize(
                rhs_data,
                rhs_input_scale,
                rhs_input_zero_point,
                output_scale,
                output_zero_point,
                "int64",
                u15,
            )

            data = relay.add(lhs_data, rhs_data)

            if output_zero_point > 0:
                output_zero_point = relay.const(output_zero_point, dtype="int64")
                data = relay.subtract(data, output_zero_point)

            data = relay.cast(data, call.args[0].checked_type.dtype)
            return data

        return visited

    def transform_function(self, func, mod, ctx):
        return self.visit(func)
