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
# pylint: disable=unused-argument,inconsistent-return-statements,unexpected-keyword-arg,bad-continuation
"""op"""

import logging
import numpy
from tvm import relay
from ..analyze import _quantized_judge

try:
    from ..relay_ops import round_right_shift
except ImportError:
    pass
from ..threshold import Threshold
from ..method_dtype import Method, DataType
from ..realize import _realize_core, operate
from ..calibrate import _calibrate_core

LOGGER = logging.getLogger("quantize")

__all__ = ("Prelu",)

VALIDCONFIG = {
    "threshold": (
        Threshold.MinMax,
        Threshold.Percentile,
        Threshold.MovingAverageMinMax,
        Threshold.L2Norm,
        Threshold.RelativeEntropy,
    ),
    "method": (Method.Symmetry, Method.Asymmetry),
    "dtype": (DataType.Int8, DataType.Int16),
}

DEFAULTCONFIG = {
    "threshold": Threshold.L2Norm,
    "method": Method.Symmetry,
    "dtype": DataType.Int8,
}


def calrightshift(v_p, mul_coef_max=127, shift_coef_max=19):
    bit = 0
    v = abs(v_p)
    while v < mul_coef_max and bit <= shift_coef_max:
        v = (v * 2).astype("float32")
        bit = bit + 1
    return numpy.floor(v_p * 2 ** (bit - 1) + 0.5), bit - 1


class Prelu:
    """Prelu"""

    name = "nn.prelu"
    controlable = False

    def __init__(self, node, vertex_config, config):
        arg = node.args[0]
        self.quantized = True
        if not vertex_config[arg].quantized:
            self.quantized = False

        if "quantized" in config:
            self.quantized = config["quantized"]

        ci0 = config["input0"]

        input_axis = -1
        if vertex_config[arg].quantized:
            input_axis = vertex_config[arg].output_config["axis"]

        input0_config = _quantized_judge(
            vertex_config, node.args[0], input_axis, self.quantized, ci0
        )

        input1_config = {
            "dtype": "float16",
            "axis": node.attrs.axis,
            "method": None,
            "threshold": None,
            "operate": "none",
        }
        self.input_config = {
            node.args[0]: input0_config,
            node.args[1]: input1_config,
        }

        # get output0_config
        output0_config = {
            "dtype": input0_config["dtype"],
            "axis": input_axis,
            "quantized_axis": "none",
            "ref_count": 0,
        }

        self.output_config = output0_config
        LOGGER.debug("[anaylze] prelu finish")

    @classmethod
    def get_config(cls, config, call):
        return {"valid_config": VALIDCONFIG, "default_config": DEFAULTCONFIG}

    def quantize_params(self, node, vertex_config):
        """quantize_params"""
        LOGGER.debug("[calibrate]prelu start and quantized is %d", self.quantized)
        arg = node.args[0]
        input_config = self.input_config[arg]

        y = _calibrate_core(arg, input_config, vertex_config)

        input_config.update(y)

        self.output_config.update(y)

    def realize(self, old_node, new_node, vertex_config, n2o):
        """realize"""
        LOGGER.debug("[realize]prelu start...")
        old_arg = old_node.args[0]
        new_arg = new_node.args[0]

        new_arg = _realize_core(self, old_arg, new_arg, vertex_config, n2o)

        if self.quantized:

            tmp = relay.frontend.common.infer_type(new_arg)
            param = numpy.array([])
            shift = numpy.array([])
            for num in new_node.args[1].data.asnumpy():
                param_num, shift_num = calrightshift(num)
                param = numpy.append(param, param_num)
                shift = numpy.append(shift, shift_num)

            axis_param = old_node.attrs.axis
            tmp_shape = tmp.checked_type.shape
            add_axis_len = len(tmp_shape) - axis_param - 1
            while add_axis_len > 0:
                add_axis_len = add_axis_len - 1
                param = numpy.expand_dims(param, axis=-1)
                shift = numpy.expand_dims(shift, axis=-1)

            # todo condisder int16!
            if "ir_pass" in relay.__dict__:
                new_node_plus = relay.maximum(new_arg, relay.const(0, "int8"))
                new_node_minus = relay.minimum(new_arg, relay.const(0, "int8"))
                new_node_minus = relay.multiply(
                    new_node_minus, relay.const(param.astype("int32")), out_dtype="int32"
                )
                new_node_minus = relay.round_right_shift(
                    new_node_minus, relay.const(shift.astype("int32")), out_dtype="int32"
                )
                new_node = relay.add(new_node_plus, new_node_minus, out_dtype="int32")
                new_node = relay.clip(new_node, -128, 127, out_dtype="int8")
            else:
                new_node_plus = relay.maximum(new_arg, relay.const(0, "int8"))
                new_node_minus = relay.minimum(new_arg, relay.const(0, "int8"))
                new_node_minus = relay.cast(new_node_minus, "int32")
                new_node_minus = relay.multiply(new_node_minus, relay.const(param.astype("int32")))
                new_node_minus = round_right_shift(
                    new_node_minus, relay.const(shift.astype("int32"))
                )
                new_node_minus = relay.cast(new_node_minus, "int8")
                new_node = relay.add(new_node_plus, new_node_minus)
        else:
            tmp = relay.frontend.common.infer_type(new_arg)
            # todo support int16
            if tmp.checked_type.dtype.startswith("int"):
                new_arg = operate(
                    "dequantize", new_arg, self.input_config[old_node.args[0]], {}, True
                )
            new_node = relay.nn.prelu(new_arg, new_node.args[1], **dict(new_node.attrs))
        return new_node
