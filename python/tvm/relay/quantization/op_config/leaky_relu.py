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
# pylint: disable=unused-argument,inconsistent-return-statements,unexpected-keyword-arg,too-many-function-args,bad-continuation
"""op"""

import logging
from tvm import relay

try:
    from ..relay_ops import round_right_shift
except ImportError:
    pass
from ..threshold import Threshold
from ..method_dtype import Method, DataType
from ..analyze import oneargdeal
from ..realize import _realize_core, operate
from ..calibrate import _calibrate_core

LOGGER = logging.getLogger("quantize")

__all__ = ("LeakyRelu",)

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


def calrightshift(v_p, mul_coef_max=255, shift_coef_max=15):
    bit = -1
    v = v_p
    while v < mul_coef_max and bit <= shift_coef_max:
        v = v * 2
        bit = bit + 1
    return int(v_p * 2 ** (bit - 1)), bit - 1


class LeakyRelu:
    """LeakyRelu"""

    name = "nn.leaky_relu"
    controlable = False

    def __init__(self, node, vertex_config, config):

        arg = node.args[0]
        self.quantized = True
        if not vertex_config[arg].quantized:
            self.quantized = False

        if "quantized" in config:
            self.quantized = config["quantized"]

        ci0 = config["input0"]

        oneargdeal(self, node, vertex_config, ci0)

        LOGGER.debug("[anaylze] leaky_relu finish")

    @classmethod
    def get_config(cls, config, call):
        return {"valid_config": VALIDCONFIG, "default_config": DEFAULTCONFIG}

    def quantize_params(self, node, vertex_config):
        """quantize_params"""
        LOGGER.debug("[calibrate]leaky_relu start and quantized is %d", self.quantized)
        arg = node.args[0]
        input_config = self.input_config[arg]

        y = _calibrate_core(arg, input_config, vertex_config)

        input_config.update(y)

        self.output_config.update(y)

    def realize(self, old_node, new_node, vertex_config, n2o):
        """realize"""
        LOGGER.debug("[realize]leaky_relu start...")
        old_arg = old_node.args[0]
        new_arg = new_node.args[0]

        new_arg = _realize_core(self, old_arg, new_arg, vertex_config, n2o)

        if self.quantized:
            alpha = old_node.attrs.alpha
            param, shift = calrightshift(alpha)
            # todo condisder int16!
            if "ir_pass" in relay.__dict__:
                new_node_plus = relay.maximum(new_arg, relay.const(0, "int8"))
                new_node_minus = relay.minimum(new_arg, relay.const(0, "int8"))
                new_node_minus = relay.multiply(
                    new_node_minus, relay.const(param, "int32"), out_dtype="int32"
                )
                new_node_minus = relay.round_right_shift(
                    new_node_minus, relay.const(shift, "int32"), out_dtype="int32"
                )
                new_node = relay.add(new_node_plus, new_node_minus)
                new_node = relay.clip(new_node, -128, 127, out_dtype="int8")
            else:
                new_node_plus = relay.maximum(new_arg, relay.const(0, "int8"))
                new_node_minus = relay.minimum(new_arg, relay.const(0, "int8"))
                new_node_minus = relay.cast(new_node_minus, "int32")
                new_node_minus = relay.multiply(new_node_minus, relay.const(param, "int32"))
                new_node_minus = round_right_shift(new_node_minus, relay.const(shift, "int32"))
                new_node_minus = relay.cast(new_node_minus, "int8")
                new_node = relay.add(new_node_plus, new_node_minus)
        else:
            tmp = relay.frontend.common.infer_type(new_arg)
            # todo support int16
            if tmp.checked_type.dtype.startswith("int"):
                new_arg = operate(
                    "dequantize", new_arg, self.input_config[old_node.args[0]], {}, True
                )
            new_node = relay.nn.leaky_relu(new_arg, **dict(new_node.attrs))
        return new_node