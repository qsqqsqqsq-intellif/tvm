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
"""op"""

import logging
from tvm import relay
from ..threshold import Threshold
from ..method_dtype import Method, DataType
from ..analyze import oneargdeal
from ..calibrate import _calibrate_core
from ..realize import operate, _realize_core

LOGGER = logging.getLogger("quantize")

__all__ = ("Clip",)

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


class Clip:
    """clip"""

    name = "clip"
    controlable = False

    def __init__(self, node, vertex_config, config):

        LOGGER.debug("[anaylze] clip start")
        self.quantized = True
        if not vertex_config[node.args[0]].quantized:
            self.quantized = False

        if "quantized" in config:
            self.quantized = config["quantized"]

        ci0 = config["input0"]

        oneargdeal(self, node, vertex_config, ci0)

        LOGGER.debug("[anaylze] clip finish")

    @classmethod
    def get_config(cls, config, call):
        return {"valid_config": VALIDCONFIG, "default_config": DEFAULTCONFIG}

    def quantize_params(self, node, vertex_config):
        """quantize_params"""
        LOGGER.info("[calibrate] clip start")
        input_config = self.input_config[node.args[0]]

        y = _calibrate_core(node.args[0], input_config, vertex_config, self.quantized)

        self.input_config[node.args[0]].update(y)
        self.output_config.update(y)

        LOGGER.debug("[calibrate]clip calibrate over")

    def realize(self, old_node, new_node, vertex_config, n2o):
        """realize"""
        LOGGER.debug("[realize]clip start")
        old_arg = old_node.args[0]
        new_arg = new_node.args[0]

        new_arg = _realize_core(self, old_arg, new_arg, vertex_config, n2o)

        if not self.quantized:
            tmp = relay.frontend.common.infer_type(new_arg)

            if tmp.checked_type.dtype.startswith("int"):
                new_arg = operate(
                    "dequantize", new_arg, self.input_config[old_node.args[0]], {}, True
                )
            new_node = relay.clip(new_arg, **dict(old_node.attrs))
        else:

            # case: batch_flatten + clip(0, 6)
            adjust_input_config = {
                "zero_point": self.output_config["quantized_zero_point"],
                "axis": self.output_config["quantized_axis"],
                "scale": self.output_config["quantized_scale"],
                "dtype": self.input_config[old_node.args[0]]["dtype"],
                "operate": "requantize",
            }
            new_arg = operate(
                "requantize",
                new_arg,
                self.input_config[old_arg],
                adjust_input_config,
                True,
                multiplier=1,
            )

            self.output_config["scale"] = self.output_config["quantized_scale"]
            self.output_config["zero_point"] = self.output_config["quantized_zero_point"]
            self.output_config["axis"] = self.output_config["quantized_axis"]

            new_node = relay.nn.relu(new_arg)
        return new_node
