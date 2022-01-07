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
"""op"""

import logging
import numpy
from tvm import relay
from ..threshold import Threshold
from ..method_dtype import Method, DataType, _get_dtype_info
from ..analyze import _conv_counter, oneargdeal
from ..calibrate import _calibrate_core
from ..realize import operate, pair_node

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

        cnt = _conv_counter()

        arg = node.args[0]
        self.quantized = True
        if not vertex_config[arg].quantized or node.attrs.a_min < 0 or cnt - 1 in []:
            self.quantized = False

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

        # convert to relu
        a_max = node.attrs.a_max
        q_min_max = _get_dtype_info(self.input_config[node.args[0]]["dtype"])
        valid_bits_range = q_min_max["qmax"]
        # in fact, the scale can be reset by output_cofig['quantized_axis']
        if y["scale"].size > 1:
            y["scale"][numpy.where(y["scale"] > a_max / valid_bits_range)] = (
                a_max / valid_bits_range
            )
        elif y["scale"] > a_max / valid_bits_range:
            y["scale"] = a_max / valid_bits_range
        tmp = {"quantized_scale": y["scale"], "quantized_zero_point": y["zero_point"]}
        LOGGER.debug("[calibrate]--clip arg quantized_scale is:")
        LOGGER.debug(y["scale"])
        vertex_config[node.args[0]].output_config.update(tmp)

        self.input_config[node.args[0]].update(y)
        self.output_config.update(y)
        LOGGER.debug("[calibrate]clip calibrate over")

    def realize(self, old_node, new_node, vertex_config, n2o):
        """realize"""
        LOGGER.debug("[realize]clip start")
        old_arg = old_node.args[0]
        new_arg = new_node.args[0]

        output_config = vertex_config[old_arg].output_config
        if "quantized_scale" in vertex_config[old_arg].output_config:
            LOGGER.debug("[realize]clip's input output_config['quantized_scale'] is:")
            LOGGER.debug(vertex_config[old_arg].output_config["quantized_scale"])
        input_config = self.input_config[old_arg]

        new_arg = operate(input_config["operate"], new_arg, output_config, input_config, True)
        pair_node(old_arg, new_arg, output_config, input_config, n2o, self.quantized)

        if not self.quantized:
            tmp = relay.frontend.common.infer_type(new_arg)

            if tmp.checked_type.dtype.startswith("int"):
                new_arg = operate(
                    "dequantize", new_arg, self.input_config[old_node.args[0]], {}, True
                )
            new_node = relay.clip(new_arg, **dict(old_node.attrs))
        else:
            new_node = relay.nn.relu(new_arg)
        return new_node
