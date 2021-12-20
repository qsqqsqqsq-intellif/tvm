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
from tvm import relay
from ..threshold import Threshold
from ..method_dtype import Method, DataType
from ..analyze import _conv_counter, oneargdeal
from ..calibrate import _calibrate_core
from ..realize import _realize_core

LOGGER = logging.getLogger("quantize")

__all__ = ("Reshape",)

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
    "threshold": Threshold.RelativeEntropy,
    "method": Method.Symmetry,
    "dtype": DataType.Int8,
}


class Reshape:
    """reshape"""

    name = "reshape"
    controlable = False

    def __init__(self, node, vertex_config, config):
        cnt = _conv_counter()

        arg = node.args[0]
        self.quantized = True
        if not vertex_config[arg].quantized or cnt - 1 in []:
            self.quantized = False

        ci0 = config["input0"]

        oneargdeal(self, node, vertex_config, ci0)
        LOGGER.debug("[analyze] reshape finish")

    @classmethod
    def get_config(cls, config, call):
        return {"valid_config": VALIDCONFIG, "default_config": DEFAULTCONFIG}

    def quantize_params(self, node, vertex_config):
        """quantize_params"""
        arg = node.args[0]
        input_config = self.input_config[arg]

        y = _calibrate_core(arg, input_config, vertex_config, self.quantized)

        input_config.update(y)

        self.output_config.update(y)

    def realize(self, old_node, new_node, vertex_config, n2o):
        """realize"""
        LOGGER.debug("[realize]reshape start...")
        old_arg = old_node.args[0]
        new_arg = new_node.args[0]

        new_arg = _realize_core(self, old_arg, new_arg, vertex_config, n2o)

        new_node = relay.reshape(new_arg, **dict(new_node.attrs))
        return new_node
