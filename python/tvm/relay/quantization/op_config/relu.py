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
from ..analyze import oneargdeal
from ..realize import _realize_core, operate
from ..calibrate import _calibrate_core

LOGGER = logging.getLogger("quantize")

__all__ = ("Relu",)

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


class Relu:
    """relu"""

    name = "nn.relu"
    controlable = False

    def __init__(self, node, vertex_config, config):

        LOGGER.debug("[anaylze] relu start")
        self.quantized = True
        if "quantized" in config:
            self.quantized = config["quantized"]

        ci0 = config["input0"]

        oneargdeal(self, node, vertex_config, ci0)

        LOGGER.debug("[anaylze] relu finish")

    @classmethod
    def get_config(cls, config, call):
        return {"valid_config": VALIDCONFIG, "default_config": DEFAULTCONFIG}

    def quantize_params(self, node, vertex_config):
        """quantize_params"""
        arg = node.args[0]
        input_config = self.input_config[arg]

        y = _calibrate_core(arg, input_config, vertex_config)

        input_config.update(y)

        self.output_config.update(y)

    def realize(self, old_node, new_node, vertex_config, n2o):
        """realize"""
        old_arg = old_node.args[0]
        new_arg = new_node.args[0]

        new_arg = _realize_core(self, old_arg, new_arg, vertex_config, n2o)
        if not self.quantized:
            tmp = relay.frontend.common.infer_type(new_arg)
            if tmp.checked_type.dtype.startswith("int") and tmp.checked_type.dtype not in ["int32"]:
                new_arg = operate("dequantize", new_arg, self.input_config[old_arg], {}, True)
            elif tmp.checked_type.dtype != "float16":
                new_arg = relay.cast(new_arg, "float16")

        new_node = relay.nn.relu(new_arg)
        return new_node