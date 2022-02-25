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


LOGGER = logging.getLogger("quantize")

__all__ = ("Cast",)

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


class Cast:
    """cast"""

    name = "cast"
    controlable = False

    def __init__(self, node, vertex_config, config):

        arg = node.args[0]
        self.quantized = True
        if not vertex_config[arg].quantized:
            self.quantized = False

        ci0 = config["input0"]

        oneargdeal(self, node, vertex_config, ci0)

        # dtype = node.attrs.dtype

        # if dtype.startswith("float"):
        #     self.output_config["dtype"] = "float16"
        # elif dtype.startswith("int") and int(dtype[3:]) > 32:
        #     self.output_config["dtype"] = "int32"
        # else:
        #     self.output_config["dtype"] = dtype

        LOGGER.debug("[anaylze] cast finish")

    @classmethod
    def get_config(cls, config, call):
        return {"valid_config": VALIDCONFIG, "default_config": DEFAULTCONFIG}

    def quantize_params(self, node, vertex_config):
        """quantize_params"""
        LOGGER.info("[calibrate] cast start")
        # input_config = self.input_config[node.args[0]]

        # y = _calibrate_core(node.args[0], input_config, vertex_config, self.quantized)

        LOGGER.debug("[calibrate]cast calibrate over")

    def realize(self, old_node, new_node, vertex_config, n2o):
        """realize"""
        LOGGER.debug("[realize]cast start")

        new_arg = new_node.args[0]
        source_dtype = relay.frontend.common.infer_type(new_arg).checked_type.dtype
        targe_dtype = old_node.attrs.dtype

        if source_dtype == targe_dtype:
            return new_arg

        if targe_dtype.startswith("float") and source_dtype in ["bool"]:
            targe_dtype = "float16"
        elif targe_dtype.startswith("int") and int(targe_dtype[3:]) > 32:
            targe_dtype = "int32"

        return relay.cast(new_arg, targe_dtype)
