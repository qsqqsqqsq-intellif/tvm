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
from ..analyze import _quantized_judge
from ..calibrate import _calibrate_core
from ..realize import _realize_core

LOGGER = logging.getLogger("quantize")

__all__ = ("UnmaxpoolUpsamle",)

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


class UnmaxpoolUpsamle:
    """Take"""

    name = "nn.unmaxpool_upsample"
    controlable = False

    def __init__(self, node, vertex_config, config):

        arg = node.args[0]
        self.quantized = True
        if not vertex_config[arg].quantized:
            self.quantized = False
        if "quantized" in config:
            self.quantized = config["quantized"]

        ci0 = config["input0"]

        LOGGER.debug("[analyze] nn.unmaxpool_upsample start...")
        # get input0_config
        input_axis = -1

        input0_config = _quantized_judge(
            vertex_config, node.args[0], input_axis, self.quantized, ci0
        )

        input1_config = {
            "dtype": DataType.Int32,
            "axis": -1,
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

        LOGGER.debug("[analyze] nn.unmaxpool_upsample finish")

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
        LOGGER.debug("[realize]slice like start...")
        old_arg = old_node.args[0]
        new_arg = new_node.args[0]

        new_arg = _realize_core(self, old_arg, new_arg, vertex_config, n2o)

        attrs = {}
        attrs["scale"] = new_node.attrs.scale
        attrs["pad_out_h"] = new_node.attrs.pad_out_h
        attrs["pad_out_w"] = new_node.attrs.pad_out_w
        attrs["layout"] = new_node.attrs.layout
        attrs["scale_h"] = new_node.attrs.scale_h
        attrs["scale_w"] = new_node.attrs.scale_w
        attrs["upsample_h"] = new_node.attrs.upsample_h
        attrs["upsample_w"] = new_node.attrs.upsample_w

        new_node = relay.nn.unmaxpool_upsample(new_arg, new_node.args[1], **attrs)
        return new_node