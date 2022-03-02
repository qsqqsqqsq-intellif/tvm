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
import numpy
from tvm import relay
from ..analyze import _conv_counter, oneargdeal
from ..calibrate import _calibrate_core
from ..realize import _realize_core

LOGGER = logging.getLogger("quantize")

__all__ = ("Pad",)

VALIDCONFIG = {}
DEFAULTCONFIG = {}


class Pad:
    """Pad"""

    name = "nn.pad"
    controlable = False

    def __init__(self, node, vertex_config, config):
        cnt = _conv_counter()

        arg = node.args[0]
        self.quantized = True
        if not vertex_config[arg].quantized or (
            "skip_conv_layers" in config and cnt in config["skip_conv_layers"]
        ):
            self.quantized = False

        oneargdeal(self, node, vertex_config, config["input0"])
        LOGGER.debug("[analyze] nn.pad finish")

    @classmethod
    def get_config(cls, config, call):
        return {"valid_config": VALIDCONFIG, "default_config": DEFAULTCONFIG}

    def quantize_params(self, node, vertex_config):
        """quantize_params"""
        LOGGER.debug("[calibrate] nn.pad start...")
        arg = node.args[0]
        input_config = self.input_config[arg]

        y = _calibrate_core(arg, input_config, vertex_config, self.quantized)

        input_config.update(y)

        self.output_config.update(y)

    def realize(self, old_node, new_node, vertex_config, n2o):
        """realize"""
        LOGGER.debug("[realize] nn.pad start...")
        old_arg = old_node.args[0]
        new_arg = new_node.args[0]

        new_arg = _realize_core(self, old_arg, new_arg, vertex_config, n2o)

        new_attrs = {}
        new_attrs["pad_width"] = new_node.attrs.pad_width
        new_attrs["pad_mode"] = new_node.attrs.pad_mode
        new_attrs["pad_value"] = new_node.attrs.pad_value
        if (
            self.quantized
            and old_node.attrs.pad_mode == "constant"
            and old_node.attrs.pad_value != 0
        ):
            pad_value = old_node.attrs.pad_value
            assert (
                self.input_config[old_arg]["scale"].size == 1
            ), "when padvalue !=0 and quantized, only support per_tensor"
            pad_value = numpy.floor(pad_value / self.input_config[old_arg]["scale"])
            if pad_value > 127:
                pad_value = 127
            elif pad_value < -128:
                pad_value = -128
            new_attrs["pad_value"] = pad_value

        new_node = relay.nn.pad(new_arg, **new_attrs)
        return new_node
