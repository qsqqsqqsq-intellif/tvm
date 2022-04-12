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
# pylint: disable=unused-argument,inconsistent-return-statements,too-many-function-args
"""op"""

import logging
from tvm import relay
from ..analyze import oneargdeal
from ..calibrate import _calibrate_core
from ..realize import _realize_core

LOGGER = logging.getLogger("quantize")

__all__ = ("Split",)

VALIDCONFIG = {}
DEFAULTCONFIG = {}


class Split:
    """Split"""

    name = "split"
    _controlable = False

    def __init__(self, node, vertex_config, configs):

        arg = node.args[0]
        self.quantized = True
        if not vertex_config[arg].quantized:
            self.quantized = False

        ci0 = configs["input0"]

        oneargdeal(self, node, vertex_config, ci0)
        LOGGER.debug("[analyze] split finish")

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
        LOGGER.debug("[realize]split start...")
        old_arg = old_node.args[0]
        new_arg = new_node.args[0]

        new_arg = _realize_core(self, old_arg, new_arg, vertex_config, n2o)

        if "ir_pass" not in relay.__dict__:
            new_node = relay.Call(
                old_node.op, [new_arg], new_node.attrs, new_node.type_args, new_node.span
            )
        else:
            new_node = relay.Call(
                old_node.op, [new_arg], new_node.attrs, new_node.type_args
            )

        return new_node
