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

from tvm import relay
from ..realize import operate, pair_node

__all__ = ("HardSigmoid",)

VALIDCONFIG = {}
DEFAULTCONFIG = {}


class HardSigmoid:
    """HardSigmoid"""

    name = "hard_sigmoid"
    controlable = False

    def __init__(self, node, vertex_config, config):
        self.quantized = False

        arg = node.args[0]
        input0_config = {}
        input0_config.update({"in_dtype": vertex_config[arg].output_config["dtype"]})
        input0_config.update({"in_axis": vertex_config[arg].output_config["axis"]})
        if not vertex_config[arg].quantized and not self.quantized:
            tmp2 = {"operate": "none"}
        elif vertex_config[arg].quantized and not self.quantized:
            tmp2 = {"operate": "dequantize"}
        input0_config.update(tmp2)
        self.input_config = {arg: input0_config}

        self.output_config = {
            "dtype": node.checked_type.dtype,
            "axis": -1,
        }

    @classmethod
    def get_config(cls, config, call):
        return {"valid_config": VALIDCONFIG, "default_config": DEFAULTCONFIG}

    def realize(self, old_node, new_node, vertex_config, n2o):
        """realize"""
        old_arg = old_node.args[0]
        new_arg = new_node.args[0]

        output_config = vertex_config[old_arg].output_config
        input_config = self.input_config[old_arg]
        new_arg = operate(input_config["operate"], new_arg, output_config, input_config, True)
        pair_node(old_arg, new_arg, output_config, input_config, n2o, self.quantized)

        new_node = relay.nn.softmax(new_arg, **dict(new_node.attrs))
        return new_node
