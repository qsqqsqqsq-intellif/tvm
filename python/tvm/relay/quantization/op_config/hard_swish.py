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
from ..realize import operate, pair_node
from ..analyze import _quantized_judge
from ..calibrate import _calibrate_core

LOGGER = logging.getLogger("quantize")

__all__ = ("HardSwish",)

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


class HardSwish:
    """HardSwish"""

    name = "hard_swish"
    controlable = True

    def __init__(self, node, vertex_config, config):
        if isinstance(node.op, relay.Function):
            self.name = getattr(node.op.attrs, "Composite")
        else:
            self.name = node.op.name

        LOGGER.debug("[analyze] %s start...", self.name)
        self.quantized = False

        self.input_config = {}
        for i, one_arg in enumerate(node.args):
            tmp = "input" + str(i)
            input_axis = -1
            if vertex_config[one_arg].quantized:
                input_axis = vertex_config[one_arg].output_config["axis"]

            input_config = _quantized_judge(
                vertex_config, one_arg, input_axis, self.quantized, config[tmp]
            )
            self.input_config[one_arg] = input_config

        output0_config = {
            "dtype": DataType.Float16,
            "axis": -1,
            "quantized_axis": "none",
            "ref_count": 0,
        }
        self.output_config = output0_config
        LOGGER.debug("[anaylze] %s finish", self.name)

    @classmethod
    def get_config(cls, config, call):
        return {"valid_config": VALIDCONFIG, "default_config": DEFAULTCONFIG}

    def quantize_params(self, node, vertex_config):
        """quantize_params"""
        LOGGER.debug("[calibrate] %s start and quantized is %d", self.name, self.quantized)

        for arg in node.args:
            input_config = self.input_config[arg]
            y = _calibrate_core(arg, input_config, vertex_config, self.quantized)
            input_config.update(y)

    def realize(self, old_node, new_node, vertex_config, n2o):
        """realize"""
        LOGGER.debug("[realize]-- %s realize...", self.name)

        realized_args = []
        for old_arg, new_arg in zip(old_node.args, new_node.args):

            output_config = vertex_config[old_arg].output_config
            input_config = self.input_config[old_arg]
            if (
                output_config["ref_count"] > 1
                and old_arg in n2o
                and (self.quantized or vertex_config[old_arg].quantized)
            ):
                new_arg = n2o[old_arg]
            else:
                new_arg = operate(
                    input_config["operate"], new_arg, output_config, input_config, True
                )

            if (
                output_config["ref_count"] > 1
                and old_arg not in n2o
                and input_config["operate"] != "none"
            ):
                n2o[old_arg] = new_arg

            realized_args.append(new_arg)

        new_realized_args = []
        for old_arg, new_arg in zip(old_node.args, realized_args):
            tmp = relay.frontend.common.infer_type(new_arg)
            if isinstance(new_arg, relay.Constant) and tmp.checked_type.dtype != "float16":
                new_arg = relay.const(new_arg.data.asnumpy().astype("float16"))
            elif tmp.checked_type.dtype.startswith("int"):
                new_arg = operate("dequantize", new_arg, self.input_config[old_arg], {}, True)
            elif tmp.checked_type.dtype != "float16":
                new_arg = relay.cast(new_arg, "float16")

            pair_node(old_arg, new_arg, {}, {"operate": "none"}, n2o, self.quantized)

            new_realized_args.append(new_arg)

        add = relay.add(new_realized_args[0], relay.const(3, "float16"))
        clip = relay.clip(add, 0, 6)
        divide = relay.multiply(clip, relay.const(1 / 6, "float16"))
        new_node = relay.multiply(new_realized_args[0], divide)
        LOGGER.debug("[realize] %s finish", self.name)
        return new_node
