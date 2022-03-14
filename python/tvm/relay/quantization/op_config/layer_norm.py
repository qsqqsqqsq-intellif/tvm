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
# pylint: disable=unused-argument,inconsistent-return-statements,too-many-function-args,bad-continuation
"""op"""

import logging
import functools
import numpy
from tvm import relay
from ..threshold import Threshold
from ..method_dtype import Method, DataType
from ..realize import operate, pair_node
from ..analyze import _quantized_judge
from ..calibrate import _calibrate_core

LOGGER = logging.getLogger("quantize")

__all__ = ("LayerNorm",)

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


class LayerNorm:
    """LayerNorm"""

    name = "nn.layer_norm"
    controlable = True

    def __init__(self, node, vertex_config, config):
        if isinstance(node.op, relay.Function):
            self.name = getattr(node.op.attrs, "Composite")
        else:
            self.name = node.op.name
        self.op = node.op

        LOGGER.debug("[analyze] %s start...", self.name.upper())
        self.quantized = False

        self.input_config = {}
        for i, one_arg in enumerate(node.args):
            tmp = "input" + str(i)
            # todo optimize per_channel
            # if vertex_config[one_arg].quantized:
            #     input_axis = vertex_config[one_arg].output_config["axis"]
            input_axis = -1
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
        LOGGER.debug("[anaylze] %s finish", self.name.upper())

    @classmethod
    def get_config(cls, config, call):
        return {"valid_config": VALIDCONFIG, "default_config": DEFAULTCONFIG}

    def quantize_params(self, node, vertex_config):
        """quantize_params"""
        LOGGER.debug("[calibrate] %s start and quantized is %d", self.name.upper(), self.quantized)

        for arg in node.args:
            input_config = self.input_config[arg]
            y = _calibrate_core(arg, input_config, vertex_config, self.quantized)
            input_config.update(y)

    def realize(self, old_node, new_node, vertex_config, n2o):
        """realize"""
        LOGGER.debug("[realize]-- %s realize...", self.name.upper())

        old_arg = old_node.args[0]
        new_arg = new_node.args[0]
        output_config = vertex_config[old_arg].output_config
        input_config = self.input_config[old_arg]
        if (
            output_config["ref_count"] > 1
            and old_arg in n2o
            and (self.quantized or vertex_config[old_arg].quantized)
        ):
            new_arg = n2o[old_arg]
        else:
            new_arg = operate(input_config["operate"], new_arg, output_config, input_config, True)

        if (
            output_config["ref_count"] > 1
            and old_arg not in n2o
            and input_config["operate"] != "none"
        ):
            n2o[old_arg] = new_arg

        tmp = relay.frontend.common.infer_type(new_arg)
        if isinstance(new_arg, relay.Constant) and tmp.checked_type.dtype != "float16":
            new_arg = relay.const(new_arg.data.asnumpy(), "float16")
        elif tmp.checked_type.dtype.startswith("int"):
            new_arg = operate("dequantize", new_arg, self.input_config[old_arg], {}, True)
        elif tmp.checked_type.dtype != "float16":
            new_arg = relay.cast(new_arg, "float16")

        pair_node(old_arg, new_arg, {}, {"operate": "none"}, n2o, self.quantized)

        if new_node.attrs.scale == 1:
            gamma = relay.const(new_node.args[1].data.asnumpy(), "float16")
        else:
            gamma = None

        if new_node.attrs.center == 1:
            beta = relay.const(new_node.args[2].data.asnumpy(), "float16")
        else:
            beta = None

        axis = new_node.attrs.axis

        mean = relay.mean(new_arg, axis, True, False)

        # variance=relay.variance(new_arg,axis,True,False)
        tmp = relay.frontend.common.infer_type(relay.mean(new_arg, axis, True, True))
        n = tmp.checked_type.concrete_shape
        n = functools.reduce(lambda x, y: x * y, n)
        n = numpy.sqrt(n)
        n = relay.const(n, "float16")
        variance = relay.sum(((new_arg - mean) / n) * ((new_arg - mean) / n), axis, True, False)

        epsilon = relay.const(new_node.attrs.epsilon, "float16")
        std = relay.sqrt(variance + epsilon)
        new_node = (new_arg - mean) / std
        if gamma is not None:
            new_node = new_node * gamma
        if beta is not None:
            new_node = new_node + beta

        LOGGER.debug("[realize] %s finish", self.name.upper())
        return new_node
