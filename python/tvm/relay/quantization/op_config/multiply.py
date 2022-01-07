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
from tvm._ffi import runtime_ctypes
from ..threshold import Threshold
from ..method_dtype import Method, DataType, _get_dtype_info
from ..realize import _realize_core, operate, pair_node
from ..analyze import _conv_counter, _quantized_judge
from ..calibrate import _calibrate_core

LOGGER = logging.getLogger("quantize")

__all__ = ("Multiply",)

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


class Multiply:
    """multiply"""

    name = "multiply"
    controlable = True

    def __init__(self, node, vertex_config, config):
        cnt = _conv_counter()
        LOGGER.debug("[analyze] MULTIPLY start...")
        self.quantized = True

        if ("skip_conv_layers" in config and cnt - 1 in config["skip_conv_layers"]) or (
            not vertex_config[node.args[0]].quantized and not vertex_config[node.args[1]].quantized
        ):
            self.quantized = False

        ci0 = config["input0"]
        ci1 = config["input1"]

        # input0_axis can support per-channel
        # the moment the best axis can support, most case can be perchannel
        input0_axis = vertex_config[node.args[0]].output_config["axis"]
        input1_axis = vertex_config[node.args[1]].output_config["axis"]

        """input_config"""
        input0_config = _quantized_judge(
            vertex_config, node.args[0], input0_axis, self.quantized, ci0
        )
        input1_config = _quantized_judge(
            vertex_config, node.args[1], input1_axis, self.quantized, ci1
        )
        self.input_config = {node.args[0]: input0_config, node.args[1]: input1_config}

        output0_config = {
            "dtype": DataType.Int32 if self.quantized else DataType.Float16,
            "axis": max(input0_axis, input1_axis),
            "quantized_axis": "none",
            "ref_count": 0,
        }
        if self.quantized:
            output0_config.update(_get_dtype_info(output0_config["dtype"]))
        self.output_config = output0_config
        LOGGER.debug("[anaylze] MULTIPLY finish")

    @classmethod
    def get_config(cls, config, call):
        return {"valid_config": VALIDCONFIG, "default_config": DEFAULTCONFIG}

    def quantize_params(self, node, vertex_config):
        """quantize_params"""
        LOGGER.debug("[calibrate] MULTIPLY start and quantized is %d", self.quantized)
        tmp = []
        # todo consider fp16

        multiply_scale = []

        for arg in node.args:
            input_config = self.input_config[arg]
            y = _calibrate_core(arg, input_config, vertex_config, self.quantized)
            input_config.update(y)
            tmp.append(y)
            # consider for fp16
            if "scale" in y:
                multiply_scale.append(y["scale"])
                LOGGER.debug("[calibrate]-- MULTIPLY arg quantized_scale is:")
                LOGGER.debug(y["scale"])

        # consider for fp16
        if self.quantized:
            scale = multiply_scale[0] * multiply_scale[1]
            zero_point = numpy.zeros_like(scale, dtype=numpy.int32)
            new_y = {"scale": scale, "zero_point": zero_point}

            self.output_config.update(new_y)

    def realize(self, old_node, new_node, vertex_config, n2o):
        """realize"""
        LOGGER.debug("[realize]-- MULTIPLY realize...")
        dtype = runtime_ctypes.DataType(self.input_config[old_node.args[0]]["dtype"])

        realized_args = []
        for old_arg, new_arg in zip(old_node.args, new_node.args):

            new_arg = _realize_core(self, old_arg, new_arg, vertex_config, n2o)

            if self.quantized:
                if dtype.CODE2STR[dtype.type_code] == "int" and dtype.bits < 32:
                    new_arg = relay.cast(new_arg, DataType.Int32)
            realized_args.append(new_arg)

        if not self.quantized:
            new_realized_args = []
            for old_arg, new_arg in zip(old_node.args, realized_args):
                tmp = relay.frontend.common.infer_type(new_arg)
                if isinstance(new_arg, relay.Constant) and tmp.checked_type.dtype != "float16":
                    new_arg = relay.const(new_arg.data.asnumpy(), "float16")
                elif tmp.checked_type.dtype.startswith("int"):
                    new_arg = operate("dequantize", new_arg, self.input_config[old_arg], {}, True)
                elif tmp.checked_type.dtype != "float16":
                    new_arg = relay.cast(new_arg, "float16")
                pair_node(old_arg, new_arg, {}, {"operate": "none"}, n2o, self.quantized)

                new_realized_args.append(new_arg)

            new_node = relay.multiply(new_realized_args[0], new_realized_args[1])

            return new_node

        new_node = relay.multiply(realized_args[0], realized_args[1])

        LOGGER.debug("[realize] MULTIPLY finish")
        return new_node
