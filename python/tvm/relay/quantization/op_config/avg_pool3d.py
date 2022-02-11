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
from tvm._ffi import runtime_ctypes
from ..threshold import Threshold
from ..method_dtype import Method, DataType, _get_dtype_info
from ..realize import operate, pair_node

__all__ = ("AvgPool3D",)

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


class AvgPool3D:
    """avg_pool3d"""

    name = "nn.avg_pool3d"
    controlable = False

    def __init__(self, node, vertex_config, config):
        arg = node.args[0]

        if vertex_config[arg].quantized:
            self.quantized = True

            input0_config = {
                "dtype": DataType.Int32,
                "axis": vertex_config[arg].output_config["axis"],
                "method": None,
                "threshold": None,
            }
            input0_config.update(_get_dtype_info(input0_config["dtype"]))
            input0_config.update({"in_dtype": vertex_config[arg].output_config["dtype"]})
            input0_config.update({"in_axis": vertex_config[arg].output_config["axis"]})
            input0_config.update({"operate": "none"})
            self.input_config = {arg: input0_config}

            output0_config = {
                "dtype": DataType.Int8,
                "axis": input0_config["axis"],
                "method": None,
                "threshold": None,
            }
            output0_config.update(_get_dtype_info(output0_config["dtype"]))
            output0_config.update(
                {
                    "in_dtype": input0_config["dtype"],
                    "in_axis": input0_config["axis"],
                    "operate": "none",
                }
            )
            self.output_config = output0_config
        else:
            self.quantized = False

            input0_config = {}
            input0_config.update({"in_dtype": vertex_config[arg].output_config["dtype"]})
            input0_config.update({"in_axis": vertex_config[arg].output_config["axis"]})
            input0_config.update({"operate": "none"})
            self.input_config = {arg: input0_config}

            self.output_config = {
                "dtype": node.checked_type.dtype,
                "axis": -1,
            }

    @classmethod
    def get_config(cls, config, call):
        return {"valid_config": VALIDCONFIG, "default_config": DEFAULTCONFIG}

    def quantize_params(self, node, vertex_config):
        """quantize_params"""
        arg = node.args[0]
        input_config = self.input_config[arg]
        scale = vertex_config[arg].output_config["scale"]
        zero_point = vertex_config[arg].output_config["zero_point"]
        y = {"scale": scale, "zero_point": zero_point}
        input_config.update(y)

        self.output_config.update(y)

    def realize(self, old_node, new_node, vertex_config, n2o):
        """realize"""
        old_arg = old_node.args[0]
        new_arg = new_node.args[0]

        output_config = vertex_config[old_arg].output_config
        input_config = self.input_config[old_arg]
        new_arg = operate(input_config["operate"], new_arg, output_config, input_config, True)

        dtype = runtime_ctypes.DataType(input_config["in_dtype"])
        if self.quantized:
            if dtype.CODE2STR[dtype.type_code] == "int" and dtype.bits < 32:
                new_arg = relay.cast(new_arg, input_config["dtype"])

        pair_node(old_arg, new_arg, output_config, input_config, n2o, self.quantized)
        new_node = relay.nn.avg_pool3d(new_arg, **dict(new_node.attrs))

        if self.quantized:
            if dtype.CODE2STR[dtype.type_code] == "int" and dtype.bits < 32:
                new_node = relay.cast(new_node, self.output_config["dtype"])
        return new_node
