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

import numpy
from tvm import relay
from tvm._ffi import runtime_ctypes
from ..threshold import Threshold
from ..method_dtype import Granularity, Method, DataType, _get_dtype_info
from ..realize import operate, pair_node

__all__ = ("Subtract",)

VALIDCONFIG = {
    "input0": {
        "threshold": (
            Threshold.MinMax,
            Threshold.Percentile,
            Threshold.MovingAverageMinMax,
            Threshold.L2Norm,
            Threshold.RelativeEntropy,
        ),
        "granularity": (Granularity.Tensor, Granularity.Channel),
        "method": (Method.Symmetry, Method.Asymmetry),
        "dtype": (DataType.Int8,),
    },
    "input1": {
        "threshold": (
            Threshold.MinMax,
            Threshold.Percentile,
            Threshold.MovingAverageMinMax,
            Threshold.L2Norm,
            Threshold.RelativeEntropy,
        ),
        "granularity": (Granularity.Tensor, Granularity.Channel),
        "method": (Method.Symmetry, Method.Asymmetry),
        "dtype": (DataType.Int8,),
    },
    "output0": {
        "threshold": (
            Threshold.MinMax,
            Threshold.Percentile,
            Threshold.MovingAverageMinMax,
            Threshold.L2Norm,
            Threshold.RelativeEntropy,
        ),
        "granularity": (Granularity.Tensor,),
        "method": (Method.Symmetry, Method.Asymmetry),
        "dtype": (DataType.Int8,),
    },
}

DEFAULTCONFIG = {
    "input0": {
        "threshold": Threshold.MinMax,
        "granularity": Granularity.Tensor,
        "method": Method.Symmetry,
        "dtype": DataType.Int8,
    },
    "input1": {
        "threshold": Threshold.MinMax,
        "granularity": Granularity.Tensor,
        "method": Method.Symmetry,
        "dtype": DataType.Int8,
    },
    "output0": {
        "threshold": Threshold.RelativeEntropy,
        "granularity": Granularity.Tensor,
        "method": Method.Symmetry,
        "dtype": DataType.Int8,
    },
}


class Subtract:
    """subtract"""

    name = "subtract"
    controlable = True

    def __init__(self, node, vertex_config, config):
        if config:
            self.quantized = True

            ci0 = config["input0"]
            ci1 = config["input1"]
            co0 = config["output0"]

            condition1 = vertex_config[node.args[0]].output_config["dtype"] == ci0["dtype"]
            _is_per_channel1 = vertex_config[node.args[0]].output_config["axis"] != -1
            _is_per_channel2 = ci0["granularity"] == Granularity.Channel
            condition2 = _is_per_channel1 == _is_per_channel2
            arg0_inherit = condition1 and condition2

            condition1 = vertex_config[node.args[1]].output_config["dtype"] == ci1["dtype"]
            _is_per_channel1 = vertex_config[node.args[1]].output_config["axis"] != -1
            _is_per_channel2 = ci1["granularity"] == Granularity.Channel
            condition2 = _is_per_channel1 == _is_per_channel2
            arg1_inherit = condition1 and condition2

            if (
                ci0["granularity"] == Granularity.Tensor
                and ci1["granularity"] == Granularity.Tensor
            ):
                input0_axis = -1
                input1_axis = -1
                out_axis = -1
            else:
                raise NotImplementedError

            input0_config = {
                "dtype": ci0["dtype"],
                "axis": input0_axis,
                "method": None if arg0_inherit else ci0["method"],
                "threshold": None
                if arg0_inherit
                else ci0["threshold"](node.args[0], input0_axis, ci0),
            }
            input0_config.update(_get_dtype_info(input0_config["dtype"]))
            input1_config = {
                "dtype": ci1["dtype"],
                "axis": input1_axis,
                "method": None if arg1_inherit else ci1["method"],
                "threshold": None
                if arg1_inherit
                else ci1["threshold"](node.args[1], input1_axis, ci1),
            }
            input1_config.update(_get_dtype_info(input1_config["dtype"]))
            self.input_config = {node.args[0]: input0_config, node.args[1]: input1_config}

            for arg in node.args:
                tmp1 = {}
                tmp1.update({"in_dtype": vertex_config[arg].output_config["dtype"]})
                tmp1.update({"in_axis": vertex_config[arg].output_config["axis"]})
                if not vertex_config[arg].quantized and self.quantized:
                    tmp2 = {"operate": "quantize"}
                elif vertex_config[arg].quantized and self.quantized:
                    tmp2 = {"operate": "requantize"}
                tmp1.update(tmp2)
                self.input_config[arg].update(tmp1)

            output0_config = {
                "dtype": co0["dtype"],
                "axis": out_axis,
                "method": co0["method"],
                "threshold": co0["threshold"](node, out_axis, co0),
            }
            output0_config.update(_get_dtype_info(output0_config["dtype"]))
            output0_config.update(
                {"in_dtype": DataType.Int32, "in_axis": input0_axis, "operate": "requantize"}
            )
            self.output_config = output0_config
        else:
            self.quantized = False

            self.input_config = {}
            for arg in node.args:
                tmp1 = {}
                tmp1.update({"in_dtype": vertex_config[arg].output_config["dtype"]})
                tmp1.update({"in_axis": vertex_config[arg].output_config["axis"]})
                if not vertex_config[arg].quantized and not self.quantized:
                    tmp2 = {"operate": "none"}
                elif vertex_config[arg].quantized and not self.quantized:
                    tmp2 = {"operate": "dequantize"}
                tmp1.update(tmp2)
                self.input_config[arg] = tmp1

            self.output_config = {
                "dtype": node.checked_type.dtype,
                "axis": -1,
            }

    @classmethod
    def get_config(cls, config, call):
        return {"valid_config": VALIDCONFIG, "default_config": DEFAULTCONFIG}

    def quantize_params(self, node, vertex_config):
        """quantize_params"""
        tmp = []
        for arg in node.args:
            input_config = self.input_config[arg]
            if input_config["method"] is not None:
                y = input_config["method"](input_config)
            else:
                scale = vertex_config[arg].output_config["scale"]
                zero_point = vertex_config[arg].output_config["zero_point"]
                y = {"scale": scale, "zero_point": zero_point}
            tmp.append(y)

        scale = numpy.array(float("-inf"), numpy.float32)
        for one in tmp:
            scale = numpy.maximum(scale, one["scale"])
        zero_point = numpy.zeros_like(scale, dtype=numpy.int32)
        new_y = {"scale": scale, "zero_point": zero_point}
        for arg in node.args:
            input_config = self.input_config[arg]
            input_config.update(new_y)

        y = self.output_config["method"](self.output_config)
        self.output_config.update(y)

    def realize(self, old_node, new_node, vertex_config, n2o):
        """realize"""
        dtype = runtime_ctypes.DataType(self.input_config[old_node.args[0]]["in_dtype"])

        realized_args = []
        for old_arg, new_arg in zip(old_node.args, new_node.args):
            output_config = vertex_config[old_arg].output_config
            input_config = self.input_config[old_arg]
            new_arg = operate(input_config["operate"], new_arg, output_config, input_config, True)
            if self.quantized:
                if dtype.CODE2STR[dtype.type_code] == "int" and dtype.bits < 32:
                    new_arg = relay.cast(new_arg, DataType.Int32)
            realized_args.append(new_arg)
            pair_node(old_arg, new_arg, output_config, input_config, n2o, self.quantized)
        new_node = relay.subtract(realized_args[0], realized_args[1])

        input_config = self.input_config[old_node.args[0]]
        scale = input_config["scale"]
        zero_point = input_config["zero_point"]
        axis = input_config["axis"]
        config = {"scale": scale, "zero_point": zero_point, "axis": axis}
        new_node = operate(
            self.output_config["operate"], new_node, config, self.output_config, False
        )

        return new_node
