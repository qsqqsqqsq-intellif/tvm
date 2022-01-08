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
from ..threshold import Threshold
from ..method_dtype import _get_dtype_info, DataType, Method
from ..realize import operate
from ..analyze import _conv_counter

LOGGER = logging.getLogger("quantize")

__all__ = ("Concatenate",)

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


class Concatenate:
    """Concatenate"""

    name = "concatenate"
    controlable = False

    def __init__(self, node, vertex_config, config):
        LOGGER.debug("[analyze] concatenate start")
        cnt = _conv_counter()
        arg = node.args[0]

        self.quantized = vertex_config[arg].quantized
        if cnt in []:
            self.quantized = False

        vertex_config[arg].output_config["ref_count"] = (
            vertex_config[arg].output_config["ref_count"] + 1
        )

        # todo according to axis judge quantized!

        input0_config = {
            "dtype": vertex_config[arg].output_config["dtype"],
            "axis": vertex_config[arg].output_config["axis"],
            "method": Method.Symmetry if self.quantized else None,
            "threshold": None,
            "operate": "none",
        }
        if input0_config["dtype"].startswith("int"):
            input0_config.update(_get_dtype_info(input0_config["dtype"]))

        self.input_config = {arg: input0_config}

        """output0_config"""
        output0_config = {
            "dtype": DataType.Int32 if self.quantized else DataType.Float16,
            "axis": vertex_config[arg].output_config["axis"],
            "quantized_axis": "none",
            "ref_count": 0,
        }

        if output0_config["dtype"] not in [DataType.Float16]:
            output0_config.update(_get_dtype_info(output0_config["dtype"]))  # todo modify this
        self.output_config = output0_config
        LOGGER.debug("[analyze] concatenate end")

    @classmethod
    def get_config(cls, config, call):
        return {"valid_config": VALIDCONFIG, "default_config": DEFAULTCONFIG}

    def quantize_params(self, node, vertex_config):
        """quantize_params"""
        LOGGER.info("[calibrate] concatenate start")
        arg_tuple = node.args[0]

        tuple_config = vertex_config[arg_tuple]
        input_config = tuple_config.input_config

        scale_all_scalar = 1  # 0: all scalar
        axis = -1
        for arg in arg_tuple.fields:
            scale_arg = input_config[arg]["scale"]
            if scale_arg.size > 1:
                scale_all_scalar = 0
                axis = input_config[arg]["axis"]
                LOGGER.debug("[calibrate] axis is %d", axis)
                break

        if scale_all_scalar == 1:
            scale_max = numpy.array(1.0, "float32")
            LOGGER.debug("[calibrate] concat input scale is all scalar, value is:")
            for arg in arg_tuple:
                scale_arg = input_config[arg]["scale"]
                scale_max = numpy.max([scale_max, scale_arg])
                LOGGER.debug(scale_arg)

            for arg in arg_tuple:
                # no need to do quantized/requantize, so may no have "quantized_scale"
                if (
                    vertex_config[arg].output_config["ref_count"] == 1
                    and "quantized_scale" in vertex_config[arg].output_config
                ):
                    vertex_config[arg].output_config["quantized_scale"] = scale_max
                    # output_config[quantized_scale] modify, input_config['scale'] modify same time
                    input_config[arg]["scale"] = scale_max

            concat_scale = scale_max
        else:
            concat_scale = numpy.array([])
            for arg in arg_tuple:
                scale_arg = input_config[arg]["scale"]
                arg_shape = relay.frontend.common.infer_type(arg).checked_type.shape
                if scale_arg.size > 1:
                    concat_scale = numpy.append(concat_scale, scale_arg)
                else:
                    # use axis, because of nhwc
                    concat_scale = numpy.append(
                        concat_scale, (scale_arg * numpy.ones(arg_shape[axis].value))
                    )

        zero_point = numpy.zeros_like(concat_scale, dtype=numpy.int32)
        y = {"scale": concat_scale, "zero_point": zero_point}
        self.input_config[arg_tuple].update(y)

        self.output_config.update(y)

    def realize(self, old_node, new_node, vertex_config, n2o):
        """realize"""
        LOGGER.info("[realize] concatenate start... and quantized is %d", self.quantized)
        old_arg = old_node.args[0]
        new_arg = new_node.args[0]

        if not self.quantized:
            return new_node

        tuple_config = vertex_config[old_arg]
        input_config = tuple_config.input_config
        # todo  add more judge strategy here
        scale_all_scalar = 1  # 0: all scalar
        for arg in old_arg.fields:
            scale_arg = input_config[arg]["scale"]
            if scale_arg.size > 1:
                scale_all_scalar = 0
                break

        if scale_all_scalar:
            scale_new = input_config[old_arg.fields[0]]["scale"]
            for arg_ in old_arg.fields[1:]:
                scale_new = max(scale_new, input_config[arg_]["scale"])

            self.adjust_input_config = {}
            self.adjust_input_config = {
                "zero_point": self.input_config[old_arg]["zero_point"],
                "axis": self.input_config[old_arg]["axis"],
                "dtype": self.input_config[old_arg]["dtype"],
                "scale": scale_new,
            }

            new_tuple_node = []
            for old_arg_, new_arg_ in zip(old_arg.fields, new_arg.fields):
                new_arg_ = operate(
                    "requantize", new_arg_, input_config[old_arg_], self.adjust_input_config, True
                )
                new_tuple_node.append(new_arg_)

            new_tup = relay.Tuple(new_tuple_node, old_arg.span)

            self.input_config[old_arg]["scale"] = scale_new

            self.output_config["scale"] = scale_new

            new_node = relay.Call(
                old_node.op, [new_tup], new_node.attrs, new_node.type_args, new_node.span
            )
            return new_node

        # TODO ADD MORE CODE DEAL vector scale!!!
        new_node = relay.Call(
            old_node.op, [new_arg], new_node.attrs, new_node.type_args, new_node.span
        )
        return new_node
