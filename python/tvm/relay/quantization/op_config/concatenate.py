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
# pylint: disable=unused-argument,inconsistent-return-statements,bad-continuation,too-many-function-args
"""op"""

import logging
import numpy
from tvm import relay
from ..threshold import Threshold
from ..method_dtype import _get_dtype_info, DataType, Method
from ..realize import operate

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
        arg = node.args[0]

        self.quantized = vertex_config[arg].quantized

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

    def realize(self, old_node, new_node, vertex_config, n2o):
        """realize"""
        LOGGER.info("[realize] concatenate start... and quantized is %d", self.quantized)
        old_arg = old_node.args[0]
        new_arg = new_node.args[0]

        if not self.quantized:
            return new_node

        tuple_config = vertex_config[old_arg]
        input_config = tuple_config.input_config

        # for arg in old_arg.fields:
        #     scale_arg = input_config[arg]["scale"]
        #     print("concat arg scale")
        #     print(scale_arg)

        scale_all_scalar = 1  # 0: all scalar
        axis_list = []
        for arg in old_arg.fields:
            axis_list.append(input_config[arg]["axis"])
            if input_config[arg]["axis"] > -1:
                scale_all_scalar = 0
        axis = max(axis_list)

        if scale_all_scalar == 1:
            concat_scale = numpy.array([])
            arg_shape = relay.frontend.common.infer_type(old_arg.fields[0]).checked_type.shape
            if len(arg_shape) == 4 and self.output_config["axis"] == 1:
                # axis use perchannel
                axis = old_node.attrs.axis
                for arg in old_arg:
                    temp_scale = input_config[arg]["scale"]
                    arg_shape = relay.frontend.common.infer_type(arg).checked_type.shape
                    arg_len = arg_shape[axis].value
                    temp_scale = numpy.ones([arg_len]) * temp_scale
                    concat_scale = numpy.append(concat_scale, temp_scale)
            else:
                scale_new = input_config[old_arg.fields[0]]["scale"]
                for arg_ in old_arg.fields[1:]:
                    scale_new = max(scale_new, input_config[arg_]["scale"])

                axis = -1
                adjust_input_config = {
                    "zero_point": input_config[old_arg.fields[0]]["zero_point"],
                    "axis": axis,
                    "dtype": input_config[old_arg.fields[0]]["dtype"],
                    "scale": scale_new,
                }

                new_tuple_node = []
                for old_arg_, new_arg_ in zip(old_arg.fields, new_arg.fields):
                    new_arg_ = operate(
                        "requantize",
                        new_arg_,
                        input_config[old_arg_],
                        adjust_input_config,
                        True,
                        multiplier=1,
                    )
                    new_tuple_node.append(new_arg_)

                if "ir_pass" not in relay.__dict__:
                    new_tup = relay.Tuple(new_tuple_node, old_arg.span)
                else:
                    new_tup = relay.Tuple(new_tuple_node)

                self.input_config[old_arg].update(adjust_input_config)
                self.output_config.update(adjust_input_config)

                if "ir_pass" not in relay.__dict__:
                    new_node = relay.Call(
                        old_node.op, [new_tup], new_node.attrs, new_node.type_args, new_node.span
                    )
                else:
                    new_node = relay.Call(
                        old_node.op, [new_tup], new_node.attrs, new_node.type_args
                    )
                return new_node

        elif scale_all_scalar == 0:
            concat_scale = numpy.array([])
            for arg in old_arg:
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
        y = {"scale": concat_scale, "zero_point": zero_point, "axis": axis}
        self.input_config[old_arg].update(y)

        self.output_config.update(y)

        # 300 no support node.span
        if "ir_pass" not in relay.__dict__:
            new_node = relay.Call(
                old_node.op, [new_arg], new_node.attrs, new_node.type_args, new_node.span
            )
        else:
            new_node = relay.Call(old_node.op, [new_arg], new_node.attrs, new_node.type_args)
        return new_node
