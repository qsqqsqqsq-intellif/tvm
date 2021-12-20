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
from ..realize import operate, pair_node
from ..analyze import _conv_counter, _quantized_judge
from ..calibrate import _calibrate_core

LOGGER = logging.getLogger("quantize")

__all__ = ("Add",)

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
    "threshold": Threshold.RelativeEntropy,
    "method": Method.Symmetry,
    "dtype": DataType.Int8,
}


class Add:
    """Add"""

    name = "add"
    controlable = True

    def __init__(self, node, vertex_config, config):

        cnt = _conv_counter
        LOGGER.debug("[analyze] ADD start...")
        self.quantized = True
        # todo  quantize judge from args?
        if cnt in []:
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
        output0_config.update(_get_dtype_info(output0_config["dtype"]))
        self.output_config = output0_config
        LOGGER.debug("[anaylze] ADD finish")

    @classmethod
    def get_config(cls, config, call):
        return {"valid_config": VALIDCONFIG, "default_config": DEFAULTCONFIG}

    def quantize_params(self, node, vertex_config):
        """quantize_params"""
        LOGGER.debug("[calibrate] ADD start and quantized is %d", self.quantized)
        tmp = []
        # todo consider fp16
        for arg in node.args:
            input_config = self.input_config[arg]
            y = _calibrate_core(arg, input_config, vertex_config, self.quantized)
            input_config.update(y)
            tmp.append(y)
            LOGGER.debug("[calibrate]-- ADD arg quantized_scale is:")
            LOGGER.debug(y["scale"])

        # --add adjust strategy, if size diff, use the biggest!
        # ---TODO also can use other strategy
        scale_left_np = tmp[0]["scale"]
        scale_right_np = tmp[1]["scale"]

        if scale_left_np.size != scale_right_np.size:
            if scale_right_np.size == 1:
                scale_max = numpy.max([scale_right_np, numpy.max(scale_left_np)])
                scale_right_np = scale_max
                scale_left_np = scale_max * numpy.ones(scale_left_np.size)
            else:
                assert scale_left_np.size == 1, "args1 size should be 1"
                scale_max = numpy.max([scale_left_np, numpy.max(scale_right_np)])
                scale_left_np = scale_max
                scale_right_np = scale_max * numpy.ones(scale_right_np.size)
        else:
            scale_right_np = numpy.amax([scale_right_np, scale_left_np], axis=0)
            scale_left_np = scale_right_np

        scale = numpy.maximum(scale_left_np, scale_right_np)
        self.max_scale = scale
        zero_point = numpy.zeros_like(scale, dtype=numpy.int32)
        axis = max(self.input_config[node.args[0]]["axis"], self.input_config[node.args[1]]["axis"])
        new_y = {"scale": scale, "zero_point": zero_point, "axis": axis}

        # reset the output_config['quantized_scale']
        if vertex_config[node.args[0]].output_config["ref_count"] == 1 and not numpy.all(
            tmp[0]["scale"] - scale_left_np == 0
        ):
            vertex_config[node.args[0]].output_config["quantized_scale"] = scale_left_np
            # no need to recusive set, do it always calibrate.py

        if vertex_config[node.args[1]].output_config["ref_count"] == 1 and not numpy.all(
            tmp[1]["scale"] - scale_right_np == 0
        ):
            vertex_config[node.args[1]].output_config["quantized_scale"] = scale_right_np
            # no need to recusive set, do it always calibrate.py

        # reset the input_config[scale]
        input_config = self.input_config[node.args[0]]
        input_config.update(
            {
                "scale": vertex_config[node.args[0]].output_config["quantized_scale"],
                "zero_point": numpy.zeros_like(scale_left_np, dtype=numpy.int32),
            }
        )
        LOGGER.debug("[calibrate]-- ADD after adjust left_scale is:")
        LOGGER.debug(input_config["scale"])

        input_config = self.input_config[node.args[1]]
        input_config.update(
            {
                "scale": vertex_config[node.args[1]].output_config["quantized_scale"],
                "zero_point": numpy.zeros_like(scale_right_np, dtype=numpy.int32),
            }
        )
        LOGGER.debug("[calibrate]-- ADD after adjust right_scale is:")
        LOGGER.debug(input_config["scale"])

        self.output_config.update(new_y)

    def realize(self, old_node, new_node, vertex_config, n2o):
        """realize"""
        LOGGER.debug("[realize]-- ADD realize...")
        dtype = runtime_ctypes.DataType(self.input_config[old_node.args[0]]["dtype"])

        realized_args = []
        for old_arg, new_arg in zip(old_node.args, new_node.args):

            output_config = vertex_config[old_arg].output_config
            input_config = self.input_config[old_arg]
            if output_config["ref_count"] > 1 and old_arg in n2o:
                new_arg = n2o[old_arg]
            else:
                new_arg = operate(
                    input_config["operate"], new_arg, output_config, input_config, True
                )

            if output_config["ref_count"] > 1 and old_arg not in n2o:
                n2o[old_arg] = new_arg

            # if self.quantized:
            #     if dtype.CODE2STR[dtype.type_code] == "int" and dtype.bits < 32:
            #         new_arg = relay.cast(new_arg, DataType.Int32)

            # the pair_node can do after scale judegment! so no need here
            # pair_node(old_arg, new_arg, output_config, input_config, n2o, self.quantized)
            realized_args.append(new_arg)

        # todo consider fp16
        # adjust add scale!!
        scale_left = self.input_config[old_node.args[0]]["scale"]
        scale_right = self.input_config[old_node.args[1]]["scale"]
        scale_max = self.max_scale

        LOGGER.debug("[realize] ADD before adjust, left scale is:")
        LOGGER.debug(scale_left)
        LOGGER.debug("[realize] ADD before adjust, right scale is:")
        LOGGER.debug(scale_right)
        LOGGER.debug("[realize] ADD final max scale is:")
        LOGGER.debug(scale_max)

        # get the final adjust scale
        self.adjust_input_config = {}
        self.adjust_input_config[old_node.args[0]] = {
            "zero_point": self.input_config[old_node.args[0]]["zero_point"],
            "axis": self.input_config[old_node.args[0]]["axis"],
            "dtype": self.input_config[old_node.args[0]]["dtype"],
            "operate": "requantize",
        }

        self.adjust_input_config[old_node.args[1]] = {
            "zero_point": self.input_config[old_node.args[1]]["zero_point"],
            "axis": self.input_config[old_node.args[1]]["axis"],
            "dtype": self.input_config[old_node.args[1]]["dtype"],
            "operate": "requantize",
        }

        if scale_left.size != scale_right.size:
            if scale_left.size == 1:
                self.adjust_input_config[old_node.args[0]].update({"scale": scale_max[0]})
                self.adjust_input_config[old_node.args[1]].update({"scale": scale_max})
            else:
                assert scale_right.size == 1, "cur scale size must be 1"
                self.adjust_input_config[old_node.args[0]].update({"scale": scale_max})
                self.adjust_input_config[old_node.args[1]].update({"scale": scale_max[0]})

        else:
            self.adjust_input_config[old_node.args[0]].update({"scale": scale_max})
            self.adjust_input_config[old_node.args[1]].update({"scale": scale_max})

        adjust_realized_args = []
        for old_arg, new_arg in zip(old_node.args, realized_args):
            new_arg = operate(
                "requantize",
                new_arg,
                self.input_config[old_arg],
                self.adjust_input_config[old_arg],
                True,
            )

            # TODO add should support int8 + int8 => int32
            if self.quantized:
                if dtype.CODE2STR[dtype.type_code] == "int" and dtype.bits < 32:
                    new_arg = relay.cast(new_arg, DataType.Int32)
            # todo confirm self.quantized
            pair_node(
                old_arg,
                new_arg,
                self.input_config[old_arg],
                self.adjust_input_config[old_arg],
                n2o,
                self.quantized,
            )
            adjust_realized_args.append(new_arg)
        new_node = relay.add(adjust_realized_args[0], adjust_realized_args[1])
        LOGGER.debug("[realize] ADD finish")
        return new_node
