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
# pylint: disable=unused-argument,inconsistent-return-statements,unexpected-keyword-arg,bad-continuation
"""op"""

import logging
import numpy
from tvm import relay
from tvm._ffi import runtime_ctypes
from ..threshold import Threshold
from ..method_dtype import Method, DataType, _get_dtype_info
from ..realize import operate, pair_node
from ..analyze import _quantized_judge
from ..calibrate import _calibrate_core

LOGGER = logging.getLogger("quantize")

__all__ = ("FixedOpTwoArgs",)

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


class FixedOpTwoArgs:
    """FixedOpTwoArgs"""

    name = "FixedOpTwoArgs"
    controlable = True

    def __init__(self, node, vertex_config, config):

        self.name = node.op.name
        self.op = node.op

        self.quantized = True

        if (
            not vertex_config[node.args[0]].quantized
            and not vertex_config[node.args[1]].quantized
            and self.name not in ["nn.bias_add"]
        ):
            self.quantized = False

        if node.op.name == "subtract":
            for arg in node.args:
                if not isinstance(arg, relay.Constant) and not vertex_config[arg].quantized:
                    self.quantized = False

        if "quantized" in config:
            self.quantized = config["quantized"]

        ci0 = config["input0"]
        ci1 = config["input1"]

        # input0_axis can support per-channel
        # the moment the best axis can support, most case can be perchannel
        input0_axis = vertex_config[node.args[0]].output_config["axis"]
        input1_axis = vertex_config[node.args[1]].output_config["axis"]

        if node.op.name == "nn.bias_add":
            input0_axis = node.attrs.axis
            input1_axis = 0

        # add arg1 is constant, 3-dims/4-dims support perch
        if (
            isinstance(node.args[1], relay.Constant)
            and len(node.args[1].data.shape) == 3
            and input0_axis > -1
        ):
            input1_axis = input0_axis - 1
        elif isinstance(node.args[1], relay.Constant) and len(node.args[1].data.shape) == 4:
            input1_axis = input0_axis
        elif isinstance(node.args[1], relay.Constant) and node.args[1].data.asnumpy().size == 1:
            input0_axis, input1_axis = -1, -1

        # set input config
        input0_config = _quantized_judge(
            vertex_config, node.args[0], input0_axis, self.quantized, ci0
        )
        input1_config = _quantized_judge(
            vertex_config, node.args[1], input1_axis, self.quantized, ci1
        )
        self.input_config = {node.args[0]: input0_config, node.args[1]: input1_config}

        self.axis = max(input0_axis, input1_axis)

        # the self.axis not the final axis, after calibrate is the last axis
        output0_config = {
            "dtype": DataType.Int32 if self.quantized else DataType.Float16,
            "axis": self.axis,
            "quantized_axis": "none",
            "ref_count": 0,
        }
        if self.quantized:
            output0_config.update(_get_dtype_info(output0_config["dtype"]))
        self.output_config = output0_config
        LOGGER.debug("[anaylze] %s finish", self.name.upper())

    @classmethod
    def get_config(cls, config, call):
        return {"valid_config": VALIDCONFIG, "default_config": DEFAULTCONFIG}

    def quantize_params(self, node, vertex_config):
        """quantize_params"""
        LOGGER.debug("[calibrate] %s start and quantized is %d", self.name.upper(), self.quantized)
        tmp = []

        for arg in node.args:
            input_config = self.input_config[arg]
            y = _calibrate_core(arg, input_config, vertex_config, self.quantized)
            input_config.update(y)
            tmp.append(y)
            # for fp 16
            if "scale" in y:
                LOGGER.debug("[calibrate]-- %s arg quantized_scale is:", self.name.upper())
                LOGGER.debug(y["scale"])

        if self.quantized:
            scale_left_np = tmp[0]["scale"]
            scale_right_np = tmp[1]["scale"]

            # only with "quantized_scale", can adjust scale
            if (
                "quantized_scale" in vertex_config[node.args[0]].output_config
                and "quantized_scale" in vertex_config[node.args[1]].output_config
            ):

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

                # reset the output_config['quantized_scale']
                if vertex_config[node.args[0]].output_config["ref_count"] == 1 and not numpy.all(
                    tmp[0]["scale"] - scale_left_np == 0
                ):
                    vertex_config[node.args[0]].output_config["quantized_scale"] = scale_left_np

                if vertex_config[node.args[1]].output_config["ref_count"] == 1 and not numpy.all(
                    tmp[1]["scale"] - scale_right_np == 0
                ):
                    vertex_config[node.args[1]].output_config["quantized_scale"] = scale_right_np

            # reset left input_config[scale]
            input_config = self.input_config[node.args[0]]
            input_config.update(
                {
                    "scale": vertex_config[node.args[0]].output_config["quantized_scale"]
                    if "quantized_scale" in vertex_config[node.args[0]].output_config
                    else vertex_config[node.args[0]].output_config["scale"],
                    "zero_point": numpy.zeros_like(scale_left_np, dtype=numpy.int32),
                }
            )
            # print("left", input_config["scale"])
            scale_left = input_config["scale"]
            LOGGER.debug("[calibrate]-- %s after adjust left_scale is:", self.name.upper())
            LOGGER.debug(input_config["scale"])

            # reset right input_config[scale]
            input_config = self.input_config[node.args[1]]
            input_config.update(
                {
                    "scale": vertex_config[node.args[1]].output_config["quantized_scale"]
                    if "quantized_scale" in vertex_config[node.args[1]].output_config
                    else vertex_config[node.args[1]].output_config["scale"],
                    "zero_point": numpy.zeros_like(scale_right_np, dtype=numpy.int32),
                }
            )
            # print("right", input_config["scale"])
            scale_right = input_config["scale"]
            LOGGER.debug("[calibrate]-- %s after adjust right_scale is:", self.name.upper())
            LOGGER.debug(input_config["scale"])

            # set the final output_config scale
            max_size = scale_left.size if scale_left.size > scale_right.size else scale_right.size
            scale_max = numpy.zeros([max_size])

            for i in range(max_size):
                left_scale = scale_left if scale_left.size == 1 else scale_left[i]
                right_scale = scale_right if scale_right.size == 1 else scale_right[i]
                scale_max[i] = left_scale if left_scale > right_scale else right_scale
            zero_point = numpy.zeros_like(scale_max, dtype=numpy.int32)

            new_y = {}
            new_y["scale"] = scale_max
            new_y["zero_point"] = zero_point

            new_y["axis"] = max(
                self.input_config[node.args[0]]["axis"], self.input_config[node.args[1]]["axis"]
            )

            if new_y["axis"] == 0:
                new_y["axis"] = 1
            self.axis = new_y["axis"]
            self.output_config.update(new_y)

    def realize(self, old_node, new_node, vertex_config, n2o):
        """realize"""
        LOGGER.debug("[realize]-- %s realize...", self.name.upper())
        if "DataType" in runtime_ctypes.__dict__:
            dtype = runtime_ctypes.DataType(self.input_config[old_node.args[0]]["dtype"])
        else:
            dtype = runtime_ctypes.TVMType(self.input_config[old_node.args[0]]["dtype"])

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

            # the pair_node can do after scale judegment! so no need here
            # pair_node(old_arg, new_arg, output_config, input_config, n2o, self.quantized)
            realized_args.append(new_arg)

        # consider fp16
        if not self.quantized:
            new_realized_args = []
            for old_arg, new_arg in zip(old_node.args, realized_args):
                tmp = relay.frontend.common.infer_type(new_arg)
                if isinstance(new_arg, relay.Constant) and tmp.checked_type.dtype != "float16":
                    new_arg = relay.const(new_arg.data.asnumpy().astype("float16"))
                # get int32, must be no-quantized op
                elif tmp.checked_type.dtype.startswith("int") and tmp.checked_type.dtype not in [
                    "int32"
                ]:
                    new_arg = operate("dequantize", new_arg, self.input_config[old_arg], {}, True)
                elif tmp.checked_type.dtype != "float16":
                    new_arg = relay.cast(new_arg, "float16")
                pair_node(old_arg, new_arg, {}, {"operate": "none"}, n2o, self.quantized)

                new_realized_args.append(new_arg)
            new_node = relay.Call(self.op, new_realized_args, old_node.attrs)

            return new_node

        # adjust add scale!!
        scale_left = self.input_config[old_node.args[0]]["scale"]
        scale_right = self.input_config[old_node.args[1]]["scale"]
        max_size = scale_left.size if scale_left.size > scale_right.size else scale_right.size
        scale_max = numpy.zeros([max_size])

        for i in range(max_size):
            left_scale = scale_left if scale_left.size == 1 else scale_left[i]
            right_scale = scale_right if scale_right.size == 1 else scale_right[i]
            scale_max[i] = left_scale if left_scale > right_scale else right_scale
        zero_point = numpy.zeros_like(scale_max, dtype=numpy.int32)

        LOGGER.debug("[realize] %s before adjust, left scale is:", self.name.upper())
        LOGGER.debug(scale_left)
        LOGGER.debug("[realize] %s before adjust, right scale is:", self.name.upper())
        LOGGER.debug(scale_right)
        LOGGER.debug("[realize] %s final max scale is:", self.name.upper())
        LOGGER.debug(scale_max)

        # print(len(numpy.where((scale_left == scale_right) == False)[0]))
        # print("add left scale is")
        # numpy.set_printoptions(precision=9, suppress=False)
        # print(scale_left)
        # print("add right scale is")
        # numpy.set_printoptions(precision=9, suppress=False)
        # print(scale_right)

        adjust_realized_args = []
        for old_arg, new_arg in zip(old_node.args, realized_args):

            adjust_input_config = {
                "zero_point": zero_point,
                "axis": self.axis,
                "scale": scale_max,
                "dtype": self.input_config[old_arg]["dtype"],
                "operate": "requantize",
            }
            # constant 3-dims axis = 0
            if isinstance(old_arg, relay.Constant) and self.input_config[old_arg]["axis"] > -1:
                adjust_input_config["axis"] = self.input_config[old_arg]["axis"]

            new_arg = operate(
                "requantize",
                new_arg,
                self.input_config[old_arg],
                adjust_input_config,
                True,
                multiplier=1,
            )

            # 400 no support output_dtype
            if self.quantized and "ir_pass" not in relay.__dict__:
                if dtype.CODE2STR[dtype.type_code] == "int" and dtype.bits < 32:
                    new_arg = relay.cast(new_arg, DataType.Int32)

            # todo no judge when add([1,c,h,h] + [1]) optimie to [1, c, h, w] + [c, 1, 1]
            # mobilenetv3 add( , 3f)
            if not isinstance(old_arg, relay.Constant):
                pair_node(
                    old_arg,
                    new_arg,
                    self.input_config[old_arg],
                    adjust_input_config,
                    n2o,
                    self.quantized,
                )
            adjust_realized_args.append(new_arg)

        if "ir_pass" in relay.__dict__:
            op_dict = {
                "add": relay.add,
                "subtract": relay.subtract,
            }

            if self.name != "nn.bias_add":
                new_node = op_dict[self.name](
                    adjust_realized_args[0], adjust_realized_args[1], out_dtype="int32"
                )
            else:
                new_node = relay.nn.bias_add(
                    adjust_realized_args[0],
                    adjust_realized_args[1],
                    axis=old_node.attrs.axis,
                    out_dtype="int32",
                )
        else:
            new_node = relay.Call(self.op, adjust_realized_args, old_node.attrs)

        LOGGER.debug("[realize] %s finish", self.name.upper())
        return new_node
