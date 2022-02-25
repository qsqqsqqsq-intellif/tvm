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
import numpy
from tvm import relay
from ..threshold import Threshold
from ..method_dtype import Method, DataType, _get_dtype_info
from ..analyze import _conv_counter, _set_conv_counter, _quantized_judge
from ..calibrate import _calibrate_core
from ..realize import _realize_core, operate

LOGGER = logging.getLogger("quantize")

__all__ = ("Conv2DTranspose",)

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


class Conv2DTranspose:
    """conv2d"""

    name = "nn.conv2d_transpose"
    controlable = True

    def __init__(self, node, vertex_config, config):
        cnt = _conv_counter()
        LOGGER.debug("[analyze] conv2d_transpose_%d start...", cnt)

        self.quantized = True
        if "skip_conv_layers" in config and cnt in config["skip_conv_layers"]:
            self.quantized = False

        _set_conv_counter(cnt + 1)

        ci0 = config["input0"]
        ci1 = config["input1"]

        """input0-config"""
        # conv2d input0_axis
        conv2d_groups = node.attrs.groups

        # todo nhwc?
        weigh_ci = node.args[1].data.shape[0]
        if conv2d_groups == weigh_ci and node.args[1].data.shape[1] == 1:
            input0_axis = node.attrs.data_layout.find("C")
        else:
            input0_axis = -1

        if (
            vertex_config[node.args[0]].quantized
            and vertex_config[node.args[0]].output_config["axis"] < input0_axis
        ):
            input0_axis = vertex_config[node.args[0]].output_config["axis"]

        # todo more consider!!!
        # if not self.quantized:
        #     input0_axis = -1

        # get input0_config
        input0_config = _quantized_judge(
            vertex_config, node.args[0], input0_axis, self.quantized, ci0
        )

        LOGGER.debug("[analyze] conv2d at first axis is %d", input0_config["axis"])

        # get input1-config  ic * gruops oc kw kh ==> oc * groups ic kw kh
        input1_axis = 10 + conv2d_groups
        input1_config = _quantized_judge(
            vertex_config, node.args[1], input1_axis, self.quantized, ci1
        )

        self.input_config = {
            node.args[0]: input0_config,
            node.args[1]: input1_config,
        }

        # get output_config
        # todo consider int16
        output0_axis = node.attrs.data_layout.find("C")
        output0_config = {
            "dtype": DataType.Int32 if self.quantized else DataType.Float16,
            "axis": output0_axis,
            "quantized_axis": "none",
            "ref_count": 0,
        }
        if self.quantized:
            output0_config.update(_get_dtype_info(output0_config["dtype"]))

        self.output_config = output0_config
        LOGGER.debug("[analyze] conv2d_%d finish", cnt)

    @classmethod
    def get_config(cls, call, config):
        return {"valid_config": VALIDCONFIG, "default_config": DEFAULTCONFIG}

    def quantize_params(self, node, vertex_config):
        """quantize_params"""
        conv_scale = []
        LOGGER.debug("[calibrate]conv2d nobias start and quantized is %d", self.quantized)

        # todo add fp16 no need to
        for arg in node.args[0:2]:
            input_config = self.input_config[arg]
            y = _calibrate_core(arg, input_config, vertex_config, self.quantized)
            LOGGER.debug("[calibrate]--conv2d arg quantized_scale is:")
            if "scale" in y:
                LOGGER.debug(y["scale"])
                # input_config is identity to 'quantize_scale'
                input_config.update(y)
                conv_scale.append(y["scale"])

        if self.quantized:
            scale = conv_scale[0] * conv_scale[1]
            zero_point = numpy.zeros_like(scale, dtype=numpy.int32)
            new_y = {"scale": scale, "zero_point": zero_point}

            self.output_config.update(new_y)

    def realize(self, old_node, new_node, vertex_config, n2o):
        """realize"""
        LOGGER.debug("[realize]conv2d_transpose start...")

        realized_args = []
        for old_arg, new_arg in zip(old_node.args, new_node.args):

            new_arg = _realize_core(self, old_arg, new_arg, vertex_config, n2o)

            realized_args.append(new_arg)

        attrs = {}
        attrs["strides"] = old_node.attrs.strides
        attrs["padding"] = old_node.attrs.padding
        attrs["dilation"] = old_node.attrs.dilation
        attrs["groups"] = old_node.attrs.groups
        attrs["channels"] = old_node.attrs.channels
        attrs["kernel_size"] = old_node.attrs.kernel_size
        attrs["data_layout"] = old_node.attrs.data_layout
        attrs["kernel_layout"] = old_node.attrs.kernel_layout
        attrs["output_padding"] = old_node.attrs.output_padding
        attrs["out_dtype"] = self.output_config["dtype"]
        if self.quantized:
            new_node = relay.nn.conv2d_transpose(realized_args[0], realized_args[1], **attrs)

            return new_node

        realized_args_n = []
        for old_arg, new_arg in zip(old_node.args, realized_args):
            tmp_expr = relay.frontend.common.infer_type(new_arg)
            if isinstance(new_arg, relay.Constant) and tmp_expr.checked_type.dtype != "float16":
                new_arg = relay.const(new_arg.data.asnumpy().astype("float16"))
            elif tmp_expr.checked_type.dtype.startswith("int"):
                new_arg = operate("dequantize", new_arg, self.input_config[old_arg], {}, True)
            elif tmp_expr.checked_type.dtype != "float16":
                new_arg = relay.cast(new_arg, "float16")
            realized_args_n.append(new_arg)

        new_node = relay.nn.conv2d_transpose(realized_args_n[0], realized_args_n[1], **attrs)
        return new_node
