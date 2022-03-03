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

__all__ = ("Conv2DTransposeBiasAdd",)

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


class Conv2DTransposeBiasAdd:
    """conv2d_bias_add"""

    name = "conv2d_transpose_bias_add"
    controlable = True

    def __init__(self, node, vertex_config, config):

        cnt = _conv_counter()
        LOGGER.debug("[analyze] conv2d_%d start...", cnt)
        # todo add judge to modify self.quantized to be false
        # todo -- ex: strides info, conv2d_idx, get from outside config
        # todo get config to support partial-quantize

        self.quantized = True
        if "skip_conv_layers" in config and cnt in config["skip_conv_layers"]:
            self.quantized = False

        _set_conv_counter(cnt + 1)

        # todo get outside
        ci0 = config["input0"]
        ci1 = config["input1"]

        temp = []

        def fvisit(expr):
            if isinstance(expr, relay.Call) and expr != node:
                temp.append(expr)

        if "analysis" in relay.__dict__:
            relay.analysis.post_order_visit(node.op, fvisit)
        else:
            relay.ir_pass.post_order_visit(node.op, fvisit)

        conv2d_transpose = temp[0]
        bias_add = temp[1]
        self._inner_config = {}

        """input0-config"""
        conv2d_transpose_groups = conv2d_transpose.attrs.groups
        weigh_ci = node.args[1].data.shape[0]
        if conv2d_transpose_groups == weigh_ci and node.args[1].data.shape[1] == 1:
            input0_axis = node.attrs.data_layout.find("C")
        else:
            input0_axis = -1

        if (
            vertex_config[node.args[0]].quantized
            and vertex_config[node.args[0]].output_config["axis"] < input0_axis
        ):
            input0_axis = vertex_config[node.args[0]].output_config["axis"]

        # todo
        # if not self.quantized:
        #     input0_axis = -1

        # get input0_config
        input0_config = _quantized_judge(
            vertex_config, node.args[0], input0_axis, self.quantized, ci0
        )
        # todo add logger information here
        LOGGER.debug("[analyze] conv2d_transpose at first axis is %d", input0_config["axis"])

        # get input1-config
        input1_axis = 10 + conv2d_transpose_groups
        input1_config = _quantized_judge(
            vertex_config, node.args[1], input1_axis, self.quantized, ci1
        )

        conv2d_transpose_input_config = {
            conv2d_transpose.args[0]: input0_config,
            conv2d_transpose.args[1]: input1_config,
        }

        conv_out_layout = (
            conv2d_transpose.attrs.out_layout
            if conv2d_transpose.attrs.out_layout != ""
            else conv2d_transpose.attrs.data_layout
        )

        conv2d_transpose_out_axis = conv_out_layout.find("C")

        # todo int16 support int64!!!
        conv2d_transpose_output_config = {
            "dtype": DataType.Int32 if self.quantized else DataType.Float16,
            "axis": conv2d_transpose_out_axis,
        }

        self._inner_config[conv2d_transpose] = {
            "input_config": conv2d_transpose_input_config,
            "output_config": conv2d_transpose_output_config,
        }

        input2_config = {
            "dtype": conv2d_transpose_output_config["dtype"],
            "axis": conv2d_transpose_output_config["axis"],
            "method": None,
            "threshold": None,
            "operate": "none",
        }
        if self.quantized:
            input2_config.update(_get_dtype_info(input2_config["dtype"]))

        input3_axis = 0
        bias_dtype = DataType.Int32
        if "target" in config and config["target"].startswith("nnp3"):
            bias_dtype = DataType.Int24
        input3_config = {
            "dtype": bias_dtype if self.quantized else DataType.Float32,
            "axis": input3_axis,
            "method": None,
            "threshold": None,
            "operate": "none",
        }
        vertex_config[node.args[2]].output_config.update(
            {
                "quantized_axis": input3_axis,
            }
        )
        if self.quantized:
            input3_config.update(_get_dtype_info(input3_config["dtype"]))
        bias_input_config = {bias_add.args[0]: input2_config, bias_add.args[1]: input3_config}

        bias_out_axis = input2_config["axis"]

        bias_output_config = {
            "dtype": DataType.Int32,
            "axis": bias_out_axis,
        }

        self._inner_config[bias_add] = {
            "input_config": bias_input_config,
            "output_config": bias_output_config,
        }

        self.input_config = {
            node.args[0]: input0_config,
            node.args[1]: input1_config,
            node.args[2]: input3_config,
        }

        # get 'operate'
        # todo use self.quantized to do weight quantized
        for arg in node.args[1:]:
            tmp1 = {}
            if not vertex_config[arg].quantized and self.quantized:
                tmp1.update({"operate": "quantize"})
            elif vertex_config[arg].quantized and self.quantized:
                if self.input_config[arg]["threshold"] is not None:
                    tmp1.update({"operate": "requantize"})
            self.input_config[arg].update(tmp1)

        # get output_config
        output0_axis = conv2d_transpose_out_axis
        output0_config = {
            "dtype": DataType.Int32 if self.quantized else DataType.Float16,
            "axis": output0_axis,
            "quantized_axis": "none",
            "ref_count": 0,
        }
        if self.quantized:
            output0_config.update(_get_dtype_info(output0_config["dtype"]))

        self.output_config = output0_config
        LOGGER.debug("[analyze] conv2d_transpose_%d finish", cnt)

    @classmethod
    def get_config(cls, call, config):
        return {"valid_config": VALIDCONFIG, "default_config": DEFAULTCONFIG}

    def quantize_params(self, node, vertex_config):
        """quantize_params"""
        conv_scale = []
        LOGGER.debug(
            "[calibrate]conv2d_transpose_biasadd start and quantized is %d", self.quantized
        )

        # todo add fp16 no need to
        for arg in node.args[0:2]:
            input_config = self.input_config[arg]
            y = _calibrate_core(arg, input_config, vertex_config, self.quantized)
            LOGGER.debug("[calibrate]--conv2d_transpose arg quantized_scale is:")
            if "scale" in y:
                LOGGER.debug(y["scale"])
                input_config.update(y)
                conv_scale.append(y["scale"])

        if self.quantized:
            scale = conv_scale[0] * conv_scale[1]
            zero_point = numpy.zeros_like(scale, dtype=numpy.int32)
            new_y = {"scale": scale, "zero_point": zero_point}
            self.input_config[node.args[2]].update(new_y)

            self.output_config.update(new_y)

    def realize(self, old_node, new_node, vertex_config, n2o):
        """realize"""
        LOGGER.debug("[realize]conv2d_transpose_bias_add start...")

        realized_args = []
        # print("conv2d_transpose scale")
        # print("left scale is", self.input_config[old_node.args[0]]["scale"])
        # print("right scale is", self.input_config[old_node.args[1]]["scale"])
        for old_arg, new_arg in zip(old_node.args, new_node.args):
            new_arg = _realize_core(self, old_arg, new_arg, vertex_config, n2o)
            realized_args.append(new_arg)

        tmp = []
        print("weight sum is ", realized_args[1].data.asnumpy().flatten().sum())
        print("bias is ", realized_args[2].data)

        def fvisit(expr):
            if isinstance(expr, relay.Call) and expr != old_node:
                tmp.append(expr)

        if "analysis" in relay.__dict__:
            relay.analysis.post_order_visit(old_node.op, fvisit)
        else:
            relay.ir_pass.post_order_visit(old_node.op, fvisit)

        conv2d_transpose_attrs = {}
        conv2d_transpose_attrs["strides"] = tmp[0].attrs.strides
        conv2d_transpose_attrs["padding"] = tmp[0].attrs.padding
        conv2d_transpose_attrs["dilation"] = tmp[0].attrs.dilation
        conv2d_transpose_attrs["groups"] = tmp[0].attrs.groups
        conv2d_transpose_attrs["channels"] = tmp[0].attrs.channels
        conv2d_transpose_attrs["kernel_size"] = tmp[0].attrs.kernel_size
        conv2d_transpose_attrs["data_layout"] = tmp[0].attrs.data_layout
        conv2d_transpose_attrs["kernel_layout"] = tmp[0].attrs.kernel_layout
        conv2d_transpose_attrs["output_padding"] = tmp[0].attrs.output_padding

        if self.quantized:
            conv2d_transpose_attrs["out_dtype"] = self._inner_config[tmp[0]]["output_config"][
                "dtype"
            ]
            if self.input_config[old_node.args[0]]["dtype"] == "int16":
                conv2d_transpose_attrs["out_dtype"] = "int64"
            conv2d_transpose_node = relay.nn.conv2d_transpose(
                realized_args[0], realized_args[1], **conv2d_transpose_attrs
            )

            bias_attrs = dict(tmp[1].attrs)
            if "ir_pass" in relay.__dict__:
                bias_attrs["out_dtype"] = conv2d_transpose_attrs["out_dtype"]
            bias_node = relay.nn.bias_add(conv2d_transpose_node, realized_args[2], **bias_attrs)

            if self.input_config[old_node.args[0]]["dtype"] == "int16":
                # for conv cpu no support data-kernel dtype diff
                weight_int16 = relay.const(
                    conv2d_transpose_node.args[1].data.asnumpy().astype("int16")
                )
                bias_int64 = relay.const(bias_node.args[1].data.asnumpy().astype("int64"))
                conv2d_transpose_node = relay.nn.conv2d_transpose(
                    realized_args[0], weight_int16, **conv2d_transpose_attrs
                )
                bias_node = relay.nn.bias_add(conv2d_transpose_node, bias_int64)
                bias_node = relay.clip(bias_node, -(2 ** 31), 2 ** 31 - 1)
                bias_node = relay.cast(bias_node, "int32")

            return bias_node

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

        conv2d_transpose_attrs["out_dtype"] = self._inner_config[tmp[0]]["output_config"]["dtype"]
        conv2d_transpose_node = relay.nn.conv2d_transpose(
            realized_args_n[0], realized_args_n[1], **conv2d_transpose_attrs
        )
        bias_attrs = dict(tmp[1].attrs)
        bias_node = relay.nn.bias_add(conv2d_transpose_node, realized_args_n[2], **bias_attrs)

        return bias_node
