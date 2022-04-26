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
from ..analyze import _quantized_judge
from ..calibrate import _calibrate_core
from ..realize import _realize_core, operate

LOGGER = logging.getLogger("quantize")

__all__ = ("Conv2DWinogradBiasAdd",)

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


class Conv2DWinogradBiasAdd:
    """Conv2DWinogradBiasAdd"""

    name = "conv2d_winograd1d_bias_add"
    controlable = True

    def __init__(self, node, vertex_config, config):

        LOGGER.debug("[analyze] Conv2DWinogradBiasAdd start")
        self.quantized = True
        if "quantized" in config:
            self.quantized = config["quantized"]

        self.bias_method = 0
        if "bias_method" in config:
            self.bias_method = config["bias_method"]

        ci0 = config["input0"]
        ci1 = config["input1"]
        ci2 = config["input2"]

        temp = []

        def fvisit(expr):
            if isinstance(expr, relay.Call) and expr != node:
                temp.append(expr)

        if "analysis" in relay.__dict__:
            relay.analysis.post_order_visit(node.op, fvisit)
        else:
            relay.ir_pass.post_order_visit(node.op, fvisit)
        conv2d = temp[0]
        bias_add = temp[1]
        self._inner_config = {}

        """input0-config"""
        # conv2d input0_axis
        conv2d_groups = conv2d.attrs.groups
        weight_co_axis = conv2d.attrs.kernel_layout.find("O")
        weigh_co = node.args[1].data.shape[weight_co_axis]
        weight_ci_axis = conv2d.attrs.kernel_layout.find("I")
        weigh_ci = node.args[1].data.shape[weight_ci_axis]

        if conv2d_groups == weigh_co and weigh_ci == 1:
            input0_axis = conv2d.attrs.data_layout.find("C")

            # input is global sumpool, do pertensor
            arg_call = node.args[0]
            name = "var"
            if isinstance(arg_call, relay.Call) and isinstance(arg_call.op, relay.Function):
                name = getattr(arg_call.op.attrs, "Composite")
                if not isinstance(name, str):
                    name = name.value
            elif isinstance(arg_call, relay.Call):
                name = arg_call.op.name

            sumpool_is_global = 0
            if name in ["nn.sum_pool2d"]:
                pool_size = arg_call.attrs.pool_size
                pool_shape = arg_call.type_args[0].shape
                if len(pool_shape) == 4 and len(pool_size) == 2:
                    if (
                        pool_shape[2].value == pool_shape[3].value
                        and pool_size[0].value == pool_size[1].value
                        and pool_shape[2].value == pool_size[0].value
                    ):
                        sumpool_is_global = 1

            if name in ["nn.global_sum_pool2d"] or sumpool_is_global == 1:
                input0_axis = -1
        else:
            input0_axis = -1

        if (
            vertex_config[node.args[0]].quantized
            and vertex_config[node.args[0]].output_config["axis"] < input0_axis
        ):
            input0_axis = vertex_config[node.args[0]].output_config["axis"]

        # get input0_config
        input0_config = _quantized_judge(
            vertex_config, node.args[0], input0_axis, self.quantized, ci0
        )
        # todo add logger information here
        LOGGER.debug("[analyze] Conv2DWinogradBiasAdd at first axis is %d", input0_config["axis"])

        # get input1-config
        input1_axis = conv2d.attrs.kernel_layout.find("O")
        input1_config = _quantized_judge(
            vertex_config, node.args[1], input1_axis, self.quantized, ci1
        )

        conv2d_input_config = {conv2d.args[0]: input0_config, conv2d.args[1]: input1_config}

        conv_out_layout = (
            conv2d.attrs.out_layout if conv2d.attrs.out_layout != "" else conv2d.attrs.data_layout
        )

        conv2d_out_axis = conv_out_layout.find("C")

        # todo int16 support int64!!!
        conv2d_output_config = {
            "dtype": DataType.Int32 if self.quantized else DataType.Float16,
            "axis": conv2d_out_axis,
        }

        self._inner_config[conv2d] = {
            "input_config": conv2d_input_config,
            "output_config": conv2d_output_config,
        }

        input2_config = {
            "dtype": conv2d_output_config["dtype"],
            "axis": conv2d_output_config["axis"],
            "method": None,
            "threshold": None,
            "operate": "none",
        }
        if self.quantized:
            input2_config.update(_get_dtype_info(input2_config["dtype"]))

        input3_axis = 0
        if self.bias_method == 0:
            input3_config = {
                "dtype": config["input2"]["dtype"] if self.quantized else DataType.Float16,
                "axis": input3_axis,
                "method": None,
                "threshold": None,
                "operate": "quantize" if self.quantized else "none",
            }
            vertex_config[node.args[2]].output_config.update(
                {
                    "quantized_axis": input3_axis,
                }
            )
            if self.quantized:
                input3_config.update(_get_dtype_info(input3_config["dtype"]))
        else:
            input3_config = _quantized_judge(
                vertex_config, node.args[2], input3_axis, self.quantized, ci2
            )

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

        # get output_config
        output0_axis = conv2d_out_axis
        output0_config = {
            "dtype": DataType.Int32 if self.quantized else DataType.Float16,
            "axis": output0_axis,
            "quantized_axis": "none",
            "ref_count": 0,
        }
        if self.quantized:
            output0_config.update(_get_dtype_info(output0_config["dtype"]))

        self.output_config = output0_config
        LOGGER.debug("[analyze] Conv2DWinogradBiasAdd finish")

    @classmethod
    def get_config(cls, call, config):
        return {"valid_config": VALIDCONFIG, "default_config": DEFAULTCONFIG}

    def quantize_params(self, node, vertex_config):
        """quantize_params"""
        conv_scale = []
        LOGGER.debug("[calibrate]Conv2DWinogradBiasAdd start and quantized is %d", self.quantized)

        # todo add fp16 no need to
        for arg in node.args[0:2]:
            input_config = self.input_config[arg]
            y = _calibrate_core(arg, input_config, vertex_config, self.quantized)
            LOGGER.debug("[calibrate]--Conv2DWinogradBiasAdd arg quantized_scale is:")
            if "scale" in y:
                LOGGER.debug(y["scale"])
                input_config.update(y)
                conv_scale.append(y["scale"])

        # self.d_scale = conv_scale[0]
        # self.w_scale = conv_scale[1]

        if self.quantized:
            scale = conv_scale[0] * conv_scale[1]
            zero_point = numpy.zeros_like(scale, dtype=numpy.int32)
            new_y = {"scale": scale, "zero_point": zero_point}
            self.input_config[node.args[2]].update(new_y)

            self.output_config.update(new_y)

        if self.bias_method == 1 and self.quantized:
            input_config = self.input_config[node.args[2]]
            bias_scale = _calibrate_core(node.args[2], input_config, vertex_config, self.quantized)
            bias_scale["scale"][numpy.where(bias_scale["scale"] == 0.01 / 127)] = 0.01 / (2 ** 23)
            maxscale = numpy.max([scale, bias_scale["scale"]], axis=0)
            weigh_adjust = scale / maxscale
            bias_adjust = bias_scale["scale"] / maxscale

            self.input_config[node.args[1]]["scale"] = (
                self.input_config[node.args[1]]["scale"] / weigh_adjust
            )

            zero_point = numpy.zeros_like(maxscale, dtype=numpy.int32)
            self.input_config[node.args[2]]["scale"] = bias_scale["scale"] / bias_adjust
            self.input_config[node.args[2]]["zero_point"] = zero_point

            new_y = {"scale": maxscale, "zero_point": zero_point}

            self.output_config.update(new_y)

    def realize(self, old_node, new_node, vertex_config, n2o):
        """realize"""
        LOGGER.debug("[realize]Conv2DWinogradBiasAdd start...")

        realized_args = []
        for old_arg, new_arg in zip(old_node.args, new_node.args):

            new_arg = _realize_core(self, old_arg, new_arg, vertex_config, n2o)
            realized_args.append(new_arg)

        # print("conv2d scale: ")
        # print("left scale: ", self.d_scale)
        # print("right scale: ", self.w_scale)
        # print("weight sum is ", realized_args[1].data.asnumpy().flatten().sum())
        # print("bias is ", realized_args[2].data)

        tmp = []

        def fvisit(expr):
            if isinstance(expr, relay.Call) and expr != old_node:
                tmp.append(expr)

        if "analysis" in relay.__dict__:
            relay.analysis.post_order_visit(old_node.op, fvisit)
        else:
            relay.ir_pass.post_order_visit(old_node.op, fvisit)

        if self.quantized:
            conv2d_attrs = dict(tmp[0].attrs)
            conv2d_attrs["out_dtype"] = self._inner_config[tmp[0]]["output_config"]["dtype"]
            if self.input_config[old_node.args[0]]["dtype"] == "int16":
                conv2d_attrs["out_dtype"] = "int64"
            conv2d_node = relay.nn.contrib_conv2d_winograd1d_without_weight_transform(
                realized_args[0], realized_args[1], **conv2d_attrs
            )
            bias_attrs = dict(tmp[1].attrs)
            if "ir_pass" in relay.__dict__:
                bias_attrs["out_dtype"] = conv2d_attrs["out_dtype"]
            bias_node = relay.nn.bias_add(conv2d_node, realized_args[2], **bias_attrs)

            if self.input_config[old_node.args[0]]["dtype"] == "int16":
                # for conv cpu no support data-kernel dtype diff
                weight_int16 = relay.const(conv2d_node.args[1].data.asnumpy(), "int16")
                bias_int64 = relay.const(bias_node.args[1].data.asnumpy(), "int64")
                conv2d_node = relay.nn.contrib_conv2d_winograd1d_without_weight_transform(
                    realized_args[0], weight_int16, **conv2d_attrs
                )
                bias_node = relay.nn.bias_add(conv2d_node, bias_int64)
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

        conv2d_attrs = dict(tmp[0].attrs)
        conv2d_attrs["out_dtype"] = self._inner_config[tmp[0]]["output_config"]["dtype"]
        conv2d_node = relay.nn.contrib_conv2d_winograd1d_without_weight_transform(
            realized_args_n[0], realized_args_n[1], **conv2d_attrs
        )
        bias_attrs = dict(tmp[1].attrs)
        bias_node = relay.nn.bias_add(conv2d_node, realized_args_n[2], **bias_attrs)

        return bias_node
