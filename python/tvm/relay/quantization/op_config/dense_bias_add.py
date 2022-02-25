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

__all__ = ("DenseBiasAdd",)

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


class DenseBiasAdd:
    """dense_bias_add"""

    name = "dense_bias_add"
    controlable = True

    def __init__(self, node, vertex_config, config):
        cnt = _conv_counter()
        LOGGER.debug("[anaylze] dense_%d start", cnt)

        # todo add judge to modify self.quantized to be false
        # todo -- ex: strides info, conv2d_idx, get from outside config
        # todo get config to support partial-quantize
        self.quantized = True
        if "skip_conv_layers" in config and cnt in config["skip_conv_layers"]:
            self.quantized = False

        _set_conv_counter(cnt + 1)

        ci0 = config["input0"]
        ci1 = config["input1"]

        temp = []

        def fvisit(expr):
            if isinstance(expr, relay.Call) and expr != node:
                temp.append(expr)

        relay.analysis.post_order_visit(node.op, fvisit)
        dense = temp[0]
        bias_add = temp[1]
        self._inner_config = {}

        """input0_config"""
        # dense only support  per-tensor
        input0_axis = -1
        input0_config = _quantized_judge(
            vertex_config, node.args[0], input0_axis, self.quantized, ci0
        )

        """input1-config"""
        # weight support per-channel
        input1_axis = 0
        input1_config = _quantized_judge(
            vertex_config, node.args[1], input1_axis, self.quantized, ci1
        )
        dense_input_config = {dense.args[0]: input0_config, dense.args[1]: input1_config}

        dense_out_axis = 1

        dense_output_config = {
            "dtype": DataType.Int32 if self.quantized else DataType.Float16,
            "axis": dense_out_axis,
        }

        self._inner_config[dense] = {
            "input_config": dense_input_config,
            "output_config": dense_output_config,
        }

        input2_config = {
            "dtype": dense_output_config["dtype"],
            "axis": dense_output_config["axis"],
            "method": None,
            "threshold": None,
            "operate": "none",
        }
        if self.quantized:
            input2_config.update(_get_dtype_info(input2_config["dtype"]))
        if input2_config["axis"] == -1:
            input3_axis = -1
        else:
            input3_axis = 0
        input3_config = {
            "dtype": input2_config["dtype"],
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

        if input2_config["axis"] == -1:
            bias_out_axis = -1
        else:
            bias_out_axis = input2_config["axis"]

        bias_output_config = {
            "dtype": DataType.Int32 if self.quantized else DataType.Float16,
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

        for arg in node.args[1:]:
            tmp1 = {}
            if not vertex_config[arg].quantized and self.quantized:
                tmp1.update({"operate": "quantize"})
            elif vertex_config[arg].quantized and self.quantized:
                if self.input_config[arg]["threshold"] is not None:
                    tmp1.update({"operate": "requantize"})
            self.input_config[arg].update(tmp1)

        output0_axis = bias_output_config["axis"]
        output0_config = {
            "dtype": DataType.Int32 if self.quantized else DataType.Float16,
            "axis": output0_axis,
            "quantized_axis": "none",
            "ref_count": 0,
        }
        if self.quantized:
            output0_config.update(_get_dtype_info(output0_config["dtype"]))
        self.output_config = output0_config
        LOGGER.debug("[anaylze] dense_%d finish", cnt)

    @classmethod
    def get_config(cls, call, config):
        return {"valid_config": VALIDCONFIG, "default_config": DEFAULTCONFIG}

    def quantize_params(self, node, vertex_config):
        """quantize_params"""
        dense_scale = []
        LOGGER.debug("[calibrate]dense_biasadd start and quantized is %d", self.quantized)

        # todo add fp16 no need to
        for arg in node.args[0:2]:
            input_config = self.input_config[arg]
            y = _calibrate_core(arg, input_config, vertex_config, self.quantized)
            LOGGER.debug("[calibrate]--dense arg quantized_scale is:")
            if "scale" in y:
                LOGGER.debug(y["scale"])
                input_config.update(y)
                dense_scale.append(y["scale"])

        if self.quantized:
            scale = dense_scale[0] * dense_scale[1]
            zero_point = numpy.zeros_like(scale, dtype=numpy.int32)
            new_y = {"scale": scale, "zero_point": zero_point}
            self.input_config[node.args[2]].update(new_y)

            self.output_config.update(new_y)

    def realize(self, old_node, new_node, vertex_config, n2o):
        """realize"""
        LOGGER.debug("[realize]dense_bias_add start...")
        realized_args = []
        for old_arg, new_arg in zip(old_node.args, new_node.args):
            new_arg = _realize_core(self, old_arg, new_arg, vertex_config, n2o)
            realized_args.append(new_arg)

        tmp = []

        def fvisit(expr):
            if isinstance(expr, relay.Call) and expr != old_node:
                tmp.append(expr)

        relay.analysis.post_order_visit(old_node.op, fvisit)

        if self.quantized:
            dense_attrs = dict(tmp[0].attrs)
            dense_attrs["out_dtype"] = self._inner_config[tmp[0]]["output_config"]["dtype"]
            dense_node = relay.nn.dense(realized_args[0], realized_args[1], **dense_attrs)
            bias_attrs = dict(tmp[1].attrs)
            bias_node = relay.nn.bias_add(dense_node, realized_args[2], **bias_attrs)

            return bias_node

        realized_args_n = []
        for old_arg, new_arg in zip(old_node.args, realized_args):
            tmp_expr = relay.frontend.common.infer_type(new_arg)
            if isinstance(new_arg, relay.Constant) and tmp_expr.checked_type.dtype != "float16":
                new_arg = relay.const(new_arg.data.asnumpy(), "float16")
            elif tmp_expr.checked_type.dtype.startswith("int"):
                new_arg = operate("dequantize", new_arg, self.input_config[old_arg], {}, True)
            elif tmp_expr.checked_type.dtype != "float16":
                new_arg = relay.cast(new_arg, "float16")
            realized_args_n.append(new_arg)

        dense_attrs = dict(tmp[0].attrs)
        dense_attrs["out_dtype"] = self._inner_config[tmp[0]]["output_config"]["dtype"]
        dense_node = relay.nn.dense(realized_args_n[0], realized_args_n[1], **dense_attrs)
        bias_attrs = dict(tmp[1].attrs)
        bias_node = relay.nn.bias_add(dense_node, realized_args_n[2], **bias_attrs)

        return bias_node
