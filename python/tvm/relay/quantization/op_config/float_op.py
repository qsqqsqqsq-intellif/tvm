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
from ..method_dtype import Method, DataType
from ..realize import operate, pair_node
from ..analyze import _quantized_judge
from ..calibrate import _calibrate_core

LOGGER = logging.getLogger("quantize")

__all__ = ("FloatOp",)

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


class FloatOp:
    """FixedOp"""

    name = "float_op"
    controlable = True

    def __init__(self, node, vertex_config, config):
        if isinstance(node.op, relay.Function):
            self.name = getattr(node.op.attrs, "Composite")
        else:
            self.name = node.op.name

        LOGGER.debug("[analyze] %s start...", self.name)
        self.quantized = False

        self.input_config = {}
        for i, one_arg in enumerate(node.args):
            tmp = "input" + str(i)

            input_axis = -1
            if vertex_config[one_arg].quantized:
                input_axis = vertex_config[one_arg].output_config["axis"]

            # about avgpool, support perchannel
            if (
                self.name.split("_")[-1].startswith("pool")
                and node.attrs is not None
                and "layout" in node.attrs.keys()
                and node.attrs.layout in ["NCHW", "NHWC"]
            ):
                input_axis = node.attrs.layout.find("C")

            input_config = _quantized_judge(
                vertex_config, one_arg, input_axis, self.quantized, config[tmp]
            )
            self.input_config[one_arg] = input_config

        output0_config = {
            "dtype": DataType.Float16,
            "axis": -1,
            "quantized_axis": "none",
            "ref_count": 0,
        }
        self.output_config = output0_config
        LOGGER.debug("[anaylze] %s finish", self.name)

    @classmethod
    def get_config(cls, config, call):
        return {"valid_config": VALIDCONFIG, "default_config": DEFAULTCONFIG}

    def quantize_params(self, node, vertex_config):
        """quantize_params"""
        LOGGER.debug("[calibrate] %s start and quantized is %d", self.name, self.quantized)

        for arg in node.args:
            input_config = self.input_config[arg]
            y = _calibrate_core(arg, input_config, vertex_config, self.quantized)
            input_config.update(y)

    def realize(self, old_node, new_node, vertex_config, n2o):
        """realize"""
        LOGGER.debug("[realize]-- %s realize...", self.name)

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

            realized_args.append(new_arg)

        use_fp16, use_fp32, use_i32 = False, False, False

        for arg_ in realized_args:
            tmp = relay.frontend.common.infer_type(arg_)
            if isinstance(arg_, (relay.Call, relay.TupleGetItem)) and tmp.checked_type.dtype in [
                "int32",
                "int64",
            ]:
                use_i32 = True
            elif (
                isinstance(arg_, (relay.Call, relay.TupleGetItem))
                and tmp.checked_type.dtype == "float32"
            ):
                use_fp32 = True
            elif (
                isinstance(arg_, relay.Constant)
                and tmp.checked_type.dtype.startswith("int")
                and int(tmp.checked_type.dtype[3:]) > 16
            ):
                use_i32 = True
            elif isinstance(arg_, relay.Constant) and tmp.checked_type.dtype.startswith("float"):
                # abs_min = numpy.min(numpy.abs(arg_.data.asnumpy()))
                # abs_max = numpy.max(numpy.abs(arg_.data.asnumpy()))
                # if (
                #     (abs_min > 0 and abs_min < pow(2.0, -24))
                #     or (abs_max > 0 and abs_max < pow(2.0, -24))
                #     or (abs_max > 65504)
                # ):
                abs_max = numpy.max(numpy.abs(arg_.data.asnumpy()))
                # int32 and float, use float32
                if abs_max > 65504 or use_i32:
                    use_fp32 = True
            else:
                use_fp16 = True

        if use_fp32 or use_i32:
            use_fp16 = False
        if use_fp32 and use_i32:
            use_i32 = False

        if self.name in ["vision.non_max_suppression"]:
            assert isinstance(realized_args[2], relay.Constant), "vision.nms arg2 should be const"
            realized_args[2] = relay.const(realized_args[2].data.asnumpy().astype("int16"))
            new_node = relay.Call(old_node.op, realized_args, old_node.attrs)
            LOGGER.debug("[realize] %s finish", self.name)
            return new_node

        new_realized_args = []
        if use_fp16:
            for old_arg, new_arg in zip(old_node.args, realized_args):
                tmp = relay.frontend.common.infer_type(new_arg)
                if isinstance(new_arg, relay.Constant) and tmp.checked_type.dtype != "float16":
                    new_arg = relay.const(new_arg.data.asnumpy().astype("float16"))
                elif tmp.checked_type.dtype.startswith("int"):
                    new_arg = operate("dequantize", new_arg, self.input_config[old_arg], {}, True)
                elif tmp.checked_type.dtype != "float16" and not (
                    isinstance(old_arg, relay.Var) and old_arg.name_hint == "im_info"
                ):
                    new_arg = relay.cast(new_arg, "float16")

                pair_node(old_arg, new_arg, {}, {"operate": "none"}, n2o, self.quantized)

                new_realized_args.append(new_arg)

        elif use_i32:
            LOGGER.info("[realize]-- %s use int32...", self.name)
            for old_arg, new_arg in zip(old_node.args, realized_args):
                tmp = relay.frontend.common.infer_type(new_arg)
                if isinstance(new_arg, relay.Constant) and tmp.checked_type.dtype != "int32":
                    new_arg = relay.const(new_arg.data.asnumpy().astype("int32"))
                elif tmp.checked_type.dtype != "int32":
                    new_arg = relay.cast(new_arg, "int32")

                pair_node(old_arg, new_arg, {}, {"operate": "none"}, n2o, self.quantized)

                new_realized_args.append(new_arg)

        else:
            LOGGER.info("[realize]-- %s use float32...", self.name)
            for old_arg, new_arg in zip(old_node.args, realized_args):
                tmp = relay.frontend.common.infer_type(new_arg)
                if isinstance(new_arg, relay.Constant) and tmp.checked_type.dtype != "float32":
                    new_arg = relay.const(new_arg.data.asnumpy().astype("float32"))
                elif tmp.checked_type.dtype != "float32":
                    new_arg = relay.cast(new_arg, "float32")

                pair_node(old_arg, new_arg, {}, {"operate": "none"}, n2o, self.quantized)

                new_realized_args.append(new_arg)

        new_node = relay.Call(old_node.op, new_realized_args, old_node.attrs)

        LOGGER.debug("[realize] %s finish", self.name)
        return new_node
