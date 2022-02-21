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
"""calibrate"""

import logging
import numpy
from tvm import relay

LOGGER = logging.getLogger("quantize")

BACK_IDENTITYSCALE_OP_LIST = ["clip", "nn.relu", "nn.max_pool2d", "nn.max_pool3d"]


def recursive_set(arg, vertex_config):
    """recursive optimize scale"""
    arg_config = vertex_config[arg]
    # --if not split node
    # --quantized_scale/axis
    # --- output_config['quantized_scale'] --> output_config[scale]
    # --> input_config['scale'] --> output_config['quantized_config']
    if (
        isinstance(arg, relay.Call)
        and not isinstance(arg.op, relay.Function)
        and arg_config.quantized
        and arg.op.name in BACK_IDENTITYSCALE_OP_LIST + ["concatenate"]
    ):
        if (
            arg.op.name in BACK_IDENTITYSCALE_OP_LIST
            and vertex_config[arg.args[0]].output_config["ref_count"] == 1
            and "quantized_scale" in vertex_config[arg.args[0]].output_config
        ):

            quantized_scale = arg_config.output_config["quantized_scale"]
            quantized_axis = arg_config.output_config["quantized_axis"]

            LOGGER.debug("[calibrate]<recursive_set> scale start and the op is %s", arg.op.name)
            LOGGER.debug("[calibrate]<recursive_set> quantized scale is:")
            LOGGER.debug(quantized_scale)
            LOGGER.debug(
                "[calibrate]<recursive_set> quantized axis is %d",
                arg_config.output_config["quantized_axis"],
            )
            # step1
            arg_config.output_config["scale"] = quantized_scale
            arg_config.output_config["zero_point"] = numpy.zeros_like(
                quantized_scale, dtype=numpy.int32
            )
            arg_config.output_config["axis"] = quantized_axis

            arg_config.input_config[arg.args[0]]["scale"] = quantized_scale
            arg_config.input_config[arg.args[0]]["zero_point"] = numpy.zeros_like(
                quantized_scale, dtype=numpy.int32
            )
            arg_config.input_config[arg.args[0]]["axis"] = quantized_axis

            vertex_config[arg.args[0]].output_config["quantized_scale"] = quantized_scale
            vertex_config[arg.args[0]].output_config["quantized_zero_point"] = numpy.zeros_like(
                quantized_scale, dtype=numpy.int32
            )
            vertex_config[arg.args[0]].output_config["quantized_axis"] = quantized_axis

            recursive_set(arg.args[0], vertex_config)

        if arg.op.name in ["concatenate"] and arg_config.quantized:
            # for arg_ in arg.fields:
            arg_tuple = arg.args[0]
            arg_split_node = 0
            arg_all_hava_quantized = 1
            for arg_ in arg_tuple.fields:
                if vertex_config[arg_].output_config["ref_count"] > 1:
                    arg_split_node = 1
                if "quantized_scale" not in vertex_config[arg_].output_config:
                    arg_all_hava_quantized = 0

            if arg_split_node == 0 and arg_all_hava_quantized == 1:
                quantized_scale = arg_config.output_config["quantized_scale"]
                quantized_axis = arg_config.output_config["quantized_axis"]
                # arg_config is the config of concat
                arg_config.output_config["scale"] = quantized_scale
                arg_config.output_config["zero_point"] = numpy.zeros_like(
                    quantized_scale, dtype=numpy.int32
                )
                arg_config.output_config["axis"] = quantized_axis

                arg_config.input_config[arg_tuple]["scale"] = quantized_scale
                arg_config.input_config[arg_tuple]["zero_point"] = numpy.zeros_like(
                    quantized_scale, dtype=numpy.int32
                )
                arg_config.input_config[arg_tuple]["axis"] = quantized_axis

                index_start = 0
                for arg_ in arg_tuple.fields:
                    if quantized_scale.size == 1:

                        vertex_config[arg_tuple].input_config[arg_]["scale"] = quantized_scale
                        vertex_config[arg_tuple].input_config[arg_][
                            "zero_point"
                        ] = numpy.zeros_like(quantized_scale, dtype=numpy.int32)
                        vertex_config[arg_tuple].input_config[arg_]["axis"] = quantized_axis

                        vertex_config[arg_].output_config["quantized_scale"] = quantized_scale
                        vertex_config[arg_].output_config[
                            "quantized_zero_point"
                        ] = numpy.zeros_like(quantized_scale, dtype=numpy.int32)
                        vertex_config[arg_].output_config["quantized_axis"] = quantized_axis

                        recursive_set(arg_, vertex_config)
                    else:
                        arg_shape = relay.frontend.common.infer_type(arg_).checked_type.shape
                        shape_value = arg_shape[arg.attrs.axis].value

                        vertex_config[arg_tuple].input_config[arg_]["scale"] = quantized_scale[
                            index_start : index_start + shape_value
                        ]
                        vertex_config[arg_tuple].input_config[arg_]["zero_point"] = numpy.zeros(
                            shape_value, dtype=numpy.int32
                        )
                        vertex_config[arg_tuple].input_config[arg_]["axis"] = quantized_axis

                        vertex_config[arg_].output_config["quantized_scale"] = quantized_scale[
                            index_start : index_start + shape_value
                        ]
                        vertex_config[arg_].output_config["quantized_zero_point"] = numpy.zeros(
                            shape_value, dtype=numpy.int32
                        )
                        vertex_config[arg_].output_config["quantized_axis"] = quantized_axis

                        index_start = index_start + shape_value

                        recursive_set(arg_, vertex_config)

    elif (
        isinstance(arg, relay.TupleGetItem)
        and isinstance(arg.tuple_value, relay.Call)
        and not isinstance(arg.tuple_value.op, relay.Function)
        and arg_config.quantized
        and arg.tuple_value.op.name in ["nn.max_pool2d"]
        and vertex_config[arg.tuple_value.args[0]].output_config["ref_count"] == 1
        and "quantized_scale" in vertex_config[arg.tuple_value.args[0]].output_config
    ):
        quantized_scale = arg_config.output_config["quantized_scale"]
        quantized_axis = arg_config.output_config["quantized_axis"]

        LOGGER.debug("[calibrate]<recursive_set> scale start and the op is maxpool index is true")
        LOGGER.debug("[calibrate]<recursive_set> quantized scale is:")
        LOGGER.debug(quantized_scale)
        LOGGER.debug(
            "[calibrate]<recursive_set> quantized axis is %d",
            arg_config.output_config["quantized_axis"],
        )

        # setp1 tupleGetItemNode
        vertex_config[arg].output_config["scale"] = quantized_scale
        vertex_config[arg].output_config["zero_point"] = numpy.zeros_like(
            quantized_scale, dtype=numpy.int32
        )
        vertex_config[arg].output_config["axis"] = quantized_axis

        vertex_config[arg].input_config[arg.tuple_value]["scale"] = quantized_scale
        vertex_config[arg].input_config[arg.tuple_value]["zero_point"] = numpy.zeros_like(
            quantized_scale, dtype=numpy.int32
        )
        vertex_config[arg].input_config[arg.tuple_value]["axis"] = quantized_axis

        # step2 arg.tuple_value
        vertex_config[arg.tuple_value].input_config[arg.tuple_value.args[0]][
            "scale"
        ] = quantized_scale
        vertex_config[arg.tuple_value].input_config[arg.tuple_value.args[0]][
            "zero_point"
        ] = numpy.zeros_like(quantized_scale, dtype=numpy.int32)
        vertex_config[arg.tuple_value].input_config[arg.tuple_value.args[0]][
            "axis"
        ] = quantized_axis

        vertex_config[arg.tuple_value].output_config["scale"] = quantized_scale
        vertex_config[arg.tuple_value].output_config["zero_point"] = numpy.zeros_like(
            quantized_scale, dtype=numpy.int32
        )
        vertex_config[arg.tuple_value].output_config["axis"] = quantized_axis

        # step3 arg.tuple_value.args[0]
        vertex_config[arg.tuple_value.args[0]].output_config["quantized_scale"] = quantized_scale
        vertex_config[arg.tuple_value.args[0]].output_config[
            "quantized_zero_point"
        ] = numpy.zeros_like(quantized_scale, dtype=numpy.int32)
        vertex_config[arg.tuple_value.args[0]].output_config["quantized_axis"] = quantized_axis

        recursive_set(arg.tuple_value.args[0], vertex_config)


def _calibrate_core(arg, input_config, vertex_config, quantized=True):
    """calibrate core"""
    # if quantized is True, reshape + conv2d no need to requantize
    #    sigmoid + conv2d, sigmoid must have "quantized_scale"
    y = {}
    if input_config["method"] is not None:
        y = input_config["method"](input_config)
        if y["scale"].size > 1:
            y["scale"][numpy.where(y["scale"] == 0)] = 0.01 / 127
        tmp = {
            "quantized_scale": y["scale"].astype("float32"),
            "quantized_zero_point": y["zero_point"],
        }
        vertex_config[arg].output_config.update(tmp)
    else:
        LOGGER.debug("[calibrate] 'method' is none")
        if "quantized_scale" in vertex_config[arg].output_config:
            LOGGER.debug("[calibrate] use the existed quantized_scale")
            scale = vertex_config[arg].output_config["quantized_scale"]
            zero_point = vertex_config[arg].output_config["quantized_zero_point"]
            axis = vertex_config[arg].output_config["quantized_axis"]
            y = {"scale": scale.astype("float32"), "zero_point": zero_point, "axis": axis}
        elif vertex_config[arg].quantized:
            LOGGER.debug("[calibrate] use the existed scale")
            assert "scale" in vertex_config[arg].output_config
            scale = vertex_config[arg].output_config["scale"]
            zero_point = vertex_config[arg].output_config["zero_point"]
            axis = vertex_config[arg].output_config["axis"]
            y = {"scale": scale.astype("float32"), "zero_point": zero_point, "axis": axis}
    if (
        "scale" in y
        and isinstance(arg, relay.Var)
        and vertex_config[arg].output_config["net_in_dtype"] in ["uint8", "int16"]
    ):
        y["scale"] = numpy.ones(y["scale"].size)
    return y


def calibrate_params(cls):
    """calibrate_params"""
    vertex_config = cls.vertex_config
    idx = -1
    for call in vertex_config:
        config = vertex_config[call]
        if isinstance(call, relay.Call):
            idx = idx + 1
            LOGGER.info("[calibrate]...idx  %d ...", idx)

            # calibrate_cond
            calibrate_cond = False
            for arg in call.args:
                if config.input_config[arg]["method"] is not None:
                    calibrate_cond = True
                    break
            cond1 = any(vertex_config[one_arg].quantized for one_arg in call.args)
            if not calibrate_cond and (config.quantized or cond1):
                calibrate_cond = True

            if calibrate_cond:

                config.quantize_params(call, vertex_config)

                # recursive scale optimize
                if isinstance(call.args[0], (relay.Call, relay.TupleGetItem)):
                    recursive_set(call.args[0], vertex_config)

                if (
                    isinstance(call, relay.Call)
                    and not isinstance(call.op, relay.Function)
                    and len(call.args) == 2
                    and not isinstance(call.args[1], relay.TupleGetItem)
                ):
                    # TupleGetItem for maxpool idx1
                    recursive_set(call.args[1], vertex_config)

                # support final node int to float
                if (
                    "is_fn_body" in config.output_config
                    and config.output_config["is_fn_body"] is True
                    and config.output_config["method"] is not None
                ):
                    y = config.output_config["method"](config.output_config)
                    if y["scale"].size > 1:
                        y["scale"][numpy.where(y["scale"] == 0)] = 0.01 / 127
                    tmp = {
                        "quantized_scale": y["scale"],
                        "quantized_zero_point": y["zero_point"],
                        "quantized_axis": -1,
                    }
                    config.output_config.update(tmp)

        elif isinstance(call, relay.Tuple):
            idx = idx + 1
            LOGGER.info("[calibrate]...idx  %d tuplenode...", idx)
            for arg in call.fields:
                # if config.input_config[arg]["method"] is not None:
                input_config = config.input_config[arg]
                y = _calibrate_core(arg, input_config, vertex_config, config.quantized)

                input_config.update(y)
                # recursive scale optimize
                if isinstance(arg, relay.Call):
                    recursive_set(arg, vertex_config)

        elif isinstance(call, relay.TupleGetItem):
            # todo update
            if config.quantized:
                arg = call.tuple_value
                input_config = config.input_config[arg]
                scale = vertex_config[arg].output_config["scale"]
                zero_point = vertex_config[arg].output_config["zero_point"]
                y = {"scale": scale, "zero_point": zero_point}
                input_config.update(y)

                config.output_config.update(y)
