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
"""collect"""

import logging
import numpy
import tvm
from tvm import relay

LOGGER = logging.getLogger("quantize")


def _get_threshold_static(quantize):
    for node in quantize.vertex_config:
        vertex_config = quantize.vertex_config[node]

        def tmp(arg, config):
            cond1 = isinstance(arg, relay.Constant)
            cond2 = config["operate"] == "quantize" or config["operate"] == "requantize"
            if cond1 and cond2:
                if config["threshold"] is not None:
                    x = quantize.collect_result[arg]
                    config["threshold"].run(x)

        if vertex_config.quantized:
            if isinstance(node, relay.Tuple):
                for arg in node.fields:
                    input_config = vertex_config.input_config[arg]
                    tmp(arg, input_config)
            elif isinstance(node, relay.TupleGetItem):
                arg = node.tuple_value
                input_config = vertex_config.input_config[arg]
                tmp(arg, input_config)
            else:
                for arg in node.args:
                    input_config = vertex_config.input_config[arg]
                    tmp(arg, input_config)


def _get_statistics_min_max(quantize, run):
    result = next(run)

    for node in quantize.vertex_config:
        vertex_config = quantize.vertex_config[node]

        def tmp(arg, config):
            cond1 = not isinstance(arg, relay.Constant)
            cond2 = config["operate"] == "quantize" or config["operate"] == "requantize"
            if cond1 and cond2:
                if config["threshold"] is not None:
                    # logging.debug("[collect] statis_min_max arg shape is")
                    # logging.debug(result[arg].shape)
                    x = result[arg]
                    # update the axis according to input_config!!
                    config["threshold"].axis = config["axis"]
                    config["threshold"].statistics_min_max(x)

        if isinstance(node, relay.Tuple):
            # LOGGER.debug("[collect] Tuple")
            for arg in node.fields:
                # use the quantized_axis and update the input_config['axis']
                quantized_axis = quantize.vertex_config[arg].output_config["quantized_axis"]
                update_threshold = False
                if quantized_axis not in ("none", vertex_config.input_config[arg]["axis"]):
                    update_threshold = True

                if quantized_axis != "none":
                    vertex_config.input_config[arg]["axis"] = quantized_axis

                if vertex_config.input_config[arg]["threshold"] is not None:
                    input_config = vertex_config.input_config[arg]
                    # update: if the axis change, some parameters like histogram may change
                    if update_threshold:
                        input_config["threshold"].update_axis(quantized_axis)
                    tmp(arg, input_config)

        elif isinstance(node, relay.TupleGetItem):
            arg = node.tuple_value
            input_config = vertex_config.input_config[arg]
            tmp(arg, input_config)

        elif isinstance(node, relay.Call):
            # use the quantized_axis and update the input_config['axis']
            for arg in node.args:
                # use the quantized_axis and update the input_config['axis']
                quantized_axis = quantize.vertex_config[arg].output_config["quantized_axis"]
                update_threshold = False
                if quantized_axis not in ("none", vertex_config.input_config[arg]["axis"]):
                    update_threshold = True

                if quantized_axis != "none":
                    vertex_config.input_config[arg]["axis"] = quantized_axis

                if vertex_config.input_config[arg]["threshold"] is not None:
                    input_config = vertex_config.input_config[arg]
                    # LOGGER.debug("[collect] output_config quantized_axis is %d", quantized_axis)
                    if update_threshold:
                        input_config["threshold"].update_axis(quantized_axis)

                    tmp(arg, input_config)

            # todo consider tuple output!!
            if (
                "is_fn_body" in vertex_config.output_config
                and vertex_config.output_config["is_fn_body"] is True
            ):
                if vertex_config.output_config["threshold"] is not None:
                    x = result[node]
                    vertex_config.output_config["threshold"].statistics_min_max(x)


def _get_threshold_dynamic(quantize, run):
    result = next(run)
    for node in quantize.vertex_config:
        vertex_config = quantize.vertex_config[node]

        def tmp(arg, config):
            cond1 = not isinstance(arg, relay.Constant)
            cond2 = config["operate"] == "quantize" or config["operate"] == "requantize"
            if cond1 and cond2:
                if config["threshold"] is not None:
                    x = result[arg]
                    # notice: update the axis according to input_config!!
                    config["threshold"].axis = config["axis"]
                    config["threshold"].run(x)

        if isinstance(node, relay.Tuple):
            for arg in node.fields:
                if vertex_config.input_config[arg]["threshold"] is not None:
                    input_config = vertex_config.input_config[arg]
                    # axis alreay set by min_max
                    # input_config['axis'] =
                    # quantize.vertex_config[arg].output_config['quantized_axis']
                    tmp(arg, input_config)

        elif isinstance(node, relay.TupleGetItem):
            arg = node.tuple_value
            # todo
            input_config = vertex_config.input_config[arg]
            tmp(arg, input_config)
        elif isinstance(node, relay.Call):
            if not isinstance(node.op, relay.Function):
                pass  # LOGGER.debug("[collect] {} do kld step2".format(node.op.name))
            for arg in node.args:
                if vertex_config.input_config[arg]["threshold"] is not None:
                    input_config = vertex_config.input_config[arg]
                    # notice: use the quantized_axis and update the input_config['axis']
                    input_config["axis"] = quantize.vertex_config[arg].output_config[
                        "quantized_axis"
                    ]
                    tmp(arg, input_config)

            if (
                "is_fn_body" in vertex_config.output_config
                and vertex_config.output_config["is_fn_body"] is True
            ):
                if vertex_config.output_config["threshold"] is not None:
                    x = result[node]
                    vertex_config.output_config["threshold"].run(x)


def _run_graph(dataset, nodes, runtime, input_keys, num_outputs):
    while True:
        try:
            batch = next(dataset)
            for key in input_keys:
                runtime.set_input(key, batch[key])
            runtime.run()
            outputs = []
            for i in range(num_outputs):
                output = runtime.get_output(i).asnumpy()
                if len(output.shape) == 0:
                    output = numpy.array([output])
                outputs.append(output)
            result = dict(zip(nodes, outputs))
            # for k,v in result.items():
            #     print("key is ", k)
            #     print("value shape is ", v.shape)
            #     if tuple(relay.frontend.common.infer_type(k).checked_type.shape) != v.shape:
            #         print("....")
            yield result
        except StopIteration:
            break


def _get_node_runtime(nodes, params, ctx, target):
    func = relay.Function(params, relay.Tuple(nodes))
    # print(relay.frontend.common.infer_type(func))
    input_keys = [str(param.name_hint) for param in func.params]
    with relay.transform.build_config(opt_level=3):
        graph, lib, params = relay.build_module.build(func, target=target)
    runtime = tvm.contrib.graph_executor.create(graph, lib, ctx)
    runtime.set_input(**params)
    num_outputs = runtime.get_num_outputs()
    return runtime, input_keys, num_outputs


def collect_stats(quantize):
    """collect_stats"""
    _get_threshold_static(quantize)
    nodes = list(quantize.collect_node)
    params = quantize.pre_processed_mod["main"].params
    runtime, input_keys, num_outputs = _get_node_runtime(
        nodes, params, quantize.ctx, quantize.target
    )

    LOGGER.debug("[collect] collect_stats nodes len is:")
    LOGGER.debug(len(nodes))
    LOGGER.debug("[collect] collect_stats num_outputs is:")
    LOGGER.debug(num_outputs)

    dataset = quantize.dataset()
    run = _run_graph(dataset, nodes, runtime, input_keys, num_outputs)
    dataset_idx = -1
    while True:
        try:
            dataset_idx = dataset_idx + 1
            if dataset_idx % 10 == 0:
                LOGGER.info("[collect] statistics_min_max now finish img index %d", dataset_idx)
            _get_statistics_min_max(quantize, run)
        except StopIteration:
            break

    dataset = quantize.dataset()
    run = _run_graph(dataset, nodes, runtime, input_keys, num_outputs)
    dataset_idx = -1
    while True:
        try:
            dataset_idx = dataset_idx + 1
            if dataset_idx % 10 == 0:
                LOGGER.info("[collect] get_threshold_dynamic now finish img index %d", dataset_idx)
            _get_threshold_dynamic(quantize, run)
        except StopIteration:
            break
