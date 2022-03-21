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
# pylint: disable=unused-argument,inconsistent-return-statements,bad-continuation,len-as-condition
"""collect"""

import logging
import numpy
import tvm
from tvm import relay

LOGGER = logging.getLogger("quantize")


def _get_threshold_static(cls):
    for node in cls.vertex_config:
        vertex_config = cls.vertex_config[node]

        def tmp(arg, config):
            cond1 = isinstance(arg, relay.Constant)
            cond2 = config["operate"] == "quantize" or config["operate"] == "requantize"
            if cond1 and cond2:
                if config["threshold"] is not None:
                    x = cls.collect_result[arg]
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


def _get_statistics_min_max(cls, run):
    result = next(run)

    for node in cls.vertex_config:
        vertex_config = cls.vertex_config[node]

        def tmp(arg, config):
            """cal core"""
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
                quantized_axis = cls.vertex_config[arg].output_config["quantized_axis"]

                if quantized_axis != "none":
                    vertex_config.input_config[arg]["axis"] = quantized_axis

                if vertex_config.input_config[arg]["threshold"] is not None:
                    input_config = vertex_config.input_config[arg]
                    input_config["threshold"].update_axis(quantized_axis)
                    tmp(arg, input_config)

        elif isinstance(node, relay.TupleGetItem):
            arg = node.tuple_value
            input_config = vertex_config.input_config[arg]
            tmp(arg, input_config)

        elif isinstance(node, relay.Call):
            for arg in node.args:
                # use the quantized_axis and update the input_config['axis']
                quantized_axis = cls.vertex_config[arg].output_config["quantized_axis"]

                if quantized_axis != "none":
                    vertex_config.input_config[arg]["axis"] = quantized_axis

                if vertex_config.input_config[arg]["threshold"] is not None:
                    input_config = vertex_config.input_config[arg]
                    # LOGGER.debug("[collect] output_config quantized_axis is %d", quantized_axis)
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


def _get_threshold_dynamic(cls, run):
    result = next(run)
    for node in cls.vertex_config:
        vertex_config = cls.vertex_config[node]

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
                    # cls.vertex_config[arg].output_config['quantized_axis']
                    tmp(arg, input_config)

        elif isinstance(node, relay.TupleGetItem):
            arg = node.tuple_value
            input_config = vertex_config.input_config[arg]
            tmp(arg, input_config)

        elif isinstance(node, relay.Call):
            if not isinstance(node.op, relay.Function):
                pass  # LOGGER.debug("[collect] {} do kld step2".format(node.op.name))
            for arg in node.args:
                if vertex_config.input_config[arg]["threshold"] is not None:
                    input_config = vertex_config.input_config[arg]
                    # notice: use the quantized_axis and update the input_config['axis']
                    input_config["axis"] = cls.vertex_config[arg].output_config["quantized_axis"]
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
            yield result
        except StopIteration:
            break


def _get_node_runtime(nodes, params, ctx, target):
    func = relay.Function(params, relay.Tuple(nodes))
    input_keys = [str(param.name_hint) for param in func.params]
    # compatible with nnp300
    if "transform" in relay.__dict__:
        try:
            with relay.transform.build_config(opt_level=3):
                graph, lib, params = relay.build_module.build(func, target=target)
        except BaseException:
            LOGGER.info("[collect] build_config use opt_level 2")
            with relay.transform.build_config(opt_level=2):
                graph, lib, params = relay.build_module.build(func, target=target)
        runtime = tvm.contrib.graph_executor.create(graph, lib, ctx)
    else:
        with relay.build_config(opt_level=2):
            graph, lib, params = relay.build(func, target="llvm")
        runtime = tvm.contrib.graph_runtime.create(graph, lib, ctx)

    runtime.set_input(**params)
    num_outputs = runtime.get_num_outputs()
    return runtime, input_keys, num_outputs


def collect_stats(cls):
    """collect_stats"""
    _get_threshold_static(cls)
    nodes = list(cls.collect_node)
    if nodes != []:
        if not isinstance(cls.pre_processed_mod, relay.Function):
            params = cls.pre_processed_mod["main"].params
        else:
            params = cls.pre_processed_mod.params
        runtime, input_keys, num_outputs = _get_node_runtime(nodes, params, cls.ctx, cls.target)

        LOGGER.debug("[collect] collect_stats nodes len is:")
        LOGGER.debug(len(nodes))
        LOGGER.debug("[collect] collect_stats num_outputs is:")
        LOGGER.debug(num_outputs)

        dataset = cls.dataset()
        run = _run_graph(dataset, nodes, runtime, input_keys, num_outputs)
        dataset_idx = -1
        while True:
            try:
                dataset_idx = dataset_idx + 1
                if dataset_idx % 10 == 0:
                    LOGGER.info("[collect] statistics_min_max now finish img index %d", dataset_idx)
                _get_statistics_min_max(cls, run)
            except StopIteration:
                break

        dataset = cls.dataset()
        run = _run_graph(dataset, nodes, runtime, input_keys, num_outputs)
        dataset_idx = -1
        while True:
            try:
                dataset_idx = dataset_idx + 1
                if dataset_idx % 10 == 0:
                    LOGGER.info(
                        "[collect] get_threshold_dynamic now finish img index %d", dataset_idx
                    )
                _get_threshold_dynamic(cls, run)
            except StopIteration:
                break
