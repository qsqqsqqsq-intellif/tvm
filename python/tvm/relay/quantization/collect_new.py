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

import os
import shutil
import logging
import numpy
import tqdm
import tvm
from tvm import relay

LOGGER = logging.getLogger("quantize")


def _get_threshold_static(cls):
    for node in tqdm.tqdm(cls.vertex_config):
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


def _get_statistics_min_max(cls, collect_path, nodes):
    def _caculate(arg, config):
        """cal core"""
        tmp_path = result[arg]
        tmp_path = os.path.join(collect_path, tmp_path)
        listdir = os.listdir(tmp_path)
        index = [int(i[6:-4]) for i in listdir]
        listdir = dict(zip(index, listdir))
        tmp1 = len(listdir) // cls.calibrate_batch
        tmp2 = len(listdir) % cls.calibrate_batch
        tmp3 = [cls.calibrate_batch] * tmp1
        if tmp2 != 0:
            tmp3 = tmp3 + [tmp2]
        start = 1
        tmp7 = []
        for tmp4 in tmp3:
            tmp8 = []
            for tmp5 in range(start, tmp4 + start):
                tmp8.append(listdir[tmp5])
            tmp7.append(tmp8)
            start = start + tmp4
        for tmp9 in tmp7:
            x = []
            for i in tmp9:
                data = numpy.load(os.path.join(tmp_path, i))
                x.append(data)
            x = numpy.concatenate(x, 0)
            config["threshold"].statistics_min_max(x)

    listdir1 = os.listdir(collect_path)
    index = [int(i[5:]) for i in listdir1]
    listdir1 = dict(zip(index, listdir1))
    result = {}
    for i in range(len(listdir1)):
        result[nodes[i]] = listdir1[i + 1]

    for node in tqdm.tqdm(cls.vertex_config):
        vertex_config = cls.vertex_config[node]

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
                    cond1 = not isinstance(arg, relay.Constant)
                    cond2 = (
                        input_config["operate"] == "quantize"
                        or input_config["operate"] == "requantize"
                    )
                    if cond1 and cond2:
                        # if input_config["threshold"] is not None:
                        input_config["threshold"].axis = input_config["axis"]
                        _caculate(arg, input_config)

        elif isinstance(node, relay.TupleGetItem):
            arg = node.tuple_value
            input_config = vertex_config.input_config[arg]
            cond1 = not isinstance(arg, relay.Constant)
            cond2 = input_config["operate"] == "quantize" or input_config["operate"] == "requantize"
            if cond1 and cond2:
                if input_config["threshold"] is not None:
                    input_config["threshold"].axis = input_config["axis"]
                    _caculate(arg, input_config)

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
                    cond1 = not isinstance(arg, relay.Constant)
                    cond2 = (
                        input_config["operate"] == "quantize"
                        or input_config["operate"] == "requantize"
                    )
                    if cond1 and cond2:
                        input_config["threshold"].axis = input_config["axis"]
                        _caculate(arg, input_config)

            # todo consider tuple output!!
            if (
                "is_fn_body" in vertex_config.output_config
                and vertex_config.output_config["is_fn_body"] is True
            ):
                if vertex_config.output_config["threshold"] is not None:
                    _caculate(node, vertex_config.output_config)


def _get_threshold_dynamic(cls, collect_path, nodes):
    def _calculate(arg, config):
        tmp_path = result[arg]
        tmp_path = os.path.join(collect_path, tmp_path)
        listdir = os.listdir(tmp_path)
        index = [int(i[6:-4]) for i in listdir]
        listdir = dict(zip(index, listdir))
        tmp1 = len(listdir) // cls.calibrate_batch
        tmp2 = len(listdir) % cls.calibrate_batch
        tmp3 = [cls.calibrate_batch] * tmp1
        if tmp2 != 0:
            tmp3 = tmp3 + [tmp2]
        start = 1
        tmp7 = []
        for tmp4 in tmp3:
            tmp8 = []
            for tmp5 in range(start, tmp4 + start):
                tmp6 = listdir[tmp5]
                tmp8.append(tmp6)
            tmp7.append(tmp8)
            start = start + tmp4
        for tmp9 in tmp7:
            x = []
            for i in tmp9:
                data = numpy.load(os.path.join(tmp_path, i))
                x.append(data)
            x = numpy.concatenate(x, 0)
            config["threshold"].run(x)

    listdir1 = os.listdir(collect_path)
    index = [int(i[5:]) for i in listdir1]
    listdir1 = dict(zip(index, listdir1))
    result = {}
    for i in range(len(listdir1)):
        result[nodes[i]] = listdir1[i + 1]

    for node in tqdm.tqdm(cls.vertex_config):
        vertex_config = cls.vertex_config[node]

        if isinstance(node, relay.Tuple):
            for arg in node.fields:
                if vertex_config.input_config[arg]["threshold"] is not None:
                    input_config = vertex_config.input_config[arg]
                    # axis alreay set by min_max
                    cond1 = not isinstance(arg, relay.Constant)
                    cond2 = (
                        input_config["operate"] == "quantize"
                        or input_config["operate"] == "requantize"
                    )
                    if cond1 and cond2:
                        input_config["threshold"].axis = input_config["axis"]
                        _calculate(arg, input_config)

        elif isinstance(node, relay.TupleGetItem):
            arg = node.tuple_value
            input_config = vertex_config.input_config[arg]
            cond1 = not isinstance(arg, relay.Constant)
            cond2 = input_config["operate"] == "quantize" or input_config["operate"] == "requantize"
            if cond1 and cond2:
                if input_config["threshold"] is not None:
                    input_config["threshold"].axis = input_config["axis"]
                    _calculate(arg, input_config)

        elif isinstance(node, relay.Call):
            if not isinstance(node.op, relay.Function):
                pass  # LOGGER.debug("[collect] {} do kld step2".format(node.op.name))
            for arg in node.args:
                if vertex_config.input_config[arg]["threshold"] is not None:
                    input_config = vertex_config.input_config[arg]
                    input_config["axis"] = cls.vertex_config[arg].output_config["quantized_axis"]
                    cond1 = not isinstance(arg, relay.Constant)
                    cond2 = (
                        input_config["operate"] == "quantize"
                        or input_config["operate"] == "requantize"
                    )
                    if cond1 and cond2:
                        input_config["threshold"].axis = input_config["axis"]
                        _calculate(arg, input_config)

            if (
                "is_fn_body" in vertex_config.output_config
                and vertex_config.output_config["is_fn_body"] is True
            ):
                if vertex_config.output_config["threshold"] is not None:
                    _calculate(node, vertex_config.output_config)


def _run_graph(dataset, runtime, input_keys, num_outputs, collect_path, calibrate_num):
    node_paths = []
    for node_index in range(1, num_outputs + 1):
        node_path = os.path.join(collect_path, "node_%s" % node_index)
        if not os.path.exists(node_path):
            os.makedirs(node_path)
        node_paths.append(node_path)

    for batch_index in tqdm.tqdm(range(1, calibrate_num + 1)):
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

        for node_path, output in zip(node_paths, outputs):
            batch_path = os.path.join(node_path, "batch_%s" % batch_index)
            numpy.save(batch_path, output)


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
    LOGGER.debug("[collect] calculate weight threshold")
    _get_threshold_static(cls)

    nodes = list(cls.collect_node)
    if nodes != []:
        collect_path = os.path.join(cls.save_path, "collect")
        if not os.path.exists(collect_path):
            os.makedirs(collect_path)

        if not isinstance(cls.pre_processed_mod, relay.Function):
            params = cls.pre_processed_mod["main"].params
        else:
            params = cls.pre_processed_mod.params
        runtime, input_keys, num_outputs = _get_node_runtime(nodes, params, cls.ctx, cls.target)

        dataset = cls.dataset()
        LOGGER.debug("[collect] collect running data")
        _run_graph(dataset, runtime, input_keys, num_outputs, collect_path, cls.calibrate_num)

        LOGGER.debug("[collect] calculate min max")
        _get_statistics_min_max(cls, collect_path, nodes)

        LOGGER.debug("[collect] calculate dataflow threshold")
        _get_threshold_dynamic(cls, collect_path, nodes)

        shutil.rmtree(collect_path)
