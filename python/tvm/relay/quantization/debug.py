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
# pylint: disable=unused-argument,inconsistent-return-statements,len-as-condition
"""debug"""

import os
import numpy
import matplotlib.pyplot as plt
import tvm
from tvm import relay


def norm_0(m1_, m2_):
    m = m1_ - m2_
    m = m.reshape(-1)
    distance = numpy.linalg.norm(m, 0)
    return distance


def norm_1(m1_, m2_):
    m = m1_ - m2_
    m = m.reshape(-1)
    distance = numpy.linalg.norm(m, 1)
    return distance


def norm_2(m1_, m2_):
    m = m1_ - m2_
    m = m.reshape(-1)
    distance = numpy.linalg.norm(m, 2)
    return distance


def norm_infinity(m1_, m2_):
    m = m1_ - m2_
    m = m.reshape(-1)
    distance = numpy.linalg.norm(m, numpy.inf)
    return distance


def cosine(m1_, m2_):
    m1_ = m1_.reshape(-1).astype("float64")
    m2_ = m2_.reshape(-1).astype("float64")
    dot = numpy.dot(m1_, m2_)
    length = numpy.linalg.norm(m1_, 2) * numpy.linalg.norm(m2_, 2)
    if length == 0:
        distance = numpy.inf
    else:
        distance = dot / length
    return distance


DISTANCE = {
    "norm_0": norm_0,
    "norm_1": norm_1,
    "norm_2": norm_2,
    "norm_infinity": norm_infinity,
    "cosine": cosine,
}


def _run_graph(batch, runtime, input_keys, num_outputs):
    for key in input_keys:
        runtime.set_input(key, batch[key])
    runtime.run()
    outputs = []
    for i in range(num_outputs):
        output = runtime.get_output(i).asnumpy()
        if len(output.shape) == 0:
            output = numpy.array([output])
        outputs.append(output)
    return outputs


def _get_graph(nodes, ctx, target, optlevel=3):

    if "analysis" in relay.__dict__:
        new_params = relay.analysis.free_vars(relay.Tuple(nodes))
    else:
        new_params = relay.ir_pass.free_vars(relay.Tuple(nodes))

    func = relay.Function(new_params, relay.Tuple(nodes))

    input_keys = [str(param.name_hint) for param in func.params]
    if "transform" in relay.__dict__:
        with relay.transform.build_config(opt_level=optlevel):
            graph, lib, params = relay.build_module.build(func, target=target)
        runtime = tvm.contrib.graph_executor.create(graph, lib, ctx)
    else:
        with relay.build_config(opt_level=2):
            graph, lib, params = relay.build(func, target=target)
        runtime = tvm.contrib.graph_runtime.create(graph, lib, ctx)
    runtime.set_input(**params)
    num_outputs = runtime.get_num_outputs()
    return runtime, input_keys, num_outputs


def pair_node(old_node, new_node, oc_, ic_, n2o, quantized):
    """pair_node"""
    if new_node not in n2o:
        n2o[new_node] = {"node": old_node}
        if ic_["operate"] == "quantize":
            assert ic_.get("scale") is not None
            assert ic_.get("axis") is not None
            scale = ic_.get("scale")
            axis = ic_.get("axis")
            n2o[new_node].update({"scale": scale, "axis": axis})
        elif ic_["operate"] == "dequantize":
            pass
        elif ic_["operate"] == "requantize":
            assert ic_.get("scale") is not None
            assert ic_.get("axis") is not None
            scale = ic_.get("scale")
            axis = ic_.get("axis")
            n2o[new_node].update({"scale": scale, "axis": axis})
        elif ic_["operate"] == "none":
            if quantized:
                assert oc_.get("scale") is not None
                assert oc_.get("axis") is not None
                scale = oc_.get("scale")
                axis = oc_.get("axis")
                n2o[new_node].update({"scale": scale, "axis": axis})
            else:
                pass


def plot_statistics(data1_, data2_, distance, name, path):
    """plot_statistics"""
    data1 = data1_.reshape(-1)
    data2 = data2_.reshape(-1)
    abs1 = numpy.abs(data1)
    abs2 = numpy.abs(data2)
    hist1 = numpy.histogram(abs1, 2048, (0, abs1.max()))[0]
    hist2 = numpy.histogram(abs2, 2048, (0, abs2.max()))[0]

    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.plot(hist1, "*")
    plt.subplot(223)
    plt.plot(hist2, "*")

    ax3 = plt.subplot(222)
    plt.plot(data1)
    plt.xticks(rotation=45)
    ax4 = plt.subplot(224)
    plt.plot(data2)
    plt.xticks(rotation=45)

    # ax1.set_title("total = " + str(len(data1)))
    # ax2.set_title("total = " + str(len(data2)))
    ax3.set_title("origin")
    ax4.set_title("quantized")
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.suptitle(name + " similarity : %s" % distance)
    if path is not None:
        plt.savefig(os.path.join(path, name) + ".png")
    else:
        plt.show()
    plt.close()


def compare_statistics(cls, method):
    """compare_statistics"""
    if cls.save_path:
        statistics_path = os.path.join(cls.save_path, "statistics")
        if not os.path.exists(statistics_path):
            os.makedirs(statistics_path)
    else:
        statistics_path = None

    old_node = []
    new_node = []
    new_scale = []
    for new in cls.new2old:
        if isinstance(cls.new2old[new], relay.Call):
            continue
        old = cls.new2old[new]["node"]
        # if isinstance(old, relay.Constant):
        #     continue
        if old not in old_node:
            old_node.append(old)
            new_node.append(new)
            tmp = {}
            if cls.new2old[new].get("scale") is not None:
                scale = cls.new2old[new].get("scale")
                axis = cls.new2old[new].get("axis")
                tmp.update({"scale": scale, "axis": axis})
            new_scale.append(tmp)

    old_r, old_ik, old_no = _get_graph(old_node, cls.ctx, cls.target, cls.opt_level)
    new_r, new_ik, new_no = _get_graph(new_node, cls.ctx, cls.target, cls.opt_level)

    assert len(old_node) == len(new_node) == old_no == new_no

    def _int2float(result, config):
        new_result = []
        for res, conf in zip(result, config):
            if conf.get("scale") is not None:
                scale = conf.get("scale")
                axis = conf.get("axis")
                if axis != -1 and scale.size != 1:
                    tmp1 = numpy.ones_like(res.shape)
                    tmp1[axis] = res.shape[axis]
                    scale = scale.reshape(tmp1)
                res = res * scale
            new_result.append(res)
        return new_result

    dataset = cls.dataset()
    all_result = []
    while True:
        try:
            batch = next(dataset)
            old_result = _run_graph(batch, old_r, old_ik, old_no)
            new_result = _run_graph(batch, new_r, new_ik, new_no)
            float_new_result = _int2float(new_result, new_scale)
            assert len(old_result) == len(float_new_result)

            one_result = []
            for i, (o_r, n_r) in enumerate(zip(old_result, float_new_result)):
                distance = DISTANCE[method](o_r, n_r)
                tmp = cls.new2old[new_node[i]]["node"]
                name = cls.node_id[tmp]
                dtype = str(old_node[i].checked_type)
                plot_statistics(o_r, n_r, distance, name, statistics_path)
                one_result.append([name, distance])
                print("{x:<30}{y:<50}{z:<40}".format(x=name, y=dtype, z=distance))
            all_result.append(one_result)
            print("一张图片统计对比结束\n")
        except StopIteration:
            break
    return all_result


def compare_statistics_api(cls, method, display_en, path):
    """compare_statistics for custom"""
    old_node = []
    new_node = []
    new_scale = []
    for new in cls.new2old:
        if isinstance(cls.new2old[new], relay.Call):
            continue
        old = cls.new2old[new]["node"]
        if isinstance(old, relay.Constant):
            continue
        if old not in old_node:
            old_node.append(old)
            new_node.append(new)
            tmp = {}
            if cls.new2old[new].get("scale") is not None:
                scale = cls.new2old[new].get("scale")
                axis = cls.new2old[new].get("axis")
                tmp.update({"scale": scale, "axis": axis})
            new_scale.append(tmp)

    old_r, old_ik, old_no = _get_graph(old_node, cls.ctx, cls.target, cls.opt_level)
    new_r, new_ik, new_no = _get_graph(new_node, cls.ctx, cls.target, cls.opt_level)

    assert len(old_node) == len(new_node) == old_no == new_no

    def _int2float(result, config):
        new_result = []
        for res, conf in zip(result, config):
            if conf.get("scale") is not None:
                scale = conf.get("scale")
                axis = conf.get("axis")
                if axis != -1 and scale.size != 1:
                    tmp1 = numpy.ones_like(res.shape)
                    tmp1[axis] = res.shape[axis]
                    scale = scale.reshape(tmp1)
                res = res * scale
            new_result.append(res)
        return new_result

    dataset = cls.dataset()
    all_result = []
    while True:
        try:
            batch = next(dataset)
            old_result = _run_graph(batch, old_r, old_ik, old_no)
            new_result = _run_graph(batch, new_r, new_ik, new_no)
            float_new_result = _int2float(new_result, new_scale)
            assert len(old_result) == len(float_new_result)

            one_result = []
            for i, (o_r, n_r) in enumerate(zip(old_result, float_new_result)):
                distance = DISTANCE[method](o_r, n_r)
                tmp = cls.new2old[new_node[i]]["node"]
                name = cls.node_id[tmp]
                dtype = str(old_node[i].checked_type.shape)
                # plot_statistics(o_r, n_r, distance, name, path)
                one_result.append([name, distance])
                print("{x:<30}{y:<50}{z:<40}".format(x=name, y=dtype, z=distance))
            all_result.append(one_result)
            print("一张图片统计对比结束\n")
        except StopIteration:
            break
    return all_result
