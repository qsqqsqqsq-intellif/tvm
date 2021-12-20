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
"""debug"""

import numpy
import tvm
from tvm import relay

# import matplotlib.pyplot as plt


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


def _get_graph(nodes, params, ctx, target, optlevel=3):
    func = relay.Function(params, relay.Tuple(nodes))
    input_keys = [str(param.name_hint) for param in func.params]
    with relay.transform.build_config(opt_level=optlevel):
        graph, lib, params = relay.build_module.build(func, target=target)
    runtime = tvm.contrib.graph_executor.create(graph, lib, ctx)
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


def data_hist_analysis(data, number):
    """data hist anaylsis"""
    data_max = numpy.max(numpy.abs(data))
    width = data_max / (number - 1)
    hist = numpy.zeros(number)
    if data_max == 0:
        return hist
    count = len(data)
    for i in range(count):
        index = (int)(numpy.abs(data[i]) / width + 0.5)
        hist[index] += 1
    return hist


# def analysis_two_data(data1_, data2_, name1="fixed", name2="float"):
#    data1 = data1_.flatten()
#    data1_list = list(data1)
#    data1_hist = data_hist_analysis(data1_list, 2048)

#    data2 = data2_.flatten()
#    data2_list = list(data2)
#    data2_hist = data_hist_analysis(data2_list, 2048)

#    ax1 = plt.subplot(221)
#    plt.plot(data1_hist, "*")
#    ax2 = plt.subplot(223)
#    plt.plot(data2_hist, "*")

#    ax3 = plt.subplot(222)
#    plt.plot(data1_list)
#    plt.xticks(rotation=90)
#    ax4 = plt.subplot(224)
#    plt.plot(data2_list)
#    plt.xticks(rotation=90)

#    ax1.set_title("total = " + str(len(data1_list)))
#    ax2.set_title("total = " + str(len(data2_list)))
#    len_name1 = len(name1)
#    len_name2 = len(name2)
#    ax3.set_title("float*" + name1[len_name1 - 10 :])
#    ax4.set_title("fixed*" + name2[len_name2 - 10 :])
#    plt.show()


def compare_statistics(cls, method):
    """compare_statistics"""
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

    old_p = cls.pre_processed_mod["main"].params
    new_p = cls.post_processed_mod["main"].params
    old_r, old_ik, old_no = _get_graph(old_node, old_p, cls.ctx, cls.target, cls.opt_level)
    new_r, new_ik, new_no = _get_graph(new_node, new_p, cls.ctx, cls.target, cls.opt_level)

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
                # analysis_two_data(o_r, n_r, name[0:10], name[0:10])
                one_result.append([name, distance])
                print("{x:<30}{y:<50}{z:<40}".format(x=name, y=dtype, z=distance))
            all_result.append(one_result)
            print("一张图片统计对比结束\n")
            # break
        except StopIteration:
            break
    return all_result
