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
from tvm import te
from tvm import topi
from tvm import tir
from tvm.contrib.edgex.uniform_schedule.graph import *


def test_graph_interfaces():
    a = te.placeholder([1, 8, 224, 224], "int8", "a")
    b = te.placeholder([1, 8, 224, 224], "int8", "b")
    c = a + b
    d = a + 1
    e = b + 1
    f = c * d
    func = te.create_prim_func([a, b, c, d, e, f])
    s = tir.schedule.Schedule(func)
    blocks = s.get_child_blocks(s.get_block("root"))
    graph = BlockGraph.create(s)
    assert len(graph.data) == 4  # 4 blocks
    assert len(graph.inputs) == 4  # a->c, a->d, b->c, b->f
    assert len(graph.inputs_by_buffer) == 2
    assert len(graph.outputs) == 2  # ->e, ->f
    assert len(graph.outputs_by_buffer) == 2
    for idx, block in enumerate(graph.get_blocks_in_topo_order()):
        assert s.get_sref(blocks[idx]).same_as(block)
    assert len(graph.get_start_subgraphs()) == 3  # c, d, e
    assert len(graph.get_end_subgraphs()) == 2  # e, f


def test_merge_subgraphs():
    def get_casted_conv2d(
        name, x, kernel_shape, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1]
    ):
        weight = te.placeholder(kernel_shape, "int8", "w_" + name)
        conv = topi.nn.conv2d(
            x, weight, strides=strides, padding=padding, dilation=dilation, out_dtype="int32"
        )
        return topi.cast(conv, "int8")

    data = te.placeholder([1, 8, 224, 224], "int8", "x")
    conv1 = get_casted_conv2d("0", data, kernel_shape=[8, 8, 3, 3], padding=[1, 1, 1, 1])
    conv2 = get_casted_conv2d("1", conv1, kernel_shape=[8, 8, 3, 3], padding=[1, 1, 1, 1])
    conv3 = get_casted_conv2d("2", conv2, kernel_shape=[8, 8, 3, 3], padding=[1, 1, 1, 1])
    conv4 = get_casted_conv2d("3", conv3, kernel_shape=[8, 8, 3, 3], padding=[1, 1, 1, 1])
    add1 = conv4 + 1
    add2 = conv4 + 2
    add3 = conv4 + 3
    func = te.create_prim_func_from_outputs([add1, add2, add3])
    s = tir.schedule.Schedule(func)
    graph = BlockGraph.create(s)
    n0, n1, n2, n3 = graph[0:3], graph[3:6], graph[6:9], graph[9:12]
    BlockGraph.merge(*n0)
    m1 = BlockGraph.merge(*n1)
    m2 = BlockGraph.merge(*n2)
    BlockGraph.merge(*n3)
    m12 = BlockGraph.merge(m1, m2)
    m12.flatten()
    graph.print_partitions()
    graph.validate()


if __name__ == "__main__":
    test_graph_interfaces()
    test_merge_subgraphs()
