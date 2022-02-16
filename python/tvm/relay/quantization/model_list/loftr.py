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

import os
import tqdm
import numpy
import torch
import tvm
from tvm import relay
import tvm.relay.quantization

torch.manual_seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ctx = tvm.cpu()
target = "llvm"

batch_size = 1
calibrate_num = 1
num_workers = 16
model_name = "loftr"
performance = {"float": None, "int8": None}
root_path = os.path.join(os.path.expanduser("~"), "Documents/quantize_result")

all_op = [
    "concatenate",
    "conv2d_bias_add",
    "nn.relu",
    "add",
    "nn.conv2d",
    "split",
    "reshape",
    "transpose",
    "nn.dense",
    "exp",
    "subtract",
    "multiply",
    "nn.batch_matmul",
    "sum",
    "divide",
    "expand_dims",
    "nn.layer_norm",
]

calibrate_data = []
for i in range(calibrate_num):
    video1 = numpy.random.randint(0, 256, [1, 1, 480, 640], numpy.uint8)
    video2 = numpy.random.randint(0, 256, [1, 1, 480, 640], numpy.uint8)
    calibrate_data.append({"input0": video1, "input1": video2})


def yield_calibrate_data():
    for i in calibrate_data:
        yield i


def evaluate(runtime):
    t = tqdm.tqdm(calibrate_data)
    for video1, video2 in t:
        video1 = numpy.clip(video1.numpy(), 0, 255).round().astype(numpy.uint8)
        video2 = numpy.clip(video2.numpy(), 0, 255).round().astype(numpy.uint8)
        data = {"input0": video1, "input1": video2}
        runtime.set_input(**data)
        runtime.run()
        output = runtime.get_output(0).asnumpy()
    return acc


path = os.path.join(root_path, model_name, "origin_mod.json")
if os.path.exists(path):
    mod = None
    params = None
else:
    scripted_model = torch.jit.load(
        "/home/zhaojinxi/Documents/quantize_result/loftr/indoor_ds_script_sub.pt"
    )
    shape_list = [("input0", [1, 1, 480, 640]), ("input1", [1, 1, 480, 640])]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

quantize_search = relay.quantization.QuantizeSearch(
    model_name=model_name,
    mod=mod,
    params=params,
    dataset=yield_calibrate_data,
    calibrate_num=calibrate_num,
    eval_func=evaluate,
    ctx=ctx,
    target=target,
    root_path=root_path,
    norm={
        "input0": {"mean": None, "std": 255, "axis": 1},
        "input1": {"mean": None, "std": 255, "axis": 1},
    },
    compare_statistics=True,
)

config = quantize_search.get_default_config()
quantize_search.quantize(config)
assert quantize_search.results[0]["other"]["similarity"][0][-1][1] >= 0.99
