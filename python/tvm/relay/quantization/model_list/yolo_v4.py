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
import numpy
from collections import namedtuple
from acctest.testing.nnp400.load_model import get_relay_module
from acctest.testing.nnp400.inference import get_batch_process, evaluate
import tvm
from tvm import relay
import tvm.relay.quantization

numpy.random.seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ctx = tvm.cpu()
target = "llvm"

batch_size = 1
calibrate_num = 500
model_name = "yolov4-darknet"
performance = {"float": 64.088, "int8": None}
root_path = os.path.join(os.path.expanduser("~"), "Documents/quantize_result")

all_op = [
    "multiply",
    "nn.conv2d",
    "add",
    "exp",
    "log",
    "tanh",
    "nn.pad",
    "concatenate",
    "maximum",
    "nn.max_pool2d",
    "image.resize",
    "conv2d_bias_add",
    "reshape",
    "split",
    "sigmoid",
    "subtract",
]

Config = namedtuple(
    "Config",
    [
        "model_name",
        "test_mode",
        "profile_nums",
        "batch_size",
        "test_batch_nums",
        "device",
        "framework",
        "log_interval",
    ],
)

cfg = Config(
    model_name,
    test_mode="fp32",
    profile_nums=calibrate_num,
    batch_size=batch_size,
    test_batch_nums=99999999,
    device=target,
    framework="tensorflow",
    log_interval=1,
)

batch_process = get_batch_process(cfg, "/data/share/dedatasets-lfs/")
# batch_process = get_batch_process(cfg, "/home/yhh/Desktop/dedatasets-lfs/")
mean = batch_process.mean
scale = batch_process.scale
batch_process.norm_en = 0
batch_process.reset(random=True)

calibrate_data = []
for i in range(calibrate_num):
    batach_data, data_info = batch_process.preprocess()
    calibrate_data.append(batach_data)


def yield_calibrate_data():
    for i in calibrate_data:
        yield i


path = os.path.join(root_path, model_name, "origin_mod.json")
if os.path.exists(path):
    mod = None
    params = None
else:
    mod, params = get_relay_module(model_name, "tensorflow", "/data/share/demodels-lfs")
    # mod, params = get_relay_module(model_name, "tensorflow", "/home/yhh/Desktop/detvm/deepeye/demodels-lfs")

quantize_search = relay.quantization.QuantizeSearch(
    model_name=model_name,
    mod=mod,
    params=params,
    dataset=yield_calibrate_data,
    calibrate_num=calibrate_num,
    eval_func=evaluate(batch_process, cfg),
    ctx=ctx,
    target=target,
    root_path=root_path,
    mean=mean,
    scale=scale,
    compare_statistics=False,
)

config = quantize_search.get_default_config()
quantize_search.quantize(config)
# quantize_search.visualize("post_processed", config)
quantize_search.evaluate("post_process", config)
