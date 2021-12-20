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
from acctest.model_zoo.utils import get_all_models
from acctest.testing.nnp400.load_model import get_relay_module
from acctest.testing.nnp400.inference import get_batch_process, evaluate
from tvm.relay.quantization.threshold import Threshold
from tvm.relay.quantization.method_dtype import Method
import tvm
from tvm import relay
import tvm.relay.quantization

numpy.random.seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if tvm.runtime.enabled("gpu"):
    ctx = tvm.cuda()
    target = "cuda"
else:
    ctx = tvm.cpu()
    target = "llvm"


SUPPORTED_MODELS = get_all_models()

batch_size = 1
calibrate_num = 50
model_name = "mobilenet_edgetpu"
root_path = os.path.join(os.path.expanduser("~"), "Documents/quantize_result")


if not os.path.exists("result/%s" % model_name):
    os.makedirs("result/%s" % model_name)

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
mean = batch_process.mean
scale = batch_process.scale
batch_process.norm_en = 1
batch_process.reset(random=True)

calibrate_data = []
for i in range(calibrate_num):
    batch_data, data_info = batch_process.preprocess()
    calibrate_data.append(batch_data)


def yield_calibrate_data():
    for i in calibrate_data:
        yield i


path = os.path.join(root_path, model_name, "origin_mod.json")
if os.path.exists(path):
    mod = None
    params = None
else:
    mod, params = get_relay_module(model_name, "tensorflow", "/data/share/demodels-lfs")

quantize_config = {}
quantize_config["call"] = {
    "threshold": Threshold.MinMax,
    "method": Method.Symmetry,
    "dtype": "int8",
}

quantize_config["skip_conv_layers"] = [i for i in range(0, 200000)]

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
    mean=None,
    scale=None,
    compare_statistics=False,
    net_in_dtype="float16",
    quantize_config=quantize_config,
)

config = quantize_search.get_default_config()
quantize_search.quantize(config)
quantize_search.evaluate("post_process", config)
