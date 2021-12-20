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
from collections import namedtuple
from acctest.testing.nnp400.load_model import get_relay_module
from acctest.testing.nnp400.inference import get_batch_process, evaluate
import tvm
from tvm import relay

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if tvm.runtime.enabled("gpu"):
    ctx = tvm.gpu()
    target = "cuda"
else:
    ctx = tvm.cpu()
    target = "llvm"

result = {
    "InceptionV4": {"float": 80.142, "int8": 78.534},
    "yolov3": {"float": None, "int8": None},
}

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

batch_size = 1
calibrate_num = 500
model_name = "InceptionV4"

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
batch_process.reset(random=True)

calibrate_data = []
for i in range(calibrate_num):
    batach_data, data_info = batch_process.preprocess()
    calibrate_data.append(batach_data)


def yield_calibrate_data():
    for i in calibrate_data:
        yield i


yield_calibrate_data.calibrate_num = calibrate_num

mod, params = get_relay_module(model_name, "tensorflow", "/data/share/demodels-lfs")
quantize_search = relay.quantization.QuantizeSearch(
    model_name,
    mod,
    params,
    yield_calibrate_data,
    evaluate(batch_process, cfg),
    ctx=ctx,
    target=target,
)

config = quantize_search.get_default_config()
quantize_search.quantize(config)
quantize_search.visualize("post_processed", config)
quantize_search.compare_statistics(config)
quantize_search.evaluate("post_process", config)
