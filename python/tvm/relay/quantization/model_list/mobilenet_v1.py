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
import numpy as np
import onnx
import tvm
from tvm import relay
import tvm.relay.quantization

path = "/data/share/demodels-lfs/onnx/mobilenet-v1/mobilenet_v1_1.0_224.onnx"

mod = onnx.load(path)
shape = {"input:0": (1, 3, 224, 224)}

mod, params = relay.frontend.from_onnx(mod, shape=shape, freeze_params=True)
mod = relay.transform.InferType()(mod)


def evaluate(runtime):

    return 0


calibrate_data = []
for _ in range(30):
    calibrate_data.append({"input:0": np.random.randn(1, 3, 224, 224)})


def yield_calibrate_data():
    for i in calibrate_data:
        yield i


quantize_search = relay.quantization.QuantizeSearch(
    model_name="mobilenet_v1",
    mod=mod,
    params=params,
    ctx=tvm.cpu(),
    dataset=yield_calibrate_data,
    calibrate_num=30,
    target="llvm",
    root_path=None,
    compare_statistics=False,
    eval_func=evaluate,
    # net_in_dtype="uint8",
    verbose=True,
)

config = quantize_search.get_default_config()
quantize_search.quantize(config)
quantized_mod = quantize_search.results[-1]["mod"]
