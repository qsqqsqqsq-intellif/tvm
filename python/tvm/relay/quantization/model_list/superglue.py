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

import onnx
import tvm
from tvm import relay
import tvm.relay.quantization
import numpy as np

path = "/data/share/pd_models/superglue_new.onnx"

mod = onnx.load(path)
shape = {
    "kpts0": (1, 1000, 2),
    "1": (1, 1000),
    "2": (1, 256, 1000),
    "kpts1": (1, 1000, 2),
    "6": (1, 1000),
    "7": (1, 256, 1000),
}

mod, params = relay.frontend.from_onnx(mod, shape=shape, freeze_params=True)
mod = relay.transform.InferType()(mod)


calibrate_data = []
for _ in range(30):
    calibrate_data.append(
        {
            "kpts0": np.random.randn(1, 1000, 2),
            "1": np.random.randn(1, 1000),
            "2": np.random.randn(1, 256, 1000),
            "kpts1": np.random.randn(1, 1000, 2),
            "6": np.random.randn(1, 1000),
            "7": np.random.randn(1, 256, 1000),
        }
    )


def yield_calibrate_data():
    for i in calibrate_data:
        yield i


def evaluate(runtime):

    return 0


quantize_search = relay.quantization.QuantizeSearch(
    model_name="superglue",
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
