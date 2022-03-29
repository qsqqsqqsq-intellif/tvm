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
import sys
import traceback
import onnx
import numpy
import tvm
import tvm.relay as relay
import tvm.relay.quantization


def _run(
    model_name: str,
    mod=None,
    params=None,
    quantize_config=None,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    axis=1,
    root_path=".",
):
    mod = relay.transform.InferType()(mod)
    single_data = {}
    for param in mod["main"].params:
        input_name = param.name_hint
        dtype = param.checked_type.dtype
        shape = [int(_) for _ in param.checked_type.shape]
        if params is not None and input_name in params:
            continue  # skip model weight params
        data = numpy.random.randint(0, 256, shape).astype(dtype)
        single_data[input_name] = data

    def eval_nothing():
        return 0.0

    quantize_search = relay.quantization.QuantizeSearch(
        model_name=model_name,
        mod=mod,
        params=params,
        dataset=lambda: iter([single_data]),
        calibrate_num=1,
        eval_func=eval_nothing,
        ctx=tvm.cpu(),
        target="llvm",
        root_path=root_path,
        norm={
            "input": {
                "mean": mean,
                "std": std,
                "axis": axis,
            },
        },
        quantize_config=quantize_config,
        compare_statistics=True,
        verbose=True,
    )

    config = quantize_search.get_default_config()
    quantize_search.quantize(config)
    print(quantize_search.results[0]["other"]["similarity"][0][-1][1] >= 0.99)


meta = {
    "mobilenet-v1": {"input": "input:0", "shape": [1, 3, 224, 224], "axis": 1},
    "mobilenet-v2": {"input": "input", "shape": [1, 3, 224, 224], "axis": 1},
    "mobilenet-v2_torchvision": {"input": "input.1", "shape": [1, 3, 224, 224], "axis": 1},
    "mobilenet-v3-small": {"input": "input_image", "shape": [1, 3, 224, 224], "axis": 1},
    "resnet18-v1": {"input": "data", "shape": [1, 3, 224, 224], "axis": 1},
    "resnet18-v2": {"input": "data", "shape": [1, 3, 224, 224], "axis": 1},
    "resnet34-v1": {"input": "data", "shape": [1, 3, 224, 224], "axis": 1},
    "resnet34-v2": {"input": "data", "shape": [1, 3, 224, 224], "axis": 1},
    "resnet50-caffe2-v1": {"input": "gpu_0/data_0", "shape": [1, 3, 224, 224], "axis": 1},
    "resnet50-v1": {"input": "data", "shape": [1, 3, 224, 224], "axis": 1},
    "resnet50-v2": {"input": "data", "shape": [1, 3, 224, 224], "axis": 1},
    "resnet101-v1": {"input": "data", "shape": [1, 3, 224, 224], "axis": 1},
    "resnet101-v2": {"input": "data", "shape": [1, 3, 224, 224], "axis": 1},
    "resnet152-v1": {"input": "data", "shape": [1, 3, 224, 224], "axis": 1},
    "resnet152-v2": {"input": "data", "shape": [1, 3, 224, 224], "axis": 1},
    "resnext50_32x4d_torchvision": {"input": "input.1", "shape": [1, 3, 224, 224], "axis": 1},
    "resnest50": {"input": "input.1", "shape": [1, 3, 224, 224], "axis": 1},
    "wide_resnet50_torchvision": {"input": "input.1", "shape": [1, 3, 224, 224], "axis": 1},
    "seresnet50": {"input": "input_image", "shape": [1, 3, 224, 224], "axis": 1},
    "squeezenet1.0": {"input": "data_0", "shape": [1, 3, 224, 224], "axis": 1},
    "squeezenet1.1": {"input": "data", "shape": [1, 3, 224, 224], "axis": 1},
    "vgg16": {"input": "data", "shape": [1, 3, 224, 224], "axis": 1},
    "vgg16-bn": {"input": "data", "shape": [1, 3, 224, 224], "axis": 1},
    "alexnet": {"input": "data_0", "shape": [1, 3, 224, 224], "axis": 1},
    "googlenet": {"input": "data_0", "shape": [1, 3, 224, 224], "axis": 1},
    "caffenet": {"input": "data_0", "shape": [1, 3, 224, 224], "axis": 1},
    "rcnn": {"input": "data_0", "shape": [1, 3, 224, 224], "axis": 1},
    "densnet121": {"input": "data_0", "shape": [1, 3, 224, 224], "axis": 1},
    "inception-v1": {"input": "data_0", "shape": [1, 3, 224, 224], "axis": 1},
    "inception-v2": {"input": "data_0", "shape": [1, 3, 224, 224], "axis": 1},
    "inception_v3_torchvision": {"input": "x.1", "shape": [1, 3, 224, 224], "axis": 1},
    "shufflenet-v1": {"input": "gpu_0/data_0", "shape": [1, 3, 224, 224], "axis": 1},
    "shufflenet-v2": {"input": "input", "shape": [1, 3, 224, 224], "axis": 1},
    "zfnet512": {"input": "gpu_0/data_0", "shape": [1, 3, 224, 224], "axis": 1},
    "efficientnet-lit4": {"input": "images:0", "shape": [1, 224, 224, 3], "axis": 3},
    "efficientnet-b0": {"input": "input.1", "shape": [1, 3, 224, 224], "axis": 1},
    "EfficientNetV2: efficientnet_v2_m": {"input": "input.1", "shape": [1, 3, 224, 224], "axis": 1},
    "EfficientNetV2: efficientnet_v2_s": {"input": "input.1", "shape": [1, 3, 224, 224], "axis": 1},
    "mnist": {"input": "Input3", "shape": [1, 1, 28, 28], "axis": 1},
    "ghostnet": {"input": "input.1", "shape": [1, 3, 224, 224], "axis": 1},
    "condensenet-v2": {"input": "input.1", "shape": [1, 3, 224, 224], "axis": 1},
    "mnasnet1_0_torchvision": {"input": "input.1", "shape": [1, 3, 224, 224], "axis": 1},
    "DDRNet23_slim": {"input": "input_image", "shape": [1, 3, 224, 224], "axis": 1},
    "DDRNet23": {"input": "input_image", "shape": [1, 3, 224, 224], "axis": 1},
    "DDRNet39": {"input": "input_image", "shape": [1, 3, 224, 224], "axis": 1},
    "HRNet_W18": {"input": "input_image", "shape": [1, 3, 224, 224], "axis": 1},
    "vovnet19": {"input": "input_image", "shape": [1, 3, 224, 224], "axis": 1},
    "vovnet27_slim": {"input": "input_image", "shape": [1, 3, 224, 224], "axis": 1},
    "vovnet39": {"input": "input_image", "shape": [1, 3, 224, 224], "axis": 1},
    "convnext: tiny": {"input": "input.1", "shape": [1, 3, 224, 224], "axis": 1},
    "convnext: small": {"input": "input.1", "shape": [1, 3, 224, 224], "axis": 1},
    "convnext: base": {"input": "input.1", "shape": [1, 3, 224, 224], "axis": 1},
    "resnet3d: r3d_18_torchvision": {"input": "0", "shape": [1, 3, 16, 112, 112], "axis": 1},
    "resnet3d: mc3_18_torchvision": {"input": "0", "shape": [1, 3, 16, 112, 112], "axis": 1},
    "resnet3d: r2plus1d_18_torchvision": {"input": "0", "shape": [1, 3, 16, 112, 112], "axis": 1},
}

source_path = "/data/share/demodels-lfs/onnx"
target_path = "/home/zhaojinxi/Documents/onnx_result"

models = {}
for name in os.listdir(source_path):
    model_path = os.path.join(source_path, name)
    for file in os.listdir(model_path):
        if os.path.splitext(file)[1] == ".onnx":
            models[name] = {"file": file}

for k, v in meta.items():
    meta[k].update(models[k])

for name, v in meta.items():
    save_path = os.path.join(target_path, name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    log_path = os.path.join(save_path, "log.txt")

    rerun = True
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            line = f.readlines()
            if line[-1] == "finished\n":
                rerun = False

    if rerun:
        log = open(log_path, "w")
        sys.stdout = log

        model_path = os.path.join(source_path, name)
        file_path = os.path.join(model_path, v["file"])
        model = onnx.load(file_path)
        log.write("input shape:\n" + str(v["shape"]) + "\n\n")

        quantize_config = {}
        quantize_config["calib_method"] = "percentile_0.9999"

        try:
            mod, params = relay.frontend.from_onnx(model, shape={v["input"]: v["shape"]})
            _run(name, mod, params, quantize_config, axis=v["axis"], root_path=target_path)
        except Exception:
            log.write(traceback.format_exc())

        log.write("finished\n")
        log.close()
