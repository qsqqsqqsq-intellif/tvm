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
    "mobilenet-v1": {
        "input": "input:0",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "mobilenet-v1/mobilenet_v1_1.0_224.onnx",
    },
    "mobilenet-v2": {
        "input": "input",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "mobilenet-v2/mobilenetv2-7.onnx",
    },
    "mobilenet-v2_torchvision": {
        "input": "input.1",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "mobilenet-v2_torchvision/mobilenetv2_torchvision.onnx",
    },
    "mobilenet-v3-small": {
        "input": "input_image",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "mobilenet-v3-small/mobilenetv3_small_new.onnx",
    },
    "resnet18-v1": {
        "input": "data",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "resnet18-v1/resnet18-v1-7.onnx",
    },
    "resnet18-v2": {
        "input": "data",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "resnet18-v2/resnet18-v2-7.onnx",
    },
    "resnet34-v1": {
        "input": "data",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "resnet34-v1/resnet34-v1-7.onnx",
    },
    "resnet34-v2": {
        "input": "data",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "resnet34-v2/resnet34-v2-7.onnx",
    },
    "resnet50-caffe2-v1": {
        "input": "gpu_0/data_0",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "resnet50-caffe2-v1/resnet50-caffe2-v1-9.onnx",
    },
    "resnet50-v1": {
        "input": "data",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "resnet50-v1/resnet50-v1-7.onnx",
    },
    "resnet50-v2": {
        "input": "data",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "resnet50-v2/resnet50-v2-7.onnx",
    },
    "resnet101-v1": {
        "input": "data",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "resnet101-v1/resnet101-v1-7.onnx",
    },
    "resnet101-v2": {
        "input": "data",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "resnet101-v2/resnet101-v2-7.onnx",
    },
    "resnet152-v1": {
        "input": "data",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "resnet152-v1/resnet152-v1-7.onnx",
    },
    "resnet152-v2": {
        "input": "data",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "resnet152-v2/resnet152-v2-7.onnx",
    },
    "resnext50_32x4d_torchvision": {
        "input": "input.1",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "resnext50_32x4d_torchvision/resnext50_32x4d.onnx",
    },
    "resnest50": {
        "input": "input.1",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "resnest50/resnest50.onnx",
    },
    "wide_resnet50_torchvision": {
        "input": "input.1",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "wide_resnet50_torchvision/wide_resnet50_2.onnx",
    },
    "seresnet50": {
        "input": "input_image",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "seresnet50/seresnet50.onnx",
    },
    "squeezenet1.0": {
        "input": "data_0",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "squeezenet1.0/squeezenet1.0-9.onnx",
    },
    "squeezenet1.1": {
        "input": "data",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "squeezenet1.1/squeezenet1.1-7.onnx",
    },
    "vgg16": {"input": "data", "shape": [1, 3, 224, 224], "axis": 1, "file": "vgg16/vgg16-7.onnx"},
    "vgg16-bn": {
        "input": "data",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "vgg16-bn/vgg16-bn-7.onnx",
    },
    "alexnet": {
        "input": "data_0",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "alexnet/bvlcalexnet-9.onnx",
    },
    "googlenet": {
        "input": "data_0",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "googlenet/googlenet-9.onnx",
    },
    "caffenet": {
        "input": "data_0",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "caffenet/caffenet-9.onnx",
    },
    "rcnn": {
        "input": "data_0",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "rcnn/rcnn-ilsvrc13-9.onnx",
    },
    "densnet121": {
        "input": "data_0",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "densnet121/densenet-9.onnx",
    },
    "inception-v1": {
        "input": "data_0",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "inception-v1/inception-v1-9.onnx",
    },
    "inception-v2": {
        "input": "data_0",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "inception-v2/inception-v2-9.onnx",
    },
    "inception_v3_torchvision": {
        "input": "x.1",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "inception_v3_torchvision/inception_v3.onnx",
    },
    "shufflenet-v1": {
        "input": "gpu_0/data_0",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "shufflenet-v1/shufflenet-9.onnx",
    },
    "shufflenet-v2": {
        "input": "input",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "shufflenet-v2/shufflenet-v2-10.onnx",
    },
    "zfnet512": {
        "input": "gpu_0/data_0",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "zfnet512/zfnet512-9.onnx",
    },
    "efficientnet-lit4": {
        "input": "images:0",
        "shape": [1, 224, 224, 3],
        "axis": 3,
        "file": "efficientnet-lit4/efficientnet-lite4-11.onnx",
    },
    "efficientnet-b0": {
        "input": "input.1",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "efficientnet-b0/efficientnet_b0.onnx",
    },
    "EfficientNetV2: efficientnet_v2_m": {
        "input": "input.1",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "EfficientNetV2/efficientnet_v2_m.onnx",
    },
    "EfficientNetV2: efficientnet_v2_s": {
        "input": "input.1",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "EfficientNetV2/efficientnet_v2_s.onnx",
    },
    "mnist": {"input": "Input3", "shape": [1, 1, 28, 28], "axis": 1, "file": "mnist/mnist-8.onnx"},
    "ghostnet": {
        "input": "input.1",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "ghostnet/ghostnet.onnx",
    },
    "condensenet-v2": {
        "input": "input.1",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "condensenet-v2/condensevetv2_a.onnx",
    },
    "mnasnet1_0_torchvision": {
        "input": "input.1",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "mnasnet1_0_torchvision/mnasnet1_0.onnx",
    },
    "DDRNet23_slim": {
        "input": "input_image",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "DDRNet23_slim/DDRNet23_slim.onnx",
    },
    "DDRNet23": {
        "input": "input_image",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "DDRNet23/DDRNet23.onnx",
    },
    "DDRNet39": {
        "input": "input_image",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "DDRNet39/DDRNet39.onnx",
    },
    "HRNet_W18": {
        "input": "input_image",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "HRNet_W18/HRNet-W18.onnx",
    },
    "vovnet19": {
        "input": "input_image",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "vovnet19/vovnet19.onnx",
    },
    "vovnet27_slim": {
        "input": "input_image",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "vovnet27_slim/vovnet27_slim.onnx",
    },
    "vovnet39": {
        "input": "input_image",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "vovnet39/vovnet39.onnx",
    },
    "convnext: convnext_tiny": {
        "input": "input.1",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "convnext/convnext_tiny.onnx",
    },
    "convnext: convnext_small": {
        "input": "input.1",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "convnext/convnext_small.onnx",
    },
    "convnext: convnext_base": {
        "input": "input.1",
        "shape": [1, 3, 224, 224],
        "axis": 1,
        "file": "convnext/convnext_base.onnx",
    },
    "r3d_18_torchvision": {
        "input": "0",
        "shape": [1, 3, 16, 112, 112],
        "axis": 1,
        "file": "r3d_18_torchvision/r3d_18.onnx",
    },
    "mc3_18_torchvision": {
        "input": "0",
        "shape": [1, 3, 16, 112, 112],
        "axis": 1,
        "file": "mc3_18_torchvision/mc3_18.onnx",
    },
    "r2plus1d_18_torchvision": {
        "input": "0",
        "shape": [1, 3, 16, 112, 112],
        "axis": 1,
        "file": "r2plus1d_18_torchvision/r2plus1d_18.onnx",
    },
}

source_path = "/data/share/demodels-lfs/onnx"
target_path = "/home/zhaojinxi/Documents/onnx_result"

models = {}
for name in os.listdir(source_path):
    model_path = os.path.join(source_path, name)
    tmp1 = []
    for file in os.listdir(model_path):
        if os.path.splitext(file)[1] == ".onnx":
            tmp1.append(file)
    models[name] = {"file": tmp1}
for k, v in meta.items():
    if ": " in k:
        tmp1, tmp2 = k.split(": ")
        for i in models[tmp1]["file"]:
            print(os.path.join(k, i))
    else:
        for i in models[k]["file"]:
            print(os.path.join(k, i))

for name, v in meta.items():
    save_path = os.path.join(target_path, name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    log_path = os.path.join(save_path, "log.txt")

    rerun = True
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            line = f.readlines()
            if line != [] and line[-1] == "finished\n":
                rerun = False

    if rerun:
        log = open(log_path, "w")
        sys.stdout = log

        file_path = os.path.join(source_path, v["file"])
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
