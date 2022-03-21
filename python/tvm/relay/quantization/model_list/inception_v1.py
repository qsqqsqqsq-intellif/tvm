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
import mxnet
import gluoncv
import tvm
from tvm import relay
import tvm.relay.quantization

mxnet.random.seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if tvm.runtime.enabled("gpu"):
    ctx = tvm.cuda()
    target = "cuda"
else:
    ctx = tvm.cpu()
    target = "llvm"

batch_size = 1
calibrate_num = 500
num_workers = 8
model_name = "googlenet"
performance = {"float": 72.882, "int8": 72.6640}
root_path = os.path.join(os.path.expanduser("~"), "Documents/quantize_result")
data_path = "/data/zhaojinxi/data/imagenet"
# data_path = "/home/yhh/Desktop/dedatasets-lfs"

all_op = [
    "conv2d_bias_add",
    "nn.relu",
    "nn.max_pool2d",
    "concatenate",
    "nn.sum_pool2d",
    "nn.batch_flatten",
    "dense_bias_add",
]


def prepare_data_loaders(data_path, batch_size):
    transforms = mxnet.gluon.data.vision.transforms.Compose(
        [
            mxnet.gluon.data.vision.transforms.Resize(256, keep_ratio=True),
            mxnet.gluon.data.vision.transforms.CenterCrop(224),
            mxnet.gluon.data.vision.transforms.ToTensor(),
        ]
    )
    dataset = gluoncv.data.imagenet.classification.ImageNet(data_path, train=False).transform_first(
        transforms
    )

    sampler = mxnet.gluon.data.RandomSampler(len(dataset))
    data_loader = mxnet.gluon.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers
    )
    return data_loader


data_loader = prepare_data_loaders(data_path, batch_size)

calibrate_data = []
for i, (image, label) in enumerate(data_loader):
    if i >= (calibrate_num // batch_size):
        break
    image = (image.asnumpy() * 255).astype(numpy.uint8)
    calibrate_data.append({"input": image})


def yield_calibrate_data():
    for i in calibrate_data:
        yield i


def evaluate(runtime):
    correct = 0
    total = 0

    t = tqdm.tqdm(data_loader)
    for image, label in t:
        image = (image.asnumpy() * 255).astype(numpy.uint8)
        data = {"input": image}
        label = label.asnumpy()
        runtime.set_input(**data)
        runtime.run()
        output = runtime.get_output(0).asnumpy()
        result = output.argmax(axis=1) == label
        correct = correct + result.astype(numpy.float32).sum()
        total = total + label.shape[0]
        acc = correct / total * 100
        t.set_postfix({"accuracy": "{:.4f}".format(acc)})
    return acc


path = os.path.join(root_path, model_name, "origin_mod.json")
if os.path.exists(path):
    mod = None
    params = None
else:
    kwargs = {}
    if model_name.startswith("resnext"):
        kwargs["use_se"] = True
    model = gluoncv.model_zoo.get_model(
        model_name, pretrained=True, ctx=[mxnet.cpu()], classes=1000, **kwargs
    )
    model.hybridize()
    shape_dict = {"input": (1, 3, 224, 224)}
    mod, params = relay.frontend.from_mxnet(model, shape_dict)

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
        "input": {
            "mean": [123.675, 116.28, 103.53],
            "std": [58.395, 57.12, 57.375],
            "axis": 1,
        },
    },
    compare_statistics=False,
)

config = quantize_search.get_default_config()
quantize_search.quantize(config)
# quantize_search.visualize("post_process", config)
quantize_search.evaluate("post_process", config)
