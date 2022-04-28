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
import torchvision
import tvm
import tvm.relay as relay
import tvm.relay.quantization

torch.manual_seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ctx = tvm.cpu()
target = "llvm"

batch_size = 1
calibrate_num = 500
num_workers = 8
model_name = "CvT-13-224x224-IN-1k"
performance = {"float32": 81.5860, "int8": 81.2240}
root_path = "/data/zhaojinxi/Documents/quantize_result"
data_path = "/data/zhaojinxi/data/imagenet"

all_op = [
    "conv2d_bias_add",
    "reshape",
    "transpose",
    "nn.layer_norm",
    "high_dimension_dense_add",
    "nn.batch_matmul",
    "multiply",
    "nn.softmax",
    "add",
    "sigmoid",
    "concatenate",
    "split",
    "squeeze",
    "dense_bias_add",
]


def prepare_data_loaders(data_path, batch_size):
    dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_path, "val"),
        torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256, interpolation=3),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
            ]
        ),
    )

    sampler = torch.utils.data.RandomSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        sampler=sampler,
        drop_last=False,
    )
    return data_loader


data_loader = prepare_data_loaders(data_path, batch_size)

calibrate_data = []
for i, (image, label) in enumerate(data_loader):
    if i >= (calibrate_num // batch_size):
        break
    image = (image.numpy() * 255).astype(numpy.uint8)
    calibrate_data.append({"input": image})


def yield_calibrate_data():
    for i in calibrate_data:
        yield i


def evaluate(runtime):
    correct = 0
    total = 0

    t = tqdm.tqdm(data_loader)
    for image, label in t:
        image = (image.numpy() * 255).astype(numpy.uint8)
        data = {"input": image}
        label = label.numpy()
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
    x = torch.randn([1, 3, 224, 224])
    model = torch.jit.load(
        "/home/zhaojinxi/Documents/quantize_result/CvT-13-224x224-IN-1k/CvT-13-224x224-IN-1k.pt"
    )
    shape_list = [("input", x.numpy().shape)]
    mod, params = relay.frontend.from_pytorch(model, shape_list)

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
    verbose=True,
)

config = quantize_search.get_default_config()
quantize_search.quantize(config)
quantize_search.evaluate("post_process", config)