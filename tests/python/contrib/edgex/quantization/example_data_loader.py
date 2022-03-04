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
import torch
import tvm
from tvm import relay

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if not tvm.runtime.enabled("gpu"):
    ctx = tvm.cuda()
    target = "cuda"
else:
    ctx = tvm.cpu()
    target = "llvm"

batch_size = 1
calibrate_num = 2
model_name = "test_pytorch"
root_path = os.path.join(os.path.expanduser("~"), "Documents/quantize_result")


class Example(torch.nn.Module):
    def __init__(self):
        super(Example, self).__init__()
        self.Conv2d1 = torch.nn.Conv2d(3, 8, 3, 2, bias=False)
        self.BatchNorm2d1 = torch.nn.BatchNorm2d(8)

        self.MaxPool2d = torch.nn.MaxPool2d(3, 2)

        self.Conv2d31 = torch.nn.Conv2d(8, 16, 3, 2, bias=False)
        self.BatchNorm2d31 = torch.nn.BatchNorm2d(16)

        self.Conv2d32 = torch.nn.Conv2d(8, 16, 3, 2, bias=False)
        self.BatchNorm2d32 = torch.nn.BatchNorm2d(16)

        self.Conv2d4 = torch.nn.Conv2d(32, 32, 3, 2, bias=False)
        self.BatchNorm2d4 = torch.nn.BatchNorm2d(32)

        self.Linear = torch.nn.Linear(5408, 1000, bias=True)

    def forward(self, x):
        x = self.Conv2d1(x)
        x = self.BatchNorm2d1(x)
        x = torch.nn.Sigmoid()(x)
        x = self.MaxPool2d(x)
        x1 = self.Conv2d31(x)
        x1 = self.BatchNorm2d31(x1)
        x2 = self.Conv2d32(x)
        x2 = self.BatchNorm2d32(x2)
        x = torch.cat((x1, x2), 1)
        x = self.Conv2d4(x)
        x = self.BatchNorm2d4(x)
        x = torch.relu(x)
        x = torch.reshape(x, [1, -1])
        x = self.Linear(x)
        # x = torch.nn.functional.softmax(x)
        return x


model = Example()

data_path = "/data/zhaojinxi/data/imagenet/val"

path = os.path.join(root_path, model_name, "origin")
if os.path.exists(path):
    mod = None
    params = None
else:
    x = torch.randn([1, 3, 224, 224])
    scripted_model = torch.jit.trace(model.eval(), x)
    shape_list = [("input", x.numpy().shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

quantize_search = relay.quantization.QuantizeSearch(
    model_name=model_name,
    mod=mod,
    params=params,
    dataset=None,
    calibrate_num=calibrate_num,
    eval_func=None,
    ctx=ctx,
    target=target,
    root_path=root_path,
    norm={
        "input": {
            "mean": [0.485 * 255, 0.456 * 255, 0.406 * 255],
            "std": [0.229 * 255, 0.224 * 255, 0.225 * 255],
            "axis": 1,
        },
    },
    image_path=data_path,
    image_size=(224, 224),
    channel_last=False,
    rgb="rgb",
    compare_statistics=True,
)

config = quantize_search.get_default_config()
quantize_search.quantize(config)
quantize_search.evaluate("post_process", config)
