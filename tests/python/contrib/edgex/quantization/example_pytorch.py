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
        x = torch.nn.functional.softmax(x)
        return x


model = Example()


def prepare_data_loaders(data_path, batch_size):
    dataset_test = torchvision.datasets.ImageFolder(
        os.path.join(data_path, "val"),
        torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(
                #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                # ),
            ]
        ),
    )

    test_sampler = torch.utils.data.RandomSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, sampler=test_sampler
    )
    return data_loader_test


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


data_path = "/data/zhaojinxi/data/imagenet"
data_loader = prepare_data_loaders(data_path, 1)

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
    top1 = AverageMeter("Acc@1", ":6.2f")
    for image, label in tqdm.tqdm(data_loader):
        image = (image.numpy() * 255).astype(numpy.uint8)
        data = {"input": image}
        label = label.numpy()
        runtime.set_input(**data)
        runtime.run()
        tvm_output = runtime.get_output(0)
        output = tvm_output.asnumpy()
        acc1 = (output.argmax(axis=1) == label).astype(numpy.float32).sum() / output.shape[0] * 100
        top1.update(acc1, image.shape[0])
        print(top1.avg)
    return top1.avg


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
    dataset=yield_calibrate_data,
    calibrate_num=calibrate_num,
    eval_func=evaluate,
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
    compare_statistics=True,
)

config = quantize_search.get_default_config()
quantize_search.quantize(config)
# quantize_search.visualize("post_processed", config)
quantize_search.evaluate("post_process", config)
