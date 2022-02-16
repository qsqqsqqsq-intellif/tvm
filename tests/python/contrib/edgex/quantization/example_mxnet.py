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
import math
import tqdm
import numpy
import mxnet
import gluoncv
import tvm
from tvm import relay

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if tvm.runtime.enabled("gpu"):
    ctx = tvm.gpu()
    target = "cuda"
else:
    ctx = tvm.cpu()
    target = "llvm"

batch_size = 1
calibrate_num = 500
model_name = "test_mxnet"
root_path = os.path.join(os.path.expanduser("~"), "Documents/quantize_result")

# kwargs = {}
# if model_name.startswith("resnext"):
#     kwargs["use_se"] = True
# model = gluoncv.model_zoo.get_model(
#     model_name, pretrained=True, ctx=[mxnet.cpu()], classes=1000, **kwargs
# )
# model.hybridize()


class Example(mxnet.gluon.HybridBlock):
    def __init__(self):
        super(Example, self).__init__()
        self.Conv2d1 = mxnet.gluon.nn.Conv2D(
            in_channels=3, channels=8, kernel_size=3, strides=2, use_bias=False
        )
        self.BatchNorm2d1 = mxnet.gluon.nn.BatchNorm(in_channels=8)
        self.Activation1 = mxnet.gluon.nn.LeakyReLU(0.1)
        self.MaxPool2d = mxnet.gluon.nn.MaxPool2D(3, 2)

        HybridSequential1 = mxnet.gluon.nn.HybridSequential()
        HybridSequential1.add(
            mxnet.gluon.nn.Conv2D(
                in_channels=8, channels=16, kernel_size=3, strides=2, use_bias=False
            )
        )
        HybridSequential1.add(mxnet.gluon.nn.BatchNorm(in_channels=16))

        HybridSequential2 = mxnet.gluon.nn.HybridSequential()
        HybridSequential2.add(
            mxnet.gluon.nn.Conv2D(
                in_channels=8, channels=16, kernel_size=3, strides=2, use_bias=False
            )
        )
        HybridSequential2.add(mxnet.gluon.nn.BatchNorm(in_channels=16))

        self.HybridConcurrent = mxnet.gluon.contrib.nn.HybridConcurrent(1)
        self.HybridConcurrent.add(HybridSequential1)
        self.HybridConcurrent.add(HybridSequential2)
        self.Activation2 = mxnet.gluon.nn.Activation("relu")

        self.Flatten = mxnet.gluon.nn.Flatten()
        self.Dense = mxnet.gluon.nn.Dense(units=1000, use_bias=True)
        # self.Activation3 = mxnet.gluon.nn.Activation('softmax')

    def forward(self, x):
        x = self.Conv2d1(x)
        x = self.BatchNorm2d1(x)
        x = self.Activation1(x)
        x = self.MaxPool2d(x)
        x = self.HybridConcurrent(x)
        x = self.Activation2(x)
        x = self.Flatten(x)
        x = self.Dense(x)
        # x = self.Activation3(x)
        return x


model = Example()
model.initialize()
model.hybridize()
x = mxnet.random.uniform(low=-1, high=1, shape=(1, 3, 224, 224), dtype="float32")
model(x)


def prepare_data_loaders(data_path, batch_size):
    resize = int(math.ceil(224 / 0.875))
    transform_test = mxnet.gluon.data.vision.transforms.Compose(
        [
            mxnet.gluon.data.vision.transforms.Resize(resize, keep_ratio=True),
            mxnet.gluon.data.vision.transforms.CenterCrop(224),
            mxnet.gluon.data.vision.transforms.ToTensor(),
            # mxnet.gluon.data.vision.transforms.Normalize(
            #     [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            # ),
        ]
    )

    val_data = mxnet.gluon.data.DataLoader(
        gluoncv.data.imagenet.classification.ImageNet(data_path, train=False).transform_first(
            transform_test
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    return val_data


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
    image = (image.asnumpy() * 255).astype(numpy.uint8)
    calibrate_data.append({"input": image})


def yield_calibrate_data():
    for i in calibrate_data:
        yield i


def evaluate(runtime):
    top1 = AverageMeter("Acc@1", ":6.2f")
    for image, label in tqdm.tqdm(data_loader):
        image = (image.asnumpy() * 255).astype(numpy.uint8)
        data = {"input": image}
        label = label.asnumpy()
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
    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    scale=[0.229 * 255, 0.224 * 255, 0.225 * 255],
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
