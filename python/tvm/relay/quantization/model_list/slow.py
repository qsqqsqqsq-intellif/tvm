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
from torchvision.transforms._transforms_video import CenterCropVideo
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
import tvm
from tvm import relay
import tvm.relay.quantization

torch.manual_seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if tvm.runtime.enabled("gpu"):
    ctx = tvm.cuda()
    target = "cuda"
else:
    ctx = tvm.cpu()
    target = "llvm"

batch_size = 1
calibrate_num = 500
num_workers = 16
model_name = "slow_r50"
performance = {"float": 63.4735, "int8": 63.3776}
root_path = os.path.join(os.path.expanduser("~"), "Documents/quantize_result")
data_path = "/data/zhaojinxi/data/kinetics400/val"

all_op = [
    "conv3d_bias_add",
    "nn.relu",
    "nn.max_pool3d",
    "add",
    "nn.sum_pool3d",
    "transpose",
    "reshape",
    "nn.dense",
    "multiply",
]


def prepare_data_loaders(data_path, batch_size):
    transform = ApplyTransformToKey(
        key="video",
        transform=torchvision.transforms.Compose(
            [
                UniformTemporalSubsample(8),
                ShortSideScale(size=256),
                CenterCropVideo(crop_size=(256, 256)),
            ]
        ),
    )

    def loader(path):
        video = EncodedVideo.from_path(path)
        video_data = video.get_clip(start_sec=0, end_sec=64 / 30)
        video_data = transform(video_data)
        inputs = video_data["video"]
        return inputs

    def is_valid_file(path):
        try:
            EncodedVideo.from_path(path)
            valid = True
        except:
            valid = False
        return valid

    dataset = torchvision.datasets.DatasetFolder(
        root=data_path,
        loader=loader,
        is_valid_file=is_valid_file,
    )

    sampler = torch.utils.data.RandomSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler
    )
    return data_loader


data_loader = prepare_data_loaders(data_path, batch_size)

calibrate_data = []
for i, (video, label) in enumerate(data_loader):
    if i >= (calibrate_num // batch_size):
        break
    video = numpy.clip(video.numpy(), 0, 255).round().astype(numpy.uint8)
    calibrate_data.append({"input": video})


def yield_calibrate_data():
    for i in calibrate_data:
        yield i


def evaluate(runtime):
    correct = 0
    total = 0

    t = tqdm.tqdm(data_loader)
    for video, label in t:
        video = numpy.clip(video.numpy(), 0, 255).round().astype(numpy.uint8)
        data = {"input": video}
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
    model = torch.hub.load("facebookresearch/pytorchvideo", model_name, pretrained=True)
    x = torch.randn([1, 3, 8, 256, 256])
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
            "mean": [114.75, 114.75, 114.75],
            "std": [57.375, 57.375, 57.375],
            "axis": 1,
        },
    },
    compare_statistics=False,
)

config = quantize_search.get_default_config()
quantize_search.quantize(config)
# quantize_search.visualize("post_process", config)
quantize_search.evaluate("post_process", config)
