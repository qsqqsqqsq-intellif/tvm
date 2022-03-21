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
from torchvision.datasets.samplers import UniformClipSampler
import tvm
from tvm import relay
import tvm.relay.quantization

torch.manual_seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ctx = tvm.cpu()
target = "llvm"

batch_size = 1
calibrate_num = 500
num_workers = 16
model_name = "r3d_18"
performance = {"float": 51.5192, "int8": 51.3242}
root_path = os.path.join(os.path.expanduser("~"), "Documents/quantize_result")
data_path = "/data/zhaojinxi/data/kinetics400"

all_op = [
    "conv3d_bias_add",
    "nn.relu",
    "add",
    "nn.adaptive_avg_pool3d",
    "reshape",
    "dense_bias_add",
]


class ConvertBHWCtoBCHW(torch.nn.Module):
    """Convert tensor from (B, H, W, C) to (B, C, H, W)"""

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(0, 3, 1, 2)


class ConvertBCHWtoCBHW(torch.nn.Module):
    """Convert tensor from (B, C, H, W) to (C, B, H, W)"""

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(1, 0, 2, 3)


def prepare_data_loaders(data_path, batch_size):
    transform = torchvision.transforms.Compose(
        [
            ConvertBHWCtoBCHW(),
            torchvision.transforms.ConvertImageDtype(torch.float32),
            torchvision.transforms.Resize((128, 171)),
            torchvision.transforms.CenterCrop((112, 112)),
            ConvertBCHWtoCBHW(),
        ]
    )

    cache_path = os.path.join(data_path, "val.pt")

    if os.path.exists(cache_path):
        dataset = torch.load(cache_path)
        dataset.transform = transform
    else:
        dataset = torchvision.datasets.Kinetics400(
            os.path.join(data_path, "val"),
            frames_per_clip=16,
            step_between_clips=16,
            transform=transform,
            num_workers=num_workers,
            frame_rate=30,
        )

        torch.save(dataset, cache_path)

    sampler = UniformClipSampler(dataset.video_clips, 5)

    def collate_fn(batch):
        batch = [(d[0], d[2]) for d in batch]
        return torch.utils.data.dataloader.default_collate(batch)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return data_loader


data_loader = prepare_data_loaders(data_path, batch_size)

calibrate_data = []
for i, (video, label) in enumerate(data_loader):
    if i >= (calibrate_num // batch_size):
        break
    video = (video.numpy() * 255).astype(numpy.uint8)
    calibrate_data.append({"input": video})


def yield_calibrate_data():
    for i in calibrate_data:
        yield i


def evaluate(runtime):
    correct = 0
    total = 0

    t = tqdm.tqdm(data_loader)
    for video, label in t:
        video = (video.numpy() * 255).astype(numpy.uint8)
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
    x = torch.randn([1, 3, 16, 112, 112])
    model = torchvision.models.video.r3d_18(pretrained=True)
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
            "mean": [110.2008, 100.63983, 95.99475],
            "std": [58.14765, 56.46975, 55.332195],
            "axis": 1,
        },
    },
    compare_statistics=False,
)

config = quantize_search.get_default_config()
quantize_search.quantize(config)
# quantize_search.visualize("post_process", config)
quantize_search.evaluate("post_process", config)
