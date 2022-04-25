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
import PIL
import copy
import random
import tqdm
import numpy
import torch
import torchvision
import pycocotools
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
model_name = "lraspp_mobilenet_v3_large"
performance = {"float": None, "int8": None}
root_path = "/data/zhaojinxi/Documents/quantize_result"
data_path = "/data/zhaojinxi/data/coco"

all_op = [
    "conv2d_bias_add",
    "hard_swish",
    "nn.relu",
    "add",
    "nn.sum_pool2d",
    "hard_sigmoid",
    "multiply",
    "nn.conv2d",
    "sigmoid",
    "image.resize2d",
]


def prepare_data_loaders(data_path, batch_size):
    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, image, target):
            for t in self.transforms:
                image, target = t(image, target)
            return image, target

    class FilterAndRemapCocoCategories:
        def __init__(self, categories, remap=True):
            self.categories = categories
            self.remap = remap

        def __call__(self, image, anno):
            anno = [obj for obj in anno if obj["category_id"] in self.categories]
            if not self.remap:
                return image, anno
            anno = copy.deepcopy(anno)
            for obj in anno:
                obj["category_id"] = self.categories.index(obj["category_id"])
            return image, anno

    class ConvertCocoPolysToMask:
        def __call__(self, image, anno):
            w, h = image.size
            segmentations = [obj["segmentation"] for obj in anno]
            cats = [obj["category_id"] for obj in anno]
            if segmentations:
                masks = self.convert_coco_poly_to_mask(segmentations, h, w)
                cats = torch.as_tensor(cats, dtype=masks.dtype)
                # merge all instance masks into a single segmentation map
                # with its corresponding categories
                target, _ = (masks * cats[:, None, None]).max(dim=0)
                # discard overlapping instances
                target[masks.sum(0) > 1] = 255
            else:
                target = torch.zeros((h, w), dtype=torch.uint8)
            target = PIL.Image.fromarray(target.numpy())
            return image, target

        def convert_coco_poly_to_mask(self, segmentations, height, width):
            masks = []
            for polygons in segmentations:
                rles = pycocotools.mask.frPyObjects(polygons, height, width)
                mask = pycocotools.mask.decode(rles)
                if len(mask.shape) < 3:
                    mask = mask[..., None]
                mask = torch.as_tensor(mask, dtype=torch.uint8)
                mask = mask.any(dim=2)
                masks.append(mask)
            if masks:
                masks = torch.stack(masks, dim=0)
            else:
                masks = torch.zeros((0, height, width), dtype=torch.uint8)
            return masks

    class SegmentationPresetEval:
        def __init__(self, min_size=520, max_size=520):
            self.min_size = min_size
            if max_size is None:
                max_size = min_size
            self.max_size = max_size

        def __call__(self, image, target):
            size = random.randint(self.min_size, self.max_size)
            image = torchvision.transforms.functional.resize(image, (size, size))
            target = torchvision.transforms.functional.resize(
                target, (size, size), interpolation=torchvision.transforms.InterpolationMode.NEAREST
            )
            image = torchvision.transforms.functional.pil_to_tensor(image)
            target = torch.as_tensor(numpy.array(target), dtype=torch.int64)
            return image, target

    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]

    transforms = Compose(
        [
            FilterAndRemapCocoCategories(CAT_LIST, remap=True),
            ConvertCocoPolysToMask(),
            SegmentationPresetEval(),
        ]
    )

    img_folder = os.path.join(data_path, "val2017")
    ann_file = os.path.join(data_path, "annotations", "instances_val2017.json")

    dataset = torchvision.datasets.CocoDetection(img_folder, ann_file, transforms=transforms)

    def cat_list(images, fill_value=0):
        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
        batch_shape = (len(images),) + max_size
        batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
        return batched_imgs

    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets

    sampler = torch.utils.data.SequentialSampler(dataset)
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
for i, (image, label) in enumerate(data_loader):
    if i >= (calibrate_num // batch_size):
        break
    image = image.numpy()
    calibrate_data.append({"input": image})


def yield_calibrate_data():
    for i in calibrate_data:
        yield i


def evaluate(runtime):
    mat = numpy.zeros((21, 21), dtype=numpy.int64)

    t = tqdm.tqdm(data_loader)
    for image, label in t:
        image = image.numpy()
        label = label.numpy()

        data = {"input": image}
        runtime.set_input(**data)
        runtime.run()
        output = runtime.get_output(0).asnumpy()

        a = label.reshape(-1)
        b = output.argmax(1).reshape(-1)
        k = (a >= 0) & (a < 21)
        inds = 21 * a[k] + b[k]
        mat = mat + numpy.bincount(inds, minlength=21 ** 2).reshape(21, 21)

        h = mat.astype(numpy.float64)
        # acc_global = numpy.diag(h).sum() / h.sum()
        # acc = numpy.diag(h) / h.sum(1)
        iu = numpy.diag(h) / (h.sum(1) + h.sum(0) - numpy.diag(h))
        miou = iu.mean() * 100
        t.set_postfix({"miou": "{:.4f}".format(miou)})
    return miou


path = os.path.join(root_path, model_name, "origin_mod.json")
if os.path.exists(path):
    mod = None
    params = None
else:
    x = torch.randn([1, 3, 520, 520])
    model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(pretrained=True)

    class LrasppMobilenetv3(torch.nn.Module):
        def __init__(self):
            super(LrasppMobilenetv3, self).__init__()
            self.model = model

        def forward(self, x):
            x = self.model(x)
            return x["out"]

    model = LrasppMobilenetv3()
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
