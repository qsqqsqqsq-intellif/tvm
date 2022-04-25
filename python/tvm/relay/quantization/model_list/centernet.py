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
import cv2
import numpy
import tqdm
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
num_workers = 16
model_name = "center_net_resnet50_v1b_coco"
performance = {"float": 32.3, "int8": None}
root_path = "/data/zhaojinxi/Documents/quantize_result"
data_path = "/data/zhaojinxi/data/coco"
# data_path = "/home/yhh/Desktop/dedatasets-lfs/coco_val2017"

all_op = [
    "conv2d_bias_add",
    "nn.relu",
    "nn.max_pool2d",
    "add",
    "nn.conv2d_transpose",
    "sigmoid",
    "equal",
    "cast",
    "multiply",
    "reshape",
    "topk",
    "divide",
    "mod",
    "transpose",
    "slice_like",
    "expand_dims",
    "tile",
    "zeros_like",
    "concatenate",
    "gather_nd",
    "subtract",
    "ones_like",
]


def prepare_data_loaders(data_path, batch_size):
    dataset = gluoncv.data.COCODetection(
        root=data_path, splits="instances_val2017", skip_empty=False
    )

    class CenterNetDefaultValTransform:
        def __init__(self, width, height):
            self._width = width
            self._height = height

        def __call__(self, src, label):
            img, bbox = src.asnumpy(), label
            input_h, input_w = self._height, self._width
            h, w, _ = src.shape
            s = max(h, w) * 1.0
            c = numpy.array([w / 2.0, h / 2.0], dtype=numpy.float32)
            trans_input = gluoncv.data.transforms.bbox.get_affine_transform(
                c, s, 0, [input_w, input_h]
            )
            inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
            output_w = input_w
            output_h = input_h
            trans_output = gluoncv.data.transforms.bbox.get_affine_transform(
                c, s, 0, [output_w, output_h]
            )
            for i in range(bbox.shape[0]):
                bbox[i, :2] = gluoncv.data.transforms.bbox.affine_transform(
                    bbox[i, :2], trans_output
                )
                bbox[i, 2:4] = gluoncv.data.transforms.bbox.affine_transform(
                    bbox[i, 2:4], trans_output
                )
            bbox[:, :2] = numpy.clip(bbox[:, :2], 0, output_w - 1)
            bbox[:, 2:4] = numpy.clip(bbox[:, 2:4], 0, output_h - 1)

            img = mxnet.ndarray.from_numpy(inp)
            img = mxnet.nd.transpose(img, [2, 0, 1])
            return img, bbox.astype("float32")

    batchify_fn = gluoncv.data.batchify.Tuple(
        gluoncv.data.batchify.Stack(), gluoncv.data.batchify.Pad(pad_val=-1)
    )

    data_loader = mxnet.gluon.data.DataLoader(
        dataset.transform(CenterNetDefaultValTransform(512, 512)),
        batch_size,
        batchify_fn=batchify_fn,
        num_workers=num_workers,
    )
    return data_loader


data_loader = prepare_data_loaders(data_path, batch_size)

calibrate_data = []
for i, (image, label) in enumerate(data_loader):
    if i >= (calibrate_num // batch_size):
        break
    image = image.asnumpy()
    calibrate_data.append({"input": image})


def yield_calibrate_data():
    for i in calibrate_data:
        yield i


def evaluate(runtime):
    dataset = gluoncv.data.COCODetection(
        root=data_path, splits="instances_val2017", skip_empty=False
    )
    metric = gluoncv.utils.metrics.coco_detection.COCODetectionMetric(
        dataset,
        model_name,
        cleanup=True,
        data_shape=(512, 512),
        post_affine=gluoncv.data.transforms.presets.center_net.get_post_transform,
    )
    metric.reset()

    for image, label in tqdm.tqdm(data_loader):
        image = image.asnumpy()
        data = {"input": image}
        label = label.asnumpy()

        runtime.set_input(**data)
        runtime.run()
        ids = runtime.get_output(0).asnumpy()
        scores = runtime.get_output(1).asnumpy()
        bboxes = runtime.get_output(2).asnumpy()

        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []

        det_ids.append(ids)
        det_scores.append(scores)
        det_bboxes.append(bboxes.clip(0, image.shape[2]))
        gt_ids.append(label[:, :, 4:5])
        gt_bboxes.append(label[:, :, 0:4])
        gt_difficults.append(None)

        metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
    names, values = metric.get()
    for k, v in zip(names, values):
        print(k, v)
    return float(values[-1])


path = os.path.join(root_path, model_name, "origin_mod.json")
if os.path.exists(path):
    mod = None
    params = None
else:
    model = gluoncv.model_zoo.get_model(model_name, pretrained=True)
    model.hybridize(static_shape=True, static_alloc=True)
    shape_dict = {"input": (1, 3, 512, 512)}
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
    verbose=True,
)

config = quantize_search.get_default_config()
quantize_search.quantize(config)
quantize_search.evaluate("post_process", config)
