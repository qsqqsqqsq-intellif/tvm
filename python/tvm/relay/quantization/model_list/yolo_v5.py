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
import time
import json
from pathlib import Path
import numpy
import tqdm
import cv2
import torch
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tvm.relay.quantization.threshold import Threshold
from tvm.relay.quantization.method_dtype import Method
import tvm
import tvm.relay as relay
import tvm.relay.quantization

torch.manual_seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if tvm.runtime.enabled("gpu"):
    ctx = tvm.gpu()
    target = "cuda"
else:
    ctx = tvm.cpu()
    target = "llvm"

batch_size = 1
calibrate_num = 300
num_workers = 16
model_name = "yolov5s"
performance = {"float": 37.098, "int8": None}
root_path = os.path.join(os.path.expanduser("~"), "Documents/quantize_result")
anno_json = "/data/zhaojinxi/data/coco/annotations/instances_val2017.json"
data_path = "/data/zhaojinxi/data/coco/val2017"
# anno_json = "/home/yhh/Desktop/dedatasets-lfs/coco_val2017/annotations/instances_val2017.json"
# data_path = "/home/yhh/Desktop/dedatasets-lfs/coco_val2017/val2017"

all_op = [
    "conv2d_bias_add",
    "swish",
    "add",
    "concatenate",
    "nn.max_pool2d",
    "image.resize2d",
    "reshape",
    "transpose",
    "sigmoid",
    "strided_slice",
    "multiply",
    "subtract",
    "power",
]

'''
import numpy as np
import matplotlib.pyplot as plt
class_name = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect="equal")
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                fill=False,
                edgecolor="red",
                linewidth=3.5,
            )
        )

        ax.text(
            bbox[0],
            bbox[1] - 2,
            "{:s} {:.3f}".format(class_name, score),
            bbox=dict(facecolor="blue", alpha=0.5),
            fontsize=14,
            color="white",
        )

    ax.set_title(
        ("{} detections with " "p({} | box) >= {:.1f}").format(class_name, class_name, thresh),
        fontsize=14,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.draw()
    plt.show()
'''


def prepare_data_loaders(data_path, batch_size):
    class LoadImagesAndLabels(torch.utils.data.Dataset):
        def __init__(self, path):
            self.img_files = []
            for i in os.listdir(path):
                self.img_files.append(os.path.join(path, i))

        def __len__(self):
            return len(self.img_files)

        def load_image(self, i):
            path = self.img_files[i]
            im = cv2.imread(path)  # BGR
            assert im is not None, f"Image Not Found {path}"
            h0, w0 = im.shape[:2]  # orig hw
            r = 640 / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                im = cv2.resize(
                    im,
                    (int(w0 * r), int(h0 * r)),
                    interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR,
                )
            return im, (h0, w0), im.shape[:2]

        def letterbox(self, im, new_shape=640, color=(114, 114, 114)):
            shape = im.shape[:2]
            new_shape = (640, 640)

            r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
            r = min(r, 1.0)

            ratio = r, r  # width, height ratios
            new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
            dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

            dw /= 2
            dh /= 2

            if shape[::-1] != new_unpad:  # resize
                im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
            return im, ratio, (dw, dh)

        def __getitem__(self, index):
            img, (h0, w0), (h, w) = self.load_image(index)

            img, ratio, pad = self.letterbox(img)
            shapes = (h0, w0), ((h / h0, w / w0), pad)

            img = img.transpose((2, 0, 1))[::-1]
            img = numpy.ascontiguousarray(img)

            return torch.from_numpy(img), self.img_files[index], shapes

    dataset = LoadImagesAndLabels(data_path)

    def collate_fn(batch):
        img, path, shapes = zip(*batch)
        return torch.stack(img, 0), path, shapes

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=torch.utils.data.RandomSampler(dataset),
        collate_fn=collate_fn,
    )
    return data_loader


data_loader = prepare_data_loaders(data_path, batch_size)

calibrate_data = []
for i, (image, _, _) in enumerate(data_loader):
    if i >= (calibrate_num // batch_size):
        break
    image = image.numpy()
    calibrate_data.append({"input": image})


def yield_calibrate_data():
    for i in calibrate_data:
        yield i


def evaluate(runtime):
    def xywh2xyxy(x):
        y = x.clone() if isinstance(x, torch.Tensor) else numpy.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def xyxy2xywh(x):
        y = x.clone() if isinstance(x, torch.Tensor) else numpy.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    def non_max_suppression(prediction):
        xc = prediction[..., 4] > 0.001
        min_wh, max_wh = 2, 4096
        max_nms = 30000
        time_limit = 10.0

        t = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):
            x = x[xc[xi]]  # confidence

            if not x.shape[0]:
                continue

            x[:, 5:] *= x[:, 4:5]

            box = xywh2xyxy(x[:, :4])

            i, j = (x[:, 5:] > 0.001).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)

            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]

            c = x[:, 5:6] * max_wh
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = torchvision.ops.nms(boxes, scores, 0.6)
            if i.shape[0] > 300:
                i = i[:300]

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                print(f"WARNING: NMS time limit {time_limit}s exceeded")
                break
        return output

    def clip_coords(boxes, shape):
        if isinstance(boxes, torch.Tensor):
            boxes[:, 0].clamp_(0, shape[1])  # x1
            boxes[:, 1].clamp_(0, shape[0])  # y1
            boxes[:, 2].clamp_(0, shape[1])  # x2
            boxes[:, 3].clamp_(0, shape[0])  # y2
        else:  # numpy.array (faster grouped)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])

    def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
        if ratio_pad is None:
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
                img1_shape[0] - img0_shape[0] * gain
            ) / 2
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]
        coords[:, [1, 3]] -= pad[1]
        coords[:, :4] /= gain
        clip_coords(coords, img0_shape)
        return coords

    jdict = []
    class_map = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]

    for image, paths, shapes in tqdm.tqdm(data_loader):
        image = image.numpy()
        data = {"input": image}
        runtime.set_input(**data)
        runtime.run()
        tvm_output = runtime.get_output(0)
        output = tvm_output.asnumpy()
        output = torch.tensor(output)

        output = non_max_suppression(output)

        for si, pred in enumerate(output):
            path, shape = Path(paths[si]), shapes[si][0]

            if len(pred) == 0:
                continue

            predn = pred.clone()
            scale_coords(image[si].shape[1:], predn[:, :4], shape, shapes[si][1])

            image_id = int(path.stem) if path.stem.isnumeric() else path.stem
            box = xyxy2xywh(predn[:, :4])
            box[:, :2] -= box[:, 2:] / 2
            for p, b in zip(predn.tolist(), box.tolist()):
                jdict.append(
                    {
                        "image_id": image_id,
                        "category_id": class_map[int(p[5])],
                        "bbox": [round(x, 3) for x in b],
                        "score": round(p[4], 5),
                    }
                )

        """
        for j in range(1, 81):
            inds = np.where(predn[:, 5] == j - 1)[0]
            cls_box = predn[inds, :5]
            im = cv2.imread(str(paths[0]))
            vis_detections(im, class_name[j-1], cls_box, 0.25)
        """

    pred_json = "%s_predictions.json" % model_name

    with open(pred_json, "w") as f:
        json.dump(jdict, f)

    anno = COCO(anno_json)
    pred = anno.loadRes(pred_json)
    eval = COCOeval(anno, pred, "bbox")

    eval.params.imgIds = [int(Path(x).stem) for x in data_loader.dataset.img_files]
    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    map, map50 = eval.stats[:2]
    os.remove(pred_json)
    print(map * 100)
    return map


path = os.path.join(root_path, model_name, "origin_mod.json")
if os.path.exists(path):
    mod = None
    params = None
else:
    x = torch.randn([1, 3, 640, 640])
    model = torch.jit.load("/home/zhaojinxi/Documents/quantize_result/yolov5s/yolov5s.pt")
    # model = torch.jit.load("/home/yhh/Desktop/detvm/deepeye/demodels-lfs/pytorch/yolov5s/yolov5s.pt")
    shape_list = [("input", x.numpy().shape)]
    mod, params = relay.frontend.from_pytorch(model, shape_list)

# notice !!!!
# if you use calibnum is 1, and see compare_statistics
# use Threshold.RelativeEntropy
quantize_config = {}
quantize_config["call"] = {
    "threshold": Threshold.PercentileAbs,
    "method": Method.Symmetry,
    "dtype": "int8",
}
quantize_config["skip_conv_layers"] = [46, 53, 60]

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
            "mean": [0, 0, 0],
            "std": [255, 255, 255],
            "axis": 1,
        },
    },
    quantize_config=quantize_config,
    compare_statistics=False,
)

config = quantize_search.get_default_config()
quantize_search.quantize(config)
# quantize_search.visualize("post_process", config)
quantize_search.evaluate("post_process", config)
