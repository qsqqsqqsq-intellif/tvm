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
import copy
import contextlib
import tqdm
import numpy
import torch
import torchvision
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
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
num_workers = 8
model_name = "detr_r50"
performance = {"float": None, "int8": None}
root_path = "/data/zhaojinxi/Documents/quantize_result"
data_path = "/data/zhaojinxi/data/coco"

all_op = []


def prepare_data_loaders(data_path, batch_size):
    img_folder = os.path.join(data_path, "val2017")
    ann_file = os.path.join(data_path, "annotations/instances_val2017.json")

    class RandomResize:
        def __init__(self, sizes):
            assert isinstance(sizes, (list, tuple))
            self.sizes = sizes

        def __call__(self, image, target=None):
            rescaled_image = torchvision.transforms.functional.resize(image, self.sizes)
            ratios = tuple(
                float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size)
            )
            ratio_width, ratio_height = ratios

            target = target.copy()
            if "boxes" in target:
                boxes = target["boxes"]
                scaled_boxes = boxes * torch.as_tensor(
                    [ratio_width, ratio_height, ratio_width, ratio_height]
                )
                target["boxes"] = scaled_boxes

            if "area" in target:
                area = target["area"]
                scaled_area = area * (ratio_width * ratio_height)
                target["area"] = scaled_area

            h, w = self.sizes
            target["size"] = torch.tensor([h, w])

            image = torch.from_numpy(numpy.array(rescaled_image, numpy.uint8))
            image = image.permute((2, 0, 1)).contiguous()

            target = target.copy()
            h, w = image.shape[-2:]
            if "boxes" in target:
                boxes = target["boxes"]
                boxes = self.box_xyxy_to_cxcywh(boxes)
                boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
                target["boxes"] = boxes
            return image, target

        def box_xyxy_to_cxcywh(self, x):
            x0, y0, x1, y1 = x.unbind(-1)
            b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
            return torch.stack(b, dim=-1)

    def collate_fn(batch):
        def _max_by_axis(the_list):
            maxes = the_list[0]
            for sublist in the_list[1:]:
                for index, item in enumerate(sublist):
                    maxes[index] = max(maxes[index], item)
            return maxes

        batch = list(zip(*batch))
        tensor_list = batch[0]
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
        batch[0] = tensor
        return batch

    class CocoDetection(torchvision.datasets.CocoDetection):
        def __init__(self, img_folder, ann_file, transforms):
            super(CocoDetection, self).__init__(img_folder, ann_file)
            self._transforms = transforms

        def __getitem__(self, idx):
            img, target = super(CocoDetection, self).__getitem__(idx)
            image_id = self.ids[idx]
            target = {"image_id": image_id, "annotations": target}
            img, target = self.prepare(img, target)
            if self._transforms is not None:
                img, target = self._transforms(img, target)
            return img, target

        def prepare(self, image, target):
            w, h = image.size

            image_id = target["image_id"]
            image_id = torch.tensor([image_id])

            anno = target["annotations"]

            anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

            boxes = [obj["bbox"] for obj in anno]
            # guard against no boxes via resizing
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            boxes[:, 2:] += boxes[:, :2]
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)

            classes = [obj["category_id"] for obj in anno]
            classes = torch.tensor(classes, dtype=torch.int64)

            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            classes = classes[keep]

            target = {}
            target["boxes"] = boxes
            target["labels"] = classes
            target["image_id"] = image_id

            # for conversion to coco api
            area = torch.tensor([obj["area"] for obj in anno])
            iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
            target["area"] = area[keep]
            target["iscrowd"] = iscrowd[keep]

            target["orig_size"] = torch.as_tensor([int(h), int(w)])
            target["size"] = torch.as_tensor([int(h), int(w)])

            return image, target

    dataset = CocoDetection(img_folder, ann_file, transforms=RandomResize([800, 800]))
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size,
        sampler=sampler,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
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
    class CocoEvaluator:
        def __init__(self, coco_gt, iou_types):
            assert isinstance(iou_types, (list, tuple))
            # coco_gt = copy.deepcopy(coco_gt)
            self.coco_gt = coco_gt

            self.iou_types = iou_types
            self.coco_eval = {}
            for iou_type in iou_types:
                self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

            self.img_ids = []
            self.eval_imgs = {k: [] for k in iou_types}

        def evaluate(self, coco_eval):
            """
            Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
            :return: None
            """
            # tic = time.time()
            # print('Running per image evaluation...')
            p = coco_eval.params
            # add backward compatibility if useSegm is specified in params
            if p.useSegm is not None:
                p.iouType = "segm" if p.useSegm == 1 else "bbox"
                print("useSegm (deprecated) is not None. Running {} evaluation".format(p.iouType))
            # print('Evaluate annotation type *{}*'.format(p.iouType))
            p.imgIds = list(numpy.unique(p.imgIds))
            if p.useCats:
                p.catIds = list(numpy.unique(p.catIds))
            p.maxDets = sorted(p.maxDets)
            coco_eval.params = p

            coco_eval._prepare()
            # loop through images, area range, max detection number
            catIds = p.catIds if p.useCats else [-1]

            if p.iouType == "segm" or p.iouType == "bbox":
                computeIoU = coco_eval.computeIoU
            elif p.iouType == "keypoints":
                computeIoU = coco_eval.computeOks
            coco_eval.ious = {
                (imgId, catId): computeIoU(imgId, catId) for imgId in p.imgIds for catId in catIds
            }

            evaluateImg = coco_eval.evaluateImg
            maxDet = p.maxDets[-1]
            evalImgs = [
                evaluateImg(imgId, catId, areaRng, maxDet)
                for catId in catIds
                for areaRng in p.areaRng
                for imgId in p.imgIds
            ]
            # this is NOT in the pycocotools code, but could be done outside
            evalImgs = numpy.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
            # toc = time.time()
            # print('DONE (t={:0.2f}s).'.format(toc-tic))
            return p.imgIds, evalImgs

        def update(self, predictions):
            img_ids = list(numpy.unique(list(predictions.keys())))
            self.img_ids.extend(img_ids)

            for iou_type in self.iou_types:
                results = self.prepare(predictions)

                # suppress pycocotools prints
                with open(os.devnull, "w") as devnull:
                    with contextlib.redirect_stdout(devnull):
                        coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
                coco_eval = self.coco_eval[iou_type]

                coco_eval.cocoDt = coco_dt
                coco_eval.params.imgIds = list(img_ids)
                img_ids, eval_imgs = self.evaluate(coco_eval)

                self.eval_imgs[iou_type].append(eval_imgs)

        def synchronize_between_processes(self):
            def merge(img_ids, eval_imgs):
                all_img_ids = [img_ids]
                all_eval_imgs = [eval_imgs]

                merged_img_ids = []
                for p in all_img_ids:
                    merged_img_ids.extend(p)

                merged_eval_imgs = []
                for p in all_eval_imgs:
                    merged_eval_imgs.append(p)

                merged_img_ids = numpy.array(merged_img_ids)
                merged_eval_imgs = numpy.concatenate(merged_eval_imgs, 2)

                # keep only unique (and in sorted order) images
                merged_img_ids, idx = numpy.unique(merged_img_ids, return_index=True)
                merged_eval_imgs = merged_eval_imgs[..., idx]

                return merged_img_ids, merged_eval_imgs

            def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
                img_ids, eval_imgs = merge(img_ids, eval_imgs)
                img_ids = list(img_ids)
                eval_imgs = list(eval_imgs.flatten())

                coco_eval.evalImgs = eval_imgs
                coco_eval.params.imgIds = img_ids
                coco_eval._paramsEval = copy.deepcopy(coco_eval.params)

            # for iou_type in self.iou_types:
            self.eval_imgs["bbox"] = numpy.concatenate(self.eval_imgs["bbox"], 2)
            create_common_coco_eval(self.coco_eval["bbox"], self.img_ids, self.eval_imgs["bbox"])

        def accumulate(self):
            for coco_eval in self.coco_eval.values():
                coco_eval.accumulate()

        def summarize(self):
            for iou_type, coco_eval in self.coco_eval.items():
                print("IoU metric: {}".format(iou_type))
                coco_eval.summarize()

        def convert_to_xywh(self, boxes):
            xmin, ymin, xmax, ymax = boxes.unbind(1)
            return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

        def prepare(self, predictions):
            coco_results = []
            for original_id, prediction in predictions.items():
                if len(prediction) == 0:
                    continue

                boxes = prediction["boxes"]
                boxes = self.convert_to_xywh(boxes).tolist()
                scores = prediction["scores"].tolist()
                labels = prediction["labels"].tolist()

                coco_results.extend(
                    [
                        {
                            "image_id": original_id,
                            "category_id": labels[k],
                            "bbox": box,
                            "score": scores[k],
                        }
                        for k, box in enumerate(boxes)
                    ]
                )
            return coco_results

    ann_file = os.path.join(data_path, "annotations/instances_val2017.json")
    coco_evaluator = CocoEvaluator(COCO(ann_file), ["bbox"])

    for image, label in tqdm.tqdm(data_loader):
        image = image.numpy()
        # image=(image.astype('float32')-numpy.array([123.675, 116.28, 103.53],'float32').reshape(1,3,1,1))/numpy.array([58.395, 57.12, 57.375],'float32').reshape(1,3,1,1)
        data = {"input": image}
        runtime.set_input(**data)
        runtime.run()
        out_logits = runtime.get_output(0).asnumpy()
        out_bbox = runtime.get_output(1).asnumpy()
        target_sizes = torch.stack([t["orig_size"] for t in label], dim=0)
        out_logits = torch.from_numpy(out_logits)
        out_bbox = torch.from_numpy(out_bbox)
        prob = torch.nn.functional.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)
        x_c, y_c, w, h = out_bbox.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        boxes = torch.stack(b, dim=-1)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

        res = {target["image_id"].item(): output for target, output in zip(label, results)}
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    return coco_evaluator


path = os.path.join(root_path, model_name, "origin_mod.json")
if os.path.exists(path):
    mod = None
    params = None
else:
    x = torch.randn([1, 3, 800, 800])
    scripted_model = torch.jit.load(
        "/data/zhaojinxi/Documents/quantize_result/detr_r50/detr_r50.pt"
    )
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
