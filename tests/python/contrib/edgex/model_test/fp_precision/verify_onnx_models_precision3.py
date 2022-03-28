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
import onnx, onnxruntime
import tvm
from tvm.contrib.edgex.utils import verify_model_precision


if __name__ == "__main__":

    path_prefix = "../../../../../../../demodels-lfs/onnx/"

    model_config = {
        "framework": "onnx",
        "input_names": ["data_0"],
        "input_shapes": [(1, 3, 224, 224)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "inception-v2/inception-v2-9.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["gpu_0/data_0"],
        "input_shapes": [(1, 3, 224, 224)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "shufflenet-v1/shufflenet-9.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["input"],
        "input_shapes": [(1, 3, 224, 224)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "shufflenet-v2/shufflenet-v2-10.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["gpu_0/data_0"],
        "input_shapes": [(1, 3, 224, 224)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "zfnet512/zfnet512-9.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["Input3"],
        "input_shapes": [(1, 1, 28, 28)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "mnist/mnist-8.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["images:0"],
        "input_shapes": [(1, 224, 224, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "efficientnet-lite4/efficientnet-lite4-11.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    # Mismatched elements: 46 / 1000(4.6 %)
    # Max absolute difference: 0.09674072
    # Max relative difference: 0.07621461
    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input.1"],
    #     "input_shapes": [(1, 3, 224, 224)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "efficientnet-b0/efficientnet_b0.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # error
    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input.1"],
    #     "input_shapes": [(1, 3, 224, 224)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "ghostnet/ghostnet.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # error
    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input.1"],
    #     "input_shapes": [(1, 3, 224, 224)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "condensenet-v2/condensevetv2_a.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # error
    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input.1"],
    #     "input_shapes": [(1, 3, 224, 224)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "resnest50/resnest50.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # error
    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input_image"],
    #     "input_shapes": [(1, 3, 224, 224)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "seresnet50/seresnet50.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["input.1"],
        "input_shapes": [(1, 3, 224, 224)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "fcn_resnet50_torchvision/fcn_resnet50.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["input.1"],
        "input_shapes": [(1, 3, 520, 520)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "deeplabv3_resnet50_torchvision/deeplabv3_resnet50.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    # error
    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["image"],
    #     "input_shapes": [(1, 3, 1200, 1200)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "ssd/ssd-10.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # dynamic rank
    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["image_tensor:0"],
    #     "input_shapes": [(1, 320, 320, 3)],
    #     "input_dtypes": ["uint8"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "ssd_mobilenet_v1/ssd_mobilenet_v1_10.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # unwork
    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["image"],
    #     "input_shapes": [(3, 512, 512)],
    #     "input_dtypes": ["float32"],
    #     "layout": "CHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "FasterRCNN/FasterRCNN-10.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # unwork
    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["image"],
    #     "input_shapes": [(3, 512, 512)],
    #     "input_dtypes": ["float32"],
    #     "layout": "CHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "MaskRCNN/MaskRCNN-10.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["input"],
        "input_shapes": [(1, 3, 480, 640)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "retinanet/retinanet-9.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["image"],
        "input_shapes": [(1, 3, 416, 416)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "tiny-yolov2/tinyyolov2-8.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["input.1"],
        "input_shapes": [(1, 3, 416, 416)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "yolov2/yolov2-coco-9.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)
