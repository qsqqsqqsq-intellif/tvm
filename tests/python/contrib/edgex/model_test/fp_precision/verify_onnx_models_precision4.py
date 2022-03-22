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
        "input_names": ["input_1", "image_shape"],
        "input_shapes": [(1, 3, 416, 416), (1, 2)],
        "input_dtypes": ["float32", "float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "yolov3/yolov3-10.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["input_1", "image_shape"],
        "input_shapes": [(1, 3, 416, 416), (1, 2)],
        "input_dtypes": ["float32", "float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "tiny-yolov3/tiny-yolov3-11.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["input_1:0"],
        "input_shapes": [(1, 416, 416, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "yolov4/yolov4.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["input/input_data:0"],
        "input_shapes": [(1, 416, 416, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "yolov3-lite/model_float32.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["input/input_data:0"],
        "input_shapes": [(1, 416, 416, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "yolov3-nano/model_float32.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["000_net"],
        "input_shapes": [(1, 3, 416, 416)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "yolov4-Snapsort/yolov4_GIX_best-416.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["000_net"],
        "input_shapes": [(1, 3, 416, 416)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "tiny-yolov4-Snapsort/yolov4-tiny_GIX-416.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["data"],
        "input_shapes": [(1, 3, 800, 800)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "DUC/ResNet101-DUC-7.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["ImageTensor:0"],
        "input_shapes": [(1, 512, 512, 3)],
        "input_dtypes": ["uint8"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "deeplabv3/deeplabv3_mnv2_pascal_train_aug.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["ImageTensor:0"],
        "input_shapes": [(1, 512, 512, 3)],
        "input_dtypes": ["uint8"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "deeplabv3/deeplabv3_mnv2_pascal_train_aug.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["x.1"],
        "input_shapes": [(1, 3, 512, 512)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "deeplabv3plus/best_model.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["x.1"],
        "input_shapes": [(1, 3, 512, 512)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "PAN/best_model.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["x.1"],
        "input_shapes": [(1, 3, 512, 512)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "UnetPlusPlus/best_model.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["input"],
        "input_shapes": [(1, 3, 512, 512)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "BiSeNet/my_param.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["Input:0"],
        "input_shapes": [(1, 256, 256, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "mobile-deeplabv3-plus/deeplabv3_plus_mnv2_aspp.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["Input:0"],
        "input_shapes": [(1, 256, 256, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "mobile-deeplabv3-plus/deeplabv3_plus_mnv3.onnx",
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
        "model_file": path_prefix + "mv3_detection/mnv3_detection_opt.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["input_1_orig"],
        "input_shapes": [(1, 96, 96, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "OneClassAnomalyDetection/weights.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["input"],
        "input_shapes": [(1, 3, 112, 112)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "pfld-106/pfld-106-v3.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["input0"],
        "input_shapes": [(1, 3, 640, 640)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "retinaface/retinaface_640x640_opt.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)
