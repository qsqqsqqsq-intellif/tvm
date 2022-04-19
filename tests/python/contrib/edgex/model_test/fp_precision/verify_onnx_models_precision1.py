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

    path_prefix = "/data/share/demodels-lfs/onnx/"

    model_config = {
        "framework": "onnx",
        "input_names": ["data"],
        "input_shapes": [(1, 3, 224, 224)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "resnet50-v1/resnet50-v1-7.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    # only for test: pass
    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["data"],
    #     "input_shapes": [(1, 3, 224, 224)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "mean_scale",
    #     "means": (0, 0, 0),
    #     "scales": (255, 255, 255),
    #     "model_file": path_prefix + "resnet50-v1/resnet50-v1-7.onnx",
    #     "dataset_dir": path_prefix + "resnet50-v1/img/",
    # }
    # verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["data"],
        "input_shapes": [(1, 3, 224, 224)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "resnet50-v2/resnet50-v2-7.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["input:0"],
        "input_shapes": [(1, 3, 224, 224)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "mobilenet-v1/mobilenet_v1_1.0_224.onnx",
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
        "model_file": path_prefix + "mobilenet-v2/mobilenetv2-7.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["input.1"],
        "input_shapes": [(1, 3, 224, 224)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "mobilenet-v2_torchvision/mobilenetv2_torchvision.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["input_image"],
        "input_shapes": [(1, 3, 224, 224)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "mobilenet-v3-small/mobilenetv3_small_new.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["images"],
        "input_shapes": [(1, 3, 640, 640)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "yolov5_ultralytics/yolov5s.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    # only for test the different opset! pass
    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["images"],
    #     "input_shapes": [(1, 3, 640, 640)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "yolov5_ultralytics/yolov5s_opset13.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["data"],
        "input_shapes": [(1, 3, 224, 224)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "resnet18-v1/resnet18-v1-7.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["data"],
        "input_shapes": [(1, 3, 224, 224)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "resnet18-v2/resnet18-v2-7.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["data"],
        "input_shapes": [(1, 3, 224, 224)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "resnet34-v1/resnet34-v1-7.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["data"],
        "input_shapes": [(1, 3, 224, 224)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "resnet34-v2/resnet34-v2-7.onnx",
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
        "model_file": path_prefix + "resnet50-caffe2-v1/resnet50-caffe2-v1-9.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["data"],
        "input_shapes": [(1, 3, 224, 224)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "resnet50-v2/resnet50-v2-7.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["data"],
        "input_shapes": [(1, 3, 224, 224)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "resnet101-v1/resnet101-v1-7.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["data"],
        "input_shapes": [(1, 3, 224, 224)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "resnet101-v2/resnet101-v2-7.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["data"],
        "input_shapes": [(1, 3, 224, 224)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "resnet152-v1/resnet152-v1-7.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["data"],
        "input_shapes": [(1, 3, 224, 224)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "resnet152-v2/resnet152-v2-7.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["input_1"],
        "input_shapes": [(1, 224, 224, 3)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "mobilenetv2-quant/mobilenet.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["input.1"],
        "input_shapes": [(1, 3, 224, 224)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "resnext50_32x4d_torchvision/resnext50_32x4d.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)
