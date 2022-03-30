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
import tflite
import tensorflow
import tvm
from tvm.contrib.edgex.utils import verify_model_precision

if __name__ == "__main__":
    path_prefix = "../../../../../../../demodels-lfs/tflite/"
    # path_prefix = "../demodels-lfs/tflite/"

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["serving_default_inputs:0"],
    #     "input_shapes": [(1, 224, 224, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/mobilenet_v3_large_100_224_classification/lite-model_imagenet_mobilenet_v3_large_100_224_classification_5_default_1.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["serving_default_inputs:0"],
    #     "input_shapes": [(1, 224, 224, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/mobilenet_v3_small_100_224_classification/lite-model_imagenet_mobilenet_v3_small_100_224_classification_5_default_1.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["serving_default_inputs:0"],
    #     "input_shapes": [(1, 224, 224, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/mobilenet_v3_small_075_224_classification/lite-model_imagenet_mobilenet_v3_small_075_224_classification_5_default_1.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    model_config = {
        "framework": "tflite",
        "input_names": ["input"],
        "input_shapes": [(1, 331, 331, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "Visual/nasnet_large/nasnet_large_1_default_1.tflite",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "tflite",
        "input_names": ["input"],
        "input_shapes": [(1, 224, 224, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "Visual/nasnet_mobile/nasnet_mobile_1_default_1.tflite",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "tflite",
        "input_names": ["serving_default_input_2:0"],
        "input_shapes": [(1, 224, 224, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix
        + "Visual/plant-disease/lite-model_plant-disease_default_1.tflite",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "tflite",
        "input_names": ["input"],
        "input_shapes": [(1, 299, 299, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "Visual/resnet_v2_101/resnet_v2_101_1_default_1.tflite",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "tflite",
        "input_names": ["Placeholder"],
        "input_shapes": [(1, 224, 224, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "Visual/squeezenet/squeezenet_1_default_1.tflite",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "tflite",
        "input_names": ["serving_default_input_1:0"],
        "input_shapes": [(1, 224, 224, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "Visual/resnet50/resnet50.tflite",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input_placeholder"],
    #     "input_shapes": [(1, 224, 224, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/seresnet101/se_resnet101.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input_placeholder"],
    #     "input_shapes": [(1, 224, 224, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/se_resnext50/se_resnext50.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input_placeholder"],
    #     "input_shapes": [(1, 224, 224, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/se_resnet50/se_resnet50.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    model_config = {
        "framework": "tflite",
        "input_names": ["input"],
        "input_shapes": [(1, 224, 224, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "Visual/resnet101/resnet101_fp32_pretrained_model.tflite",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "tflite",
        "input_names": ["Placeholder"],
        "input_shapes": [(1, 224, 224, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "Visual/vgg16/vgg16-20160129.tflite",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "tflite",
        "input_names": ["input"],
        "input_shapes": [(1, 224, 224, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "Visual/vgg_19/frozen_inference_graph.tflite",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "tflite",
        "input_names": ["input"],
        "input_shapes": [(1, 224, 224, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix
        + "Visual/pnasnet-5_mobile_2017_12_13/frozen_inference_graph.tflite",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "tflite",
        "input_names": ["data"],
        "input_shapes": [(1, 62, 62, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "Visual/age_gender_recognition/age_gender_recognition.tflite",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "tflite",
        "input_names": ["serving_default_actual_input_1:0"],
        "input_shapes": [(1, 128, 128, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "Visual/anti_spoof_mn3/anti_spoof_mn3.tflite",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "tflite",
        "input_names": ["serving_default_data:0"],
        "input_shapes": [(1, 112, 112, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix
        + "Visual/face_recognition_resnet100_arcface_onnx/face_recognition_resnet100_arcface_onnx.tflite",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "tflite",
        "input_names": ["serving_default_data:0"],
        "input_shapes": [(1, 112, 112, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "Visual/face_recognizer_fast/face_recognizer_fast.tflite",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "tflite",
        "input_names": ["input"],
        "input_shapes": [(1, 224, 224, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "Visual/nsfw/nsfw.tflite",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "tflite",
        "input_names": ["serving_default_input_1:0"],
        "input_shapes": [(1, 32, 32, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "Visual/open_closed_eye/open_closed_eye.tflite",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "tflite",
        "input_names": ["0"],
        "input_shapes": [(1, 160, 80, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix
        + "Visual/person_attributes_recognition_crossroad/person_attributes_recognition_crossroad.tflite",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["inputs"],
    #     "input_shapes": [(1, 256, 128, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/person_reidentification/person_reidentification.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    model_config = {
        "framework": "tflite",
        "input_names": ["serving_default_input:0"],
        "input_shapes": [(1, 256, 128, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "Visual/person_reid_youtu/person_reid_youtu.tflite",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "tflite",
        "input_names": ["serving_default_input:0"],
        "input_shapes": [(1, 72, 72, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix
        + "Visual/vehicle_attributes_recognition_barrier/vehicle_attributes_recognition_barrier.tflite",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)
