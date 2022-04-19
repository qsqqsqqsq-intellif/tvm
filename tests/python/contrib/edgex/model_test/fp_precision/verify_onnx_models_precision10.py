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

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["image"],
    #     "input_shapes": [(1, 3, 320, 320)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "person_detection_asl/person-detection-asl.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input_1"],
    #     "input_shapes": [(1, 3, 512, 512)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "person_vehicle_bike_detection_crossroad/person-vehicle-bike-detection-crossroad-1016.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["normalized_input_image_tensor"],
    #     "input_shapes": [(1, 3, 320, 320)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "spaghettinet_edgetpu/spaghettinet_edgetpu.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input"],
    #     "input_shapes": [(1, 3, 480, 640)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "text_detection_db/text_detection_db.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["image"],
    #     "input_shapes": [(1, 3, 256, 256)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "vehicle_detection/vehicle-detection-0200.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["Placeholder"],
    #     "input_shapes": [(1, 3, 300, 300)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "vehicle_license_plate_detection_barrier/vehicle-license-plate-detection-barrier-0106.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["input.1"],
        "input_shapes": [(1, 3, 550, 550)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "yolact_resnet50_fpn/yolact-resnet50-fpn.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["inputs"],
        "input_shapes": [(1, 3, 608, 608)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "YOLOF/YOLOF.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["src", "bgr"],
    #     "input_shapes": [(1, 3, 720, 1280), (1, 3, 720, 1280)],
    #     "input_dtypes": ["float32", "float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "BackgroundMattingV2/BackgroundMattingV2.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["sub_2"],
    #     "input_shapes": [(1, 240, 320, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "BodyPix/BodyPix.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["input_1:0"],
        "input_shapes": [(1, 200, 400, 3)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "DeeplabV3_plus/DeeplabV3_plus.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input_1:0"],
    #     "input_shapes": [(1, 256, 512, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "ERFNet/ERFNet.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input.1"],
    #     "input_shapes": [(1, 3, 192, 384)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Fast_SCNN/Fast_SCNN.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["x"],
    #     "input_shapes": [(1, 3, 192, 192)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "human_segmentation_pphumanseg/human_segmentation_pphumanseg.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input_1:0"],
    #     "input_shapes": [(1, 96, 160,3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "MediaPipe_Meet_Segmentation/MediaPipe_Meet_Segmentation.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input"],
    #     "input_shapes": [(1, 3, 224, 224)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "MODNet/MODNet.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input_1"],
    #     "input_shapes": [(1, 3, 240, 320)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "SUIM_Net/SUIM_Net.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["serving_default_input_2:0"],
    #     "input_shapes": [(1, 3, 512, 512)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "vision_segmentation_default_argmax/models_edgetpu_checkpoint_and_tflite_vision_segmentation-edgetpu_tflite_default_argmax.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["serving_default_input_2:0"],
    #     "input_shapes": [(1, 3, 512, 512)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "vision_segmentation_fused_argmax/models_edgetpu_checkpoint_and_tflite_vision_segmentation-edgetpu_tflite_fused_argmax.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["data:0"],
    #     "input_shapes": [(1, 62, 62, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "age_gender_recognition/age_gender_recognition.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)
