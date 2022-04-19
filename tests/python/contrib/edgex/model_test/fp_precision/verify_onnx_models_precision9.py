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
    #     "input_names": ["input:0"],
    #     "input_shapes": [(1, 256, 256, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "AnimeGANv2/model_float32.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["src", "bgr"],
    #     "input_shapes": [(1, 3, 720, 1280), (1, 3, 720, 1280)],
    #     "input_dtypes": ["float32", "float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "backgroudmatv2/backgroundmattingv2_hd_720x1280.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["input"],
        "input_shapes": [(1, 3, 256, 256)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "gray2rgb/G_8_gray2rgb_256.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input"],
    #     "input_shapes": [(1, 3, 120, 160)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "YuNet/face_detection_yunet_120x160.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input_image"],
    #     "input_shapes": [(1, 3, 1024, 1024)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "DDRNet23_slim_seg/DDRNet23_slim_seg.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input_image"],
    #     "input_shapes": [(1, 3, 1024, 1024)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "DDRNet23_seg/DDRNet23_seg.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input_image"],
    #     "input_shapes": [(1, 3, 520, 520)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "HRNet_W48_ocr_seg/HRNet_W48_OCR_coco.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["images"],
        "input_shapes": [(1, 3, 640, 640)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "yolox_series/yolox_s.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input_image"],
    #     "input_shapes": [(1, 3, 1280, 1280)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "yolor/yolor_p6.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["images"],
    #     "input_shapes": [(1, 3, 640, 640)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "yolop/yolop-640-640.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input_image"],
    #     "input_shapes": [(1, 3, 800, 1216)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "FCOS/fcos_imprv_R_50_FPN_1x.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input_image"],
    #     "input_shapes": [(1, 3, 1024, 1024)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "faceboxes/faceboxes.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input"],
    #     "input_shapes": [(1, 3, 320, 320)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "CascadeTableNet/CascadeTableNet.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["samples"],
        "input_shapes": [(1, 3, 256, 256)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "detr/detr.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["serving_default_images:0"],
    #     "input_shapes": [(1, 320, 320, 3)],
    #     "input_dtypes": ["uint8"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "efficientdet_lite/efficientdet_lite.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["rgb_to_grayscale_1:0"],
    #     "input_shapes": [(200, 32, 32, 1)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "knift/knift.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["normalized_input_image_tensor"],
    #     "input_shapes": [(1, 3, 192, 192)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "object_detection_mobile_object_localizer/object_detection_mobile_object_localizer.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["data"],
    #     "input_shapes": [(1, 3, 384, 672)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "pedestrian_detection_adas/pedestrian-detection-adas-0002.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["data"],
    #     "input_shapes": [(1, 3, 384, 672)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "pedestrian_and-vehicle_detector_adas/pedestrian-and-vehicle-detector-adas-0001.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["image"],
    #     "input_shapes": [(1, 3, 512, 512)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "person_detection/person-detection-0202.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)
