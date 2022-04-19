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
    #     "input_names": ["input"],
    #     "input_shapes": [(1, 256, 256, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "movenet/movenet_multipose_lightning_1.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input"],
    #     "input_shapes": [(1, 256, 256, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "movenet/movenet_singlepose_thunder_4.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["input_1:0"],
        "input_shapes": [(1, 224, 224, 3)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "WHENet/model_float32.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["right_eye_image:0", "left_eye_image:0", "head_pose_angles:0"],
    #     "input_shapes": [(1, 60, 60, 3), (1, 60, 60, 3), (1, 3)],
    #     "input_dtypes": ["float32", "float32", "float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "gaze-estimation-adas/model_float32.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # mismatch
    model_config = {
        "framework": "onnx",
        "input_names": ["input_res1:0"],
        "input_shapes": [(1, 256, 256, 3)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "EfficientPoseI/model_float32.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    # mismatch
    model_config = {
        "framework": "onnx",
        "input_names": ["input_res1:0"],
        "input_shapes": [(1, 368, 368, 3)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "EfficientPoseII/model_float32.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input_1_0:0"],
    #     "input_shapes": [(1, 256, 256, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "EfficientPoseI_LITE/model_float32.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input:0"],
    #     "input_shapes": [(1, 128, 128, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "BlazePose/model_float32.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input:0"],
    #     "input_shapes": [(1, 128, 128, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "BlazeFace/face_detection_front_128x128_float32.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input:0"],
    #     "input_shapes": [(1, 256, 256, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "BlazeFace/face_detection_back_256x256_float32.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["inputs:0"],
    #     "input_shapes": [(1, 256, 256, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Hand_Detection_and_Tracking/model_float32.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input.1"],
    #     "input_shapes": [(1, 3, 320, 320)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "DBFace/dbface_mbnv3_320x320.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input"],
    #     "input_shapes": [(1, 1, 224, 224)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "super-resolution/super-resolution-10.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input1"],
    #     "input_shapes": [(1, 3, 224, 224)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "style-transfer/candy-9.onnx",
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
    #     "model_file": path_prefix + "esrgan/esrgan.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input.1"],
    #     "input_shapes": [(1, 3, 384, 288)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "lite_hrnet/lite_hr.onnx",
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
    #     "model_file": path_prefix + "EnlightenGAN/enlighten.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["inputs:0"],
    #     "input_shapes": [(1, 256, 256, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "MonocularDepthEstimator-midas/midas_v2_1_small.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input.1"],
    #     "input_shapes": [(1, 3, 320, 320)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "DeblurGANv2/deblurganv2_mbnv2_320x320.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input:0"],
    #     "input_shapes": [(1, 256, 256, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Noise2Noise/model_float32.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)
