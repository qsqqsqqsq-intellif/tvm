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
    #     "input_names": ["input_1"],
    #     "input_shapes": [(1, 256, 256, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/TextBoxes++/TextBoxes++.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input"],
    #     "input_shapes": [(1, 480, 640, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/text_detection_db/text_detection_db.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["serving_default_image:0"],
    #     "input_shapes": [(1, 256, 256, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix
    #     + "Visual/vehicle_detection/vehicle-detection-0200.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["serving_default_Placeholder:0"],
    #     "input_shapes": [(1, 300, 300, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/vehicle_license_plate_detection-barrier/vehicle-license-plate-detection-barrier-0106.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input_1"],
    #     "input_shapes": [(1, 550, 550, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/yolact_edge/yolact_edge.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["serving_default_input_1:0"],
    #     "input_shapes": [(1, 550, 550, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/yolact_resnet50_fpn/yolact-resnet50-fpn.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["image"],
    #     "input_shapes": [(1, 608, 608, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/YOLOF/YOLOF.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["serving_default_input_1:0"],
    #     "input_shapes": [(1, 192, 192, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/3DMPPE_POSENET/3DMPPE_POSENET.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input"],
    #     "input_shapes": [(1, 256, 256, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/BlazeFace/BlazeFace.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input_res1:0"],
    #     "input_shapes": [(1, 256, 256, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/EfficientPose/EfficientPose.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input_1"],
    #     "input_shapes": [(1, 192, 192, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/FaceMesh/FaceMesh.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["prior_based_hand/input:0"],
    #     "input_shapes": [(1, 128, 128, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/face_detection_adas/face_detection_adas.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["images"],
    #     "input_shapes": [(1, 160, 160, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/Face_Landmark/Face_Landmark.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["serving_default_data:0"],
    #     "input_shapes": [(1, 300, 300, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/Face_Mask_Detection/Face_Mask_Detection.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input_1:0"],
    #     "input_shapes": [(1, 256, 256, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/hand_recrop/hand_recrop.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input_image_tensor:0"],
    #     "input_shapes": [(1, 128, 128, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/Head_Pose_Estimation/Head_Pose_Estimation.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["serving_default_data:0"],
    #     "input_shapes": [(1, 384, 672, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/head_pose_estimation_adas/head_pose_estimation_adas.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["serving_default_data:0"],
    #     "input_shapes": [(1, 180, 320, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/Human_Pose_Estimation_3D/Human_Pose_Estimation_3D.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input_1:0"],
    #     "input_shapes": [(1, 64, 64, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/Iris_Landmark/Iris_Landmark.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input_1:0"],
    #     "input_shapes": [(1, 256, 256, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/Minimal_Hand/Minimal_Hand.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input"],
    #     "input_shapes": [(1, 256, 256, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/MobileHumanPose/MobileHumanPose.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["image:0"],
    #     "input_shapes": [(1, 368, 432, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/Mobilenetv2_Pose_Estimation/Mobilenetv2_Pose_Estimation.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["inputs"],
    #     "input_shapes": [(1, 224, 224, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/Mobilenetv3_Pose_Estimation/Mobilenetv3_Pose_Estimation.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input:0"],
    #     "input_shapes": [(1, 192, 192, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/MoveNet/MoveNet.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input:0"],
    #     "input_shapes": [(1, 192, 192, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/MoveNet_MultiPose/MoveNet_MultiPose.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["image:0"],
    #     "input_shapes": [(1, 368, 432, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/Posenet/Posenet.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["serving_default_input_1:0"],
    #     "input_shapes": [(1, 224, 224, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/RetinaFace/RetinaFace.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["image"],
    #     "input_shapes": [(1, 368, 656, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/tf_pose_estimation/tf_pose_estimation.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    model_config = {
        "framework": "tflite",
        "input_names": ["serving_default_data:0"],
        "input_shapes": [(1, 60, 60, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "Visual/WHENet/WHENet.tflite",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)
