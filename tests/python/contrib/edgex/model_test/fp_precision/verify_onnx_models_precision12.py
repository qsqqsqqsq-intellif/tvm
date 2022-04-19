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
    #     "input_shapes": [(1, 192, 192, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "MoveNet/MoveNet.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["inputs:0"],
        "input_shapes": [(1, 368, 656, 3)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "tf_pose_estimation/tf_pose_estimation.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input.1", "input.4", "input.7"],
    #     "input_shapes": [(1, 3, 448, 448), (1, 3, 448, 448), (1, 3, 448, 448)],
    #     "input_dtypes": ["float32", "float32", "float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "ThreeDPoseUnityBarracuda/ThreeDPoseUnityBarracuda.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["tensor.1"],
    #     "input_shapes": [(1, 256)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Lightweight-GAN/ro_gan.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["segmap", "input.1"],
    #     "input_shapes": [(1, 36, 256, 512), (1, 36, 4, 8)],
    #     "input_dtypes": ["float32", "float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "GauGAN/gaugan.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["line", "line_draft", "hint"],
    #     "input_shapes": [(1, 1, 512, 512), (1, 1, 128, 128), (1, 4, 128, 128)],
    #     "input_dtypes": ["float32", "float32", "float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "SketchColorization/SketchColorizationModel.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input.1", "input.169"],
    #     "input_shapes": [(1, 3, 256, 256), (1, 3, 256, 256)],
    #     "input_dtypes": ["float32", "float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "CFNet/cfnet_sceneflow_256x256.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["input"],
    #     "input_shapes": [(1, 2, 240, 320)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "hitnet/eth3d_model_float32.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["style"],
    #     "input_shapes": [(1, 512)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "MobileStyleGAN/mobilestylegan_ffhq_snet.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["var"],
    #     "input_shapes": [(1, 512)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "MobileStyleGAN/mobilestylegan_ffhq_mnet.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "onnx",
    #     "input_names": ["0"],
    #     "input_shapes": [(1, 3, 224, 224)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NCHW",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "r3d_18_torchvision/r3d_18.onnx",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["0"],
        "input_shapes": [(1, 3, 16, 112, 112)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "mc3_18_torchvision/mc3_18.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["0"],
        "input_shapes": [(1, 3, 16, 112, 112)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "r2plus1d_18_torchvision/r2plus1d_18.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)
