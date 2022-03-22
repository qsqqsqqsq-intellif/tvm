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
    #     "input_names": ["imgs_ph"],
    #     "input_shapes": [(1, 512, 1024, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/ENet/ENet.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    model_config = {
        "framework": "tflite",
        "input_names": ["sub_7"],
        "input_shapes": [(1, 257, 257, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "Visual/deeplabv3/deeplabv3_1_default_1.tflite",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["image_tensor"],
    #     "input_shapes": [(1, 256, 256, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/mask_rcnn_inception_v2_coco/mask_rcnn_inception_v2_coco_256_weight_quant.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["i"],
    #     "input_shapes": [(1, 320, 320, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/NanoDet/model_float32.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["x"],
    #     "input_shapes": [(1, 256, 256, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/u2netp_256x256_float32/u2netp_256x256_float32.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input"],
    #     "input_shapes": [(1, 512, 512, 4)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/hair_segmentation/hair_segmentation_512x512_float32.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["serving_default_input_1:0"],
    #     "input_shapes": [(1, 200, 400, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/deeplab_v3_plus_mnv2_decoder/model_float32.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["data"],
    #     "input_shapes": [(1, 512, 896, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/road-segmentation-adas/road-segmentation-adas.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["data"],
    #     "input_shapes": [(1, 256, 256, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/Selfie_Segmentation/Selfie_Segmentation.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input"],
    #     "input_shapes": [(1, 120, 160, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/YuNet/YuNet.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["sub_2"],
    #     "input_shapes": [(1, 240, 320, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/BodyPix/BodyPix.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["serving_default_input_1:0"],
    #     "input_shapes": [(1, 200, 400, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/DeeplabV3_plus/DeeplabV3_plus.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input_1"],
    #     "input_shapes": [(1, 256, 512, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/ERFNet/ERFNet.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["serving_default_input_1:0"],
    #     "input_shapes": [(1, 192, 384, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/Fast_SCNN/Fast_SCNN.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["serving_default_x:0"],
    #     "input_shapes": [(1, 192, 192, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/human_segmentation_pphumanseg/human_segmentation_pphumanseg.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input_1"],
    #     "input_shapes": [(1, 96, 160, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/MediaPipe_Meet_Segmentation/MediaPipe_Meet_Segmentation.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input"],
    #     "input_shapes": [(1, 128, 128, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/MODNet/MODNet.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input_1"],
    #     "input_shapes": [(1, 240, 320, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/SUIM_Net/SUIM_Net.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["inputs"],
    #     "input_shapes": [(1, 512, 512, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/vision_segmentation_default_argmax/vision_segmentation_default_argmax.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["inputs"],
    #     "input_shapes": [(1, 512, 512, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/vision_segmentation_fused_argmax/vision_segmentation_fused_argmax.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)
