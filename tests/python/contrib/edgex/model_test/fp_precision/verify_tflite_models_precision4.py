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

    model_config = {
        "framework": "tflite",
        "input_names": ["input_1"],
        "input_shapes": [(1, 320, 320, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "Visual/DBFace/model_float32.tflite",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["dronet_net1"],
    #     "input_shapes": [(1, 608, 608, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/DroNet/DroNet.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input_1"],
    #     "input_shapes": [(1, 256, 256, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/hand_landmark/hand_landmark.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input"],
    #     "input_shapes": [(1, 640, 480, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/object_detection_3d_chair/object_detection_3d_chair.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input_1"],
    #     "input_shapes": [(1, 96, 96, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/one_class_anomaly_detection/one_class_anomaly_detection.tflite",
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
    #     "model_file": path_prefix + "Visual/SCRFD/SCRFD.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input_1"],
    #     "input_shapes": [(1, 608, 608, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/SFA3D/SFA3D.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["images"],
    #     "input_shapes": [(1, 320, 320, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/yolor_ssss_s2d/yolor_ssss_s2d.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input/input_data:0"],
    #     "input_shapes": [(1, 256, 256, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/yolov3-lite/yolov3-lite.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input/input_data:0"],
    #     "input_shapes": [(1, 256, 256, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/yolov3-nano/yolov3-nano.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["serving_default_input_1:0"],
    #     "input_shapes": [(1, 416, 416, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/yolov4/yolov4.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["data"],
    #     "input_shapes": [(1, 256, 320, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/yolov5_Face/YOLOv5_Face.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["images"],
    #     "input_shapes": [(1, 320, 320, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/yolox/yolox.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["serving_default_input_1:0"],
    #     "input_shapes": [(1, 32, 32, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/CenterFace/CenterFace.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input_1"],
    #     "input_shapes": [(1, 128, 64, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/deepsort/deepsort.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["image"],
    #     "input_shapes": [(1, 224, 224, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/weld-porosity-detection/weld-porosity-detection.tflite",
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
    #     "model_file": path_prefix + "Visual/Yolact/Yolact.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["normalized_input_image_tensor"],
    #     "input_shapes": [(1, 320, 320, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/ssd_mobilenet_v2_mnasfpn/ssd_mobilenet_v2_mnasfpn.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["inputs"],
    #     "input_shapes": [(1, 416, 416, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/yolov3/frozen_darknet_yolov3_model.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["serving_default_input:0"],
    #     "input_shapes": [(1, 320, 320, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/CascadeTableNet/CascadeTableNet.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["image_arrays"],
    #     "input_shapes": [(1, 256, 256, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/efficientDet/EfficientDet.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input_images"],
    #     "input_shapes": [(1, 320, 320, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/efficientDet_lite/efficientDet_lite.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["input_1"],
    #     "input_shapes": [(1, 320, 320, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/keras_retinanet/keras-retinanet.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["rgb_to_grayscale_1"],
    #     "input_shapes": [(200, 32, 32, 1)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/knift/knift.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["normalized_input_image_tensor"],
    #     "input_shapes": [(1, 300, 300, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/Mobilenetv2_SSDlite/Mobilenetv2-SSDlite.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["normalized_input_image_tensor"],
    #     "input_shapes": [(1, 320, 320, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/Mobilenetv3_SSD/Mobilenetv3-SSD.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["normalized_input_image_tensor:0"],
    #     "input_shapes": [(1, 192, 192, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/object_detection_mobile_object_localizer/object_detection_mobile_object_localizer.tflite",
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
    #     "model_file": path_prefix + "Visual/pedestrian_and_vehicle_detector_adas/pedestrian-and-vehicle-detector-adas-0001.tflite",
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
    #     "model_file": path_prefix + "Visual/pedestrian_detection_adas/pedestrian-detection-adas-0002.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["serving_default_image:0"],
    #     "input_shapes": [(1, 320, 320, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/person-detection_asl/person-detection-asl.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["serving_default_image:0"],
    #     "input_shapes": [(1, 512, 512, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/person_detection/person-detection-0202.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["serving_default_input_1:0"],
    #     "input_shapes": [(1, 512, 512, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/person_vehicle_bike_detection_crossroad/person-vehicle-bike-detection-crossroad-1016.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["serving_default_image:0"],
    #     "input_shapes": [(1, 320, 320, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/PP_PicoDet/PP-PicoDet.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["normalized_input_image_tensor:0"],
    #     "input_shapes": [(1, 320, 320, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/spaghettinet_edgetpu/spaghettinet_edgetpu.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["normalized_input_image_tensor"],
    #     "input_shapes": [(1, 320, 320, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/SSDlite_MobileDet_cpu/SSDlite_MobileDet_cpu.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)

    # model_config = {
    #     "framework": "tflite",
    #     "input_names": ["normalized_input_image_tensor"],
    #     "input_shapes": [(1, 320, 320, 3)],
    #     "input_dtypes": ["float32"],
    #     "layout": "NHWC",
    #     "preproc_method": "pass_through",
    #     "model_file": path_prefix + "Visual/SSDlite_MobileDet_edgetpu/SSDlite_MobileDet_edgetpu.tflite",
    #     "dataset_dir": None,
    # }
    # verify_model_precision(model_config)
