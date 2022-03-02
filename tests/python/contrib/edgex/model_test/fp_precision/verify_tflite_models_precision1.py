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

    model_config = {
        "framework": "tflite",
        "input_names": ["input"],
        "input_shapes": [(1, 224, 224, 3)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix
        + "Visual/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224_1_default_1.tflite",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)
