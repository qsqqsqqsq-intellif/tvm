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
        "input_names": ["input.48"],
        "input_shapes": [(1, 3, 224, 224)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "CvT/CvT-13-224x224-IN-1k.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    # mismatch
    model_config = {
        "framework": "onnx",
        "input_names": ["data0.1", "data1.1"],
        "input_shapes": [(1, 3, 480, 640), (1, 3, 480, 640)],
        "input_dtypes": ["float32", "float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "loftr/indoor_ds_script_sub.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["input.1"],
        "input_shapes": [(1, 3, 256, 256)],
        "input_dtypes": ["float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "conformer/conformer.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    # mismatch
    model_config = {
        "framework": "onnx",
        "input_names": ["kpts0", "1", "2", "kpts1", "6", "7"],
        "input_shapes": [
            (1, 1000, 2),
            (1, 1000),
            (1, 256, 1000),
            (1, 1000, 2),
            (1, 1000),
            (1, 256, 1000),
        ],
        "input_dtypes": ["float32", "float32", "float32", "float32", "float32", "float32"],
        "layout": "NCHW",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "SuperGlue/superglue_v6.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["input"],
        "input_shapes": [(1, 3, 224, 224)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "DINO/dino_deits8.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["input"],
        "input_shapes": [(1, 3, 224, 224)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "DINO/dino_vitb8.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["x"],
        "input_shapes": [(1, 3, 224, 224)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "CaiT/cait_S24_224.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["x"],
        "input_shapes": [(1, 3, 224, 224)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "DeiT/deit_base_distilled_patch16_224.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    model_config = {
        "framework": "onnx",
        "input_names": ["x"],
        "input_shapes": [(1, 3, 224, 224)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "PatchConvnet/S60.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)

    # mismatch
    model_config = {
        "framework": "onnx",
        "input_names": ["x"],
        "input_shapes": [(1, 3, 224, 224)],
        "input_dtypes": ["float32"],
        "layout": "NHWC",
        "preproc_method": "pass_through",
        "model_file": path_prefix + "ResMLP/resmlp_S24_dist.onnx",
        "dataset_dir": None,
    }
    verify_model_precision(model_config)
