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
""" load frontend """
import tvm.relay.frontend


def get_from_framework_frontend(name):
    """ get from_xxx func from relay frontend """
    mod = tvm.relay.frontend
    framework_infer = "".join(["from_", name.lower()])
    return getattr(mod, framework_infer)


def frontend_args(input_names, input_shapes, input_dtypes, framework):
    """ unify from_xxx args """
    args = {}
    if framework in ("onnx", "tflite"):
        shape_dict = dict(zip(input_names, input_shapes))
        dtype_dict = dict(zip(input_names, input_dtypes))
        if framework == "onnx":
            args.update({"shape": shape_dict, "dtype": dtype_dict})
            args.update({"freeze_params": True})
        else:
            args.update({"shape_dict": shape_dict, "dtype_dict": dtype_dict})
    elif framework == "pytorch":
        input_infos = tuple(zip(input_names, tuple(zip(input_shapes, input_dtypes))))
        args.update({"input_infos": input_infos})
    else:
        assert 0, "{} is not supported in frontend_params!".format(framework)

    return args
