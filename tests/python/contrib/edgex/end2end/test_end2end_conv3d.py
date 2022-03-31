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
import pytest
import tvm
from tvm import relay
import numpy as np
from tvm.ir.module import IRModule
import tvm.testing
import tvm.contrib.edgex
from tvm.contrib.edgex.relay.transform import (
    ConvertDepthwiseConv2D,
)
from tvm.contrib.edgex.testing import check_edgex_relay_build
from tvm.relay.build_module import bind_params_by_name


def verify_conv3d(
    input_shape, input_dtype, weight_shape, weight_dtype, groups=1, **conv_attrs
):
    x = relay.var("input", dtype=input_dtype, shape=input_shape)
    w = relay.var("weight", dtype=weight_dtype, shape=weight_shape)
    y = relay.nn.conv3d(x, w, groups=groups, **conv_attrs)
    mod = IRModule.from_expr(relay.Function([x, w], y))
    mod = relay.transform.InferType()(mod)
    relay_params = {}
    weight_data = tvm.nd.array(np.random.randint(-64, 64, weight_shape).astype(weight_dtype))
    relay_params["weight"] = weight_data
    mod = relay.transform.FuseOps(fuse_opt_level=0)(mod)
    check_edgex_relay_build(mod, params=relay_params, check_cpu=True, test_fused=True)


def test_single_conv3d_end2end():
    verify_conv3d(
        input_shape=[1, 3, 5, 224, 224],
        input_dtype="uint8",
        weight_shape=[10, 3, 3, 3, 3],
        weight_dtype="int8",
        strides=[1, 1, 1],
        padding=[1, 1, 1],
        dilation=[1, 1, 1],
        kernel_size=[3, 3, 3],
        out_dtype="int32",
        data_layout="NCDHW",
        kernel_layout="OIDHW",
    )


if __name__ == "__main__":
    # pytest.main([__file__])
    test_single_conv3d_end2end()
