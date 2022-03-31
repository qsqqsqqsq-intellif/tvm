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
from tvm.contrib.edgex.testing import check_edgex_relay_build


def verify_conv3d(input_shape, input_dtype, weight_shape, weight_dtype, groups=1, **conv_attrs):
    relay_params = {}
    weight_data = tvm.nd.array(np.random.randint(-64, 64, weight_shape).astype(weight_dtype))
    input_data = tvm.nd.array(np.random.randint(-64, 64, input_shape).astype(input_dtype))
    relay_params["weight"] = weight_data
    relay_params["input"] = input_data

    x = relay.var("input", dtype=input_dtype, shape=input_shape)
    w = relay.var("weight", dtype=weight_dtype, shape=weight_shape)
    y = relay.nn.conv3d(x, w, groups=groups, **conv_attrs)
    mod = IRModule.from_expr(relay.Function([x, w], y))
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.FuseOps(fuse_opt_level=0)(mod)
    check_edgex_relay_build(mod, params=relay_params, check_cpu=True, test_fused=True)


def test_conv3d_end2end_d1_nontile():
    verify_conv3d(
        input_shape=[1, 3, 1, 128, 128],
        input_dtype="uint8",
        weight_shape=[16, 3, 3, 3, 3],
        weight_dtype="int8",
        strides=[1, 1, 1],
        padding=[1, 1, 1],
        dilation=[1, 1, 1],
        channels=16,
        kernel_size=[3, 3, 3],
        out_dtype="int32",
        data_layout="NCDHW",
        kernel_layout="OIDHW",
    )


@pytest.mark.skip
def test_conv3d_end2end_d1_tile():
    verify_conv3d(
        input_shape=[1, 3, 1, 224, 224],
        input_dtype="uint8",
        weight_shape=[16, 3, 3, 3, 3],
        weight_dtype="uint8",
        strides=[1, 1, 1],
        padding=[1, 1, 1],
        dilation=[1, 1, 1],
        channels=16,
        kernel_size=[3, 3, 3],
        out_dtype="int32",
        data_layout="NCDHW",
        kernel_layout="OIDHW",
    )


@pytest.mark.skip
def test_conv3d_end2end_d3_nontile():
    verify_conv3d(
        input_shape=[1, 3, 3, 128, 128],
        input_dtype="uint8",
        weight_shape=[16, 3, 3, 3, 3],
        weight_dtype="int8",
        strides=[1, 1, 1],
        padding=[1, 1, 1],
        dilation=[1, 1, 1],
        channels=16,
        kernel_size=[3, 3, 3],
        out_dtype="int32",
        data_layout="NCDHW",
        kernel_layout="OIDHW",
    )


if __name__ == "__main__":
    pytest.main([__file__])
