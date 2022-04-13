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
import tvm
import pytest
from tvm import relay
import numpy as np
from tvm.ir.module import IRModule
import tvm.testing
import tvm.contrib.edgex
from tvm.contrib.edgex.relay.transform import (
    ConvertDepthwiseConv2D,
)
from tvm.contrib.edgex.testing import check_edgex_relay_build
from tvm.relay.quantization.relay_ops import round_right_shift
from tvm.relay.build_module import bind_params_by_name


def do_test_single_conv2d(
    input_shape, input_dtype, weight_shape, weight_dtype, groups=1, **conv_attrs
):
    x = relay.var("input", dtype=input_dtype, shape=input_shape)
    w = relay.var("weight", dtype=weight_dtype, shape=weight_shape)
    y = relay.nn.conv2d(x, w, groups=groups, **conv_attrs)
    mod = IRModule.from_expr(relay.Function([x, w], y))
    mod = relay.transform.InferType()(mod)
    relay_params = {}
    weight_data = tvm.nd.array(np.random.randint(-64, 64, weight_shape).astype(weight_dtype))
    relay_params["weight"] = weight_data
    if groups > 1:
        mod = relay.transform.DefuseOps()(mod)
        func_with_params = bind_params_by_name(mod["main"], relay_params)
        mod, relay_params = ConvertDepthwiseConv2D()(mod, relay_params)
        mod = relay.transform.FoldConstant()(mod)
    mod = relay.transform.FuseOps(fuse_opt_level=0)(mod)
    check_edgex_relay_build(mod, params=relay_params, check_cpu=True, test_fused=True)


def do_test_quantized_conv2d(input_shape, input_dtype, weight_shape, weight_dtype, **conv_attrs):
    x = relay.var("input", shape=input_shape, dtype=input_dtype)
    w = relay.var("weight", shape=weight_shape, dtype=weight_dtype)
    c_o = weight_shape[0]  # assume OIHW
    bias = relay.var("bias", shape=[c_o], dtype="int32")
    mulnorm = relay.var("mulnorm", shape=[c_o, 1, 1], dtype="int64")
    shiftnorm = relay.var("shiftnorm", shape=[c_o, 1, 1], dtype="int64")
    y = relay.nn.conv2d(x, w, **conv_attrs)
    y = relay.nn.bias_add(y, bias)
    y = relay.cast(y, "int64")
    y = relay.multiply(y, mulnorm)
    y = round_right_shift(y, shiftnorm)
    y = relay.clip(y, -128, 127)
    y = relay.cast(y, "int8")
    y = relay.nn.relu(y)
    attrs = tvm.ir.make_node("DictAttrs", **{"Primitive": 1})
    fused_func = relay.Function([x, w, bias, mulnorm, shiftnorm], y, attrs=attrs)

    def wrap_relay_fused_function(relay_function):
        new_args = [relay.Var(p.name_hint, p.type_annotation) for p in relay_function.params]
        return relay.Function(new_args, relay.Call(relay_function, new_args))

    function = wrap_relay_fused_function(fused_func)
    mod = tvm.IRModule.from_expr(function)
    mod = relay.transform.InferType()(mod)

    relay_params = {}
    relay_params["weight"] = np.random.randint(0, 5, weight_shape).astype("int8")
    relay_params["bias"] = np.random.randint(-128, 127, [c_o]).astype("int32")
    relay_params["mulnorm"] = np.random.randint(0, 6, [c_o, 1, 1]).astype("int64")
    relay_params["shiftnorm"] = np.random.randint(1, 3, [c_o, 1, 1]).astype("int64")
    check_edgex_relay_build(mod, params=relay_params, check_cpu=True, test_fused=True)


def test_single_conv2d_end2end():
    do_test_single_conv2d(
        input_shape=[1, 3, 224, 224],
        input_dtype="uint8",
        weight_shape=[16, 3, 7, 7],
        weight_dtype="int8",
        strides=[2, 2],
        padding=[3, 3, 3, 3],
        dilation=[1, 1],
        channels=16,
        kernel_size=[7, 7],
        out_dtype="int32",
    )


def test_single_depthwise_conv2d_end2end():
    do_test_single_conv2d(
        input_shape=[1, 3, 32, 32],
        input_dtype="uint8",
        weight_shape=[3, 1, 1, 1],
        weight_dtype="int8",
        groups=3,
        strides=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        kernel_size=[1, 1],
        out_dtype="int32",
    )
    # large group number
    do_test_single_conv2d(
        input_shape=[1, 960, 7, 7],
        input_dtype="uint8",
        weight_shape=[960, 1, 3, 3],
        weight_dtype="int8",
        groups=960,
        strides=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        kernel_size=[3, 3],
        out_dtype="int32",
    )


def test_single_group_conv2d_end2end():
    do_test_single_conv2d(
        input_shape=[1, 32, 16, 16],
        input_dtype="uint8",
        weight_shape=[32, 2, 1, 1],
        weight_dtype="int8",
        strides=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=16,
        kernel_size=[1, 1],
        out_dtype="int32",
    )


def test_single_conv2d_oc17_end2end():
    do_test_single_conv2d(
        input_shape=[1, 3, 224, 224],
        input_dtype="uint8",
        weight_shape=[17, 3, 7, 7],
        weight_dtype="int8",
        strides=[2, 2],
        padding=[3, 3, 3, 3],
        dilation=[1, 1],
        kernel_size=[7, 7],
        out_dtype="int32",
    )


def test_single_conv2d_ic17_end2end():
    do_test_single_conv2d(
        input_shape=[1, 17, 56, 56],
        input_dtype="uint8",
        weight_shape=[17, 17, 7, 7],
        weight_dtype="int8",
        strides=[2, 2],
        padding=[3, 3, 3, 3],
        dilation=[1, 1],
        kernel_size=[7, 7],
        out_dtype="int32",
    )


def test_quantized_conv2d_end2end():
    do_test_quantized_conv2d(
        input_shape=[1, 3, 224, 224],
        input_dtype="uint8",
        weight_shape=[16, 3, 7, 7],
        weight_dtype="int8",
        strides=[2, 2],
        kernel_size=[7, 7],
        padding=[3, 3, 3, 3],
        dilation=[1, 1],
        out_dtype="int32",
    )


def test_quantized_conv2d_finetuning_delta_end2end():
    do_test_quantized_conv2d(
        input_shape=[1, 16, 112, 112],
        input_dtype="uint8",
        weight_shape=[96, 16, 1, 1],
        weight_dtype="int8",
        dilation=[1, 1],
        strides=[1, 1],
        kernel_size=[1, 1],
        padding=[0, 0, 0, 0],
        out_dtype="int32",
    )


def test_single_conv2d_tile_co_end2end():
    do_test_single_conv2d(
        input_shape=[1, 3, 224, 224],
        input_dtype="uint8",
        weight_shape=[64, 3, 7, 7],
        weight_dtype="int8",
        strides=[2, 2],
        padding=[3, 3, 3, 3],
        dilation=[1, 1],
        channels=64,
        kernel_size=[7, 7],
        out_dtype="int32",
    )


def test_conv2d_nchwc_end2end():
    do_test_single_conv2d(
        input_shape=[1, 1, 224, 224, 16],
        input_dtype="uint8",
        weight_shape=[4, 1, 7, 7, 16, 16],
        weight_dtype="int8",
        strides=[2, 2],
        padding=[3, 3, 3, 3],
        dilation=[1, 1],
        channels=64,
        kernel_size=[7, 7],
        out_dtype="int32",
        data_layout="NCHW16c",
        kernel_layout="OIHW16i16o",
    )


def test_quantized_conv2d_tiling_end2end():
    do_test_quantized_conv2d(
        input_shape=[1, 17, 1024, 1024],
        input_dtype="uint8",
        weight_shape=[64, 17, 3, 3],
        weight_dtype="int8",
        strides=[2, 2],
        kernel_size=[3, 3],
        padding=[3, 0, 3, 0],
        dilation=[1, 1],
        out_dtype="int32",
    )


def test_quantized_conv2d_h1w1_end2end():
    do_test_quantized_conv2d(
        input_shape=[1, 16, 1, 1],
        input_dtype="uint8",
        weight_shape=[8, 16, 1, 1],
        weight_dtype="int8",
        strides=[1, 1],
        kernel_size=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        out_dtype="int32",
    )


@pytest.mark.skip("not done")
def test_superpoint_conv2d_bias_relu_3():
    do_test_quantized_conv2d(
        input_shape=[1, 128, 120, 160],
        input_dtype="int8",
        weight_shape=[128, 128, 3, 3],
        weight_dtype="int8",
        strides=[1, 1],
        kernel_size=[3, 3],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        out_dtype="int32",
    )


if __name__ == "__main__":
    test_single_conv2d_end2end()
    test_single_depthwise_conv2d_end2end()
    test_single_group_conv2d_end2end()
    test_quantized_conv2d_end2end()
    test_single_conv2d_oc17_end2end()
    test_single_conv2d_ic17_end2end()
    test_quantized_conv2d_finetuning_delta_end2end()
    test_single_conv2d_tile_co_end2end()
    test_conv2d_nchwc_end2end()
    test_quantized_conv2d_tiling_end2end()
    test_quantized_conv2d_h1w1_end2end()
    test_superpoint_conv2d_bias_relu_3()
