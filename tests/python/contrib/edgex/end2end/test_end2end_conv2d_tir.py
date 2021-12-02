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
from tvm import tir
from tvm.contrib.edgex.edgex import build_config_nnp
import tvm.testing
from tvm.script import tir as T
from tvm.contrib.edgex.tir.schedule import EdgexSchedule
import numpy as np
from tvm.contrib.edgex.topi.conv2d import schedule_edgex_conv_block


# fmt: off
@T.prim_func
def fused_conv_with_bias_norm_relu(x: T.handle, weight: T.handle, y: T.handle,
                                   bias_param: T.handle, mul_param: T.handle, shift_param: T.handle) -> None:
    c_i = T.var("int32")
    c_o = T.var("int32")
    h = T.var("int32")
    w = T.var("int32")
    k_h = T.var("int32")
    k_w = T.var("int32")
    Y = T.match_buffer(y, [1, c_o, h // 2, w // 2], dtype="int8")
    W = T.match_buffer(weight, [c_o, c_i, k_h, k_w], dtype="int8")
    X = T.match_buffer(x, [1, c_i, h, w], dtype="uint8")
    BiasParam = T.match_buffer(bias_param, [c_o], dtype="int32")
    MulParam = T.match_buffer(mul_param, [c_o, 1, 1], dtype="int64")
    ShiftParam = T.match_buffer(shift_param, [c_o, 1, 1], dtype="int64")
    Xpad = T.alloc_buffer([1, c_i, h + 6, w + 6], dtype="uint8")
    T_conv = T.alloc_buffer([1, c_o, h // 2, w // 2], dtype="int32")
    T_expand_dims = T.alloc_buffer([c_o, 1, 1], dtype="int32")
    T_add = T.alloc_buffer([1, c_o, h // 2, w // 2], dtype="int32")
    compute_1 = T.alloc_buffer([1, c_o, h // 2, w // 2], dtype="int64")
    T_multiply = T.alloc_buffer([1, c_o, h // 2, w // 2], dtype="int64")
    T_round_right_shift = T.alloc_buffer([1, c_o, h // 2, w // 2], dtype="int64")
    compute_2 = T.alloc_buffer([1, c_o, h // 2, w // 2], dtype="int64")
    compute_3 = T.alloc_buffer([1, c_o, h // 2, w // 2], dtype="int8")
    for i0, i1, i2, i3 in T.grid(1, c_i, h + 6, w + 6):
        with T.block("pad_temp"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            Xpad[ax0, ax1, ax2, ax3] = T.if_then_else(
                T.likely(3 <= ax2, dtype="bool") and \
                T.likely(ax2 < 227, dtype="bool") and \
                T.likely(3 <= ax3, dtype="bool") and \
                T.likely(ax3 < 227, dtype="bool"),
                X[ax0, ax1, ax2 - 3, ax3 - 3], T.uint8(0), dtype="uint8")
    for i0, i1, i2, i3, i4, i5, i6 in T.grid(1, c_o, h // 2, w // 2, c_i, k_h, k_w):
        with T.block("compute"):
            nn, cc, yy, xx, rc, ry, rx = T.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
            T_conv[nn, cc, yy, xx] = T_conv[nn, cc, yy, xx] + \
                T.cast(Xpad[nn, rc, yy*2 + ry, xx*2 + rx], "int32") * T.cast(W[cc, rc, ry, rx], "int32")
    for i0, i1, i2 in T.grid(c_o, 1, 1):
        with T.block("T_expand_dims"):
            ax0, ax1, ax2 = T.axis.remap("SSS", [i0, i1, i2])
            T_expand_dims[ax0, ax1, ax2] = BiasParam[ax0]
    for i0, i1, i2, i3 in T.grid(1, c_o, h // 2, w // 2):
        with T.block("T_add"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_add[ax0, ax1, ax2, ax3] = (T_conv[ax0, ax1, ax2, ax3] + T_expand_dims[ax1, 0, 0])
    for i0, i1, i2, i3 in T.grid(1, c_o, h // 2, w // 2):
        with T.block("compute_1"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            compute_1[ax0, ax1, ax2, ax3] = T.cast(T_add[ax0, ax1, ax2, ax3], "int64")
    for i0, i1, i2, i3 in T.grid(1, c_o, h // 2, w // 2):
        with T.block("T_multiply"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_multiply[ax0, ax1, ax2, ax3] = compute_1[ax0, ax1, ax2, ax3] * MulParam[ax1, 0, 0]
    for i0, i1, i2, i3 in T.grid(1, c_o, h // 2, w // 2):
        with T.block("T_round_right_shift"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_round_right_shift[ax0, ax1, ax2, ax3] = T.nnp_round_right_shift(
                T_multiply[ax0, ax1, ax2, ax3], ShiftParam[ax1, 0, 0], dtype="int64")
    for i0, i1, i2, i3 in T.grid(1, c_o, h // 2, w // 2):
        with T.block("compute_2"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            compute_2[ax0, ax1, ax2, ax3] = T.max(T.min(
                T_round_right_shift[ax0, ax1, ax2, ax3], T.int64(127)), T.int64(-128))
    for i0, i1, i2, i3 in T.grid(1, c_o, h // 2, w // 2):
        with T.block("compute_3"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            compute_3[ax0, ax1, ax2, ax3] = T.cast(compute_2[ax0, ax1, ax2, ax3], "int8")
    for i0, i1, i2, i3 in T.grid(1, c_o, h // 2, w // 2):
        with T.block("T_relu"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            Y[ax0, ax1, ax2, ax3] = T.max(compute_3[ax0, ax1, ax2, ax3], T.int8(0))


@T.prim_func
def single_conv(x: T.handle, weight: T.handle, y: T.handle) -> None:
    c_i = T.var("int32")
    c_o = T.var("int32")
    h = T.var("int32")
    w = T.var("int32")
    k_h = T.var("int32")
    k_w = T.var("int32")
    Y = T.match_buffer(y, [1, c_o, h // 2, w // 2], dtype="int32")
    W = T.match_buffer(weight, [c_o, c_i, k_h, k_w], dtype="int8")
    X = T.match_buffer(x, [1, c_i, h, w], dtype="uint8")
    Xpad = T.alloc_buffer([1, c_i, h + 6, w + 6], dtype="uint8")
    for i0, i1, i2, i3 in T.grid(1, c_i, h + 6, w + 6):
        with T.block("pad_temp"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            Xpad[ax0, ax1, ax2, ax3] = T.if_then_else(
                T.likely(3 <= ax2, dtype="bool") and \
                T.likely(ax2 < 227, dtype="bool") and \
                T.likely(3 <= ax3, dtype="bool") and \
                T.likely(ax3 < 227, dtype="bool"),
                X[ax0, ax1, ax2 - 3, ax3 - 3], T.uint8(0), dtype="uint8")
    for i0, i1, i2, i3, i4, i5, i6 in T.grid(1, c_o, h // 2, w // 2, c_i, k_h, k_w):
        with T.block("compute"):
            nn, cc, yy, xx, rc, ry, rx = T.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
            Y[nn, cc, yy, xx] = Y[nn, cc, yy, xx] + \
                T.cast(Xpad[nn, rc, yy*2 + ry, xx*2 + rx], "int32") * T.cast(W[cc, rc, ry, rx], "int32")
# fmt: on


def test_tir_func_single_conv2d():
    c_i = 3
    c_o = 16
    h = 224
    w = 224
    input_shape = [1, c_i, h, w]
    weight_shape = [c_o, c_i, 7, 7]
    output_shape = [1, c_o, h // 2, w // 2]
    strides = [2, 2]
    kernel_size = [7, 7]
    padding = [3, 3, 3, 3]
    dilation = [1, 1]

    p_input, p_weight = single_conv.params[0:2]
    primfunc = single_conv.specialize(
        {
            p_input: tir.decl_buffer(input_shape, "int8"),
            p_weight: tir.decl_buffer(weight_shape, "int8"),
        }
    )
    s = EdgexSchedule(primfunc)
    conv = s.get_block("compute")
    schedule_edgex_conv_block(
        s,
        conv,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=1,
    )
    func = s.mod["main"]

    new_weight_shape = [int(x) for x in func.buffer_map[func.params[1]].shape]
    x_np = np.random.randint(0, 128, input_shape).astype("uint8")
    w_np_raw = np.ones(weight_shape).astype("int8")
    w_np_rewrite = np.ones(new_weight_shape).astype("int8")
    y_np = np.zeros(output_shape).astype("int32")

    with tvm.ir.transform.PassContext():
        ctx = tvm.cpu()
        cpu_mod = tvm.build(primfunc, [], target="llvm")
        x_nd = tvm.nd.array(x_np, ctx)
        w_nd_raw = tvm.nd.array(w_np_raw, ctx)
        y_nd = tvm.nd.array(y_np, ctx)
        cpu_mod(x_nd, w_nd_raw, y_nd)
        cpu_res = y_nd.asnumpy()

    with build_config_nnp():
        ctx = tvm.edgex()
        edgex_mod = tvm.build(s.mod["main"], [], target="edgex", name="tir_single_conv2d")
        x_nd = tvm.nd.array(x_np, ctx)
        w_nd_rewrite = tvm.nd.array(w_np_rewrite, ctx)
        y_nd = tvm.nd.array(y_np, ctx)
        edgex_mod(x_nd, w_nd_rewrite, y_nd)
        edgex_res = y_nd.asnumpy()
        tvm.testing.assert_allclose(edgex_res, cpu_res, rtol=1e-5)


def test_tir_func_quantized_conv2d():
    c_i = 3
    c_o = 64
    h = 224
    w = 224
    k_h = 7
    k_w = 7
    input_shape = [1, c_i, h, w]
    weight_shape = [c_o, c_i, k_h, k_w]
    output_shape = [1, c_o, h // 2, w // 2]
    strides = [2, 2]
    padding = [3, 3, 3, 3]
    dilation = [1, 1]
    kernel_size = [k_h, k_w]

    p_input, p_weight = fused_conv_with_bias_norm_relu.params[0:2]
    primfunc = fused_conv_with_bias_norm_relu.specialize(
        {
            p_input: tir.decl_buffer(input_shape, "int8"),
            p_weight: tir.decl_buffer(weight_shape, "int8"),
        }
    )
    s = EdgexSchedule(primfunc)
    conv = s.get_block("compute")
    schedule_edgex_conv_block(
        s,
        conv,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=1,
    )
    func = s.mod["main"]

    new_weight_shape = [int(x) for x in func.buffer_map[func.params[1]].shape]
    x_np = np.random.randint(0, 128, input_shape).astype("uint8")
    w_np_raw = np.ones(weight_shape).astype("int8") * 0
    w_np_rewrite = np.ones(new_weight_shape).astype("int8")
    y_np = np.zeros(output_shape).astype("int8")
    bias_np = np.random.randint(-128, 127, [c_o]).astype("int32")
    mulnorm_np = np.random.randint(0, 10, [c_o, 1, 1]).astype("int64")
    shiftnorm_np = np.random.randint(1, 30, [c_o, 1, 1]).astype("int64")

    bias_np = np.ones([c_o]).astype("int32") * (2 ** 16 - 1)
    mulnorm_np = np.ones([c_o, 1, 1]).astype("int64") * 128
    shiftnorm_np = np.ones([c_o, 1, 1]).astype("int64") * 32

    # transform bias/norm parameters
    lines = (c_o + 15) // 16
    bias_np_rewrite = np.reshape(bias_np.view("int8"), [lines, 64])
    merged_np = np.concatenate([mulnorm_np, shiftnorm_np], axis=2)
    merged_np = np.reshape(merged_np.astype("int8"), [lines, 32])
    merged_np = np.concatenate([bias_np_rewrite, merged_np], axis=1)

    with tvm.ir.transform.PassContext():
        ctx = tvm.cpu()
        p_input, p_weight = single_conv.params[0:2]
        primfunc_without_quantize = single_conv.specialize(
            {
                p_input: tir.decl_buffer(input_shape, "int8"),
                p_weight: tir.decl_buffer(weight_shape, "int8"),
            }
        )
        cpu_mod = tvm.build(primfunc_without_quantize, [], target="llvm")
        x_nd = tvm.nd.array(x_np, ctx)
        w_nd_raw = tvm.nd.array(w_np_raw, ctx)
        y_nd = tvm.nd.array(np.zeros(output_shape).astype("int32"), ctx)
        cpu_mod(x_nd, w_nd_raw, y_nd)
        cpu_res = y_nd.asnumpy()
        # quantize with numpy
        cpu_res = cpu_res.astype("int64")
        cpu_res = (cpu_res + np.expand_dims(np.expand_dims(bias_np, -1), -1)) * mulnorm_np
        cpu_res = (cpu_res + (1 << (shiftnorm_np - 1))) >> shiftnorm_np
        cpu_res = np.maximum(cpu_res, np.ones_like(cpu_res) * -128)
        cpu_res = np.minimum(cpu_res, np.ones_like(cpu_res) * 127)
        cpu_res = cpu_res.astype("int8")
        cpu_res = cpu_res * (cpu_res > 0)

    with build_config_nnp():
        ctx = tvm.edgex()
        edgex_mod = tvm.build(s.mod["main"], [], target="edgex", name="tir_quantized_conv2d")
        x_nd = tvm.nd.array(x_np, ctx)
        w_nd_rewrite = tvm.nd.array(w_np_rewrite, ctx)
        y_nd = tvm.nd.array(y_np, ctx)
        params_nd = tvm.nd.array(merged_np, ctx)
        edgex_mod(x_nd, w_nd_rewrite, y_nd, params_nd)
        edgex_res = y_nd.asnumpy()
        # the result may only match on specified data ranges
        tvm.testing.assert_allclose(edgex_res, cpu_res, rtol=1e-5)


if __name__ == "__main__":
    test_tir_func_single_conv2d()
    test_tir_func_quantized_conv2d()
