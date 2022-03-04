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
from tvm.contrib.edgex.testing import check_edgex_tir_build
import tvm.testing
from tvm.script import tir as T
from tvm.contrib.edgex.tir.schedule import EdgexSchedule
import numpy as np
from tvm.contrib.edgex.topi.conv2d import Conv2dScheduleConfig, schedule_edgex_conv_block
from tvm.contrib.edgex.relay.transform import PostScheduleArgumentRewriteManager


@T.prim_func
def single_conv(
    x: T.handle,
    weight: T.handle,
    y: T.handle,
    stride_h: T.int32,
    stride_w: T.int32,
    pad_top: T.int32,
    pad_left: T.int32,
    pad_bottom: T.int32,
    pad_right: T.int32,
) -> None:
    c_i = T.var("int32")
    c_o = T.var("int32")
    h = T.var("int32")
    w = T.var("int32")
    k_h = T.var("int32")
    k_w = T.var("int32")
    Y = T.match_buffer(
        y,
        [
            1,
            c_o,
            (h + pad_top + pad_bottom - k_h) // stride_h + 1,
            (w + pad_left + pad_right - k_w) // stride_w + 1,
        ],
        dtype="int32",
    )
    W = T.match_buffer(weight, [c_o, c_i, k_h, k_w], dtype="int8")
    X = T.match_buffer(x, [1, c_i, h, w], dtype="int8")
    Xpad = T.alloc_buffer(
        [1, c_i, h + pad_top + pad_bottom, w + pad_left + pad_right], dtype="int8"
    )
    for i0, i1, i2, i3 in T.grid(1, c_i, h + pad_top + pad_bottom, w + pad_left + pad_right):
        with T.block("pad_temp"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            Xpad[ax0, ax1, ax2, ax3] = T.if_then_else(
                T.likely(pad_top <= ax2, dtype="bool")
                and T.likely(ax2 < h + pad_bottom, dtype="bool")
                and T.likely(pad_left <= ax3, dtype="bool")
                and T.likely(ax3 < w + pad_right, dtype="bool"),
                X[ax0, ax1, ax2 - pad_top, ax3 - pad_left],
                T.int8(0),
                dtype="int8",
            )
    for i0, i1, i2, i3, i4, i5, i6 in T.grid(
        1,
        c_o,
        (h + pad_top + pad_bottom - k_h) // stride_h + 1,
        (w + pad_left + pad_right - k_w) // stride_w + 1,
        c_i,
        k_h,
        k_w,
    ):
        with T.block("conv"):
            nn, cc, yy, xx, rc, ry, rx = T.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
            with T.init():
                Y[nn, cc, yy, xx] = 0
            Y[nn, cc, yy, xx] = Y[nn, cc, yy, xx] + T.cast(
                Xpad[nn, rc, yy * stride_h + ry, xx * stride_w + rx], "int32"
            ) * T.cast(W[cc, rc, ry, rx], "int32")


@T.prim_func
def post_conv_bias_add(x: T.handle, y: T.handle, bias_param: T.handle) -> None:
    c = T.var("int32")
    h = T.var("int32")
    w = T.var("int32")
    Y = T.match_buffer(y, [1, c, h, w], dtype="int32")
    X = T.match_buffer(x, [1, c, h, w], dtype="int32")
    T_expand_dims = T.alloc_buffer([c, 1, 1], dtype="int32")
    BiasParam = T.match_buffer(bias_param, [c], dtype="int32")
    for i0, i1, i2 in T.grid(c, 1, 1):
        with T.block("T_expand_dims"):
            ax0, ax1, ax2 = T.axis.remap("SSS", [i0, i1, i2])
            T_expand_dims[ax0, ax1, ax2] = BiasParam[ax0]
    for i0, i1, i2, i3 in T.grid(1, c, h, w):
        with T.block("T_add"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            Y[ax0, ax1, ax2, ax3] = X[ax0, ax1, ax2, ax3] + T_expand_dims[ax1, 0, 0]


@T.prim_func
def post_conv_quantize(
    x: T.handle, y: T.handle, mul_param: T.handle, shift_param: T.handle
) -> None:
    c = T.var("int32")
    h = T.var("int32")
    w = T.var("int32")
    Y = T.match_buffer(y, [1, c, h, w], dtype="int8")
    X = T.match_buffer(x, [1, c, h, w], dtype="int32")
    MulParam = T.match_buffer(mul_param, [c, 1, 1], dtype="int64")
    ShiftParam = T.match_buffer(shift_param, [c, 1, 1], dtype="int64")
    compute_1 = T.alloc_buffer([1, c, h, w], dtype="int64")
    T_multiply = T.alloc_buffer([1, c, h, w], dtype="int64")
    T_round_right_shift = T.alloc_buffer([1, c, h, w], dtype="int64")
    compute_2 = T.alloc_buffer([1, c, h, w], dtype="int64")
    compute_3 = T.alloc_buffer([1, c, h, w], dtype="int8")
    for i0, i1, i2, i3 in T.grid(1, c, h, w):
        with T.block("compute_1"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            compute_1[ax0, ax1, ax2, ax3] = T.cast(X[ax0, ax1, ax2, ax3], "int64")
    for i0, i1, i2, i3 in T.grid(1, c, h, w):
        with T.block("T_multiply"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_multiply[ax0, ax1, ax2, ax3] = compute_1[ax0, ax1, ax2, ax3] * MulParam[ax1, 0, 0]
    for i0, i1, i2, i3 in T.grid(1, c, h, w):
        with T.block("T_round_right_shift"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T_round_right_shift[ax0, ax1, ax2, ax3] = T.nnp_round_right_shift(
                T_multiply[ax0, ax1, ax2, ax3], ShiftParam[ax1, 0, 0], dtype="int64"
            )
    for i0, i1, i2, i3 in T.grid(1, c, h, w):
        with T.block("compute_2"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            compute_2[ax0, ax1, ax2, ax3] = T.max(
                T.min(T_round_right_shift[ax0, ax1, ax2, ax3], T.int64(127)), T.int64(-128)
            )
    for i0, i1, i2, i3 in T.grid(1, c, h, w):
        with T.block("compute_3"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            compute_3[ax0, ax1, ax2, ax3] = T.cast(compute_2[ax0, ax1, ax2, ax3], "int8")
    for i0, i1, i2, i3 in T.grid(1, c, h, w):
        with T.block("T_relu"):
            ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            Y[ax0, ax1, ax2, ax3] = T.max(compute_3[ax0, ax1, ax2, ax3], T.int8(0))


def get_conv2d_primfunc(
    input_shape,
    weight_shape,
    strides=[
        1,
        1,
    ],
    padding=[0, 0, 0, 0],
    dilation=[1, 1],
):
    """helper to specialize conv template"""
    assert dilation[0] == 1 and dilation[1] == 1
    x, w, _, stride_h, stride_w, pad_top, pad_left, pad_bottom, pad_right = single_conv.params
    return single_conv.specialize(
        {
            x: tir.decl_buffer(input_shape, "int8"),
            w: tir.decl_buffer(weight_shape, "int8"),
            stride_h: strides[0],
            stride_w: strides[1],
            pad_top: padding[0],
            pad_left: padding[1],
            pad_bottom: padding[2],
            pad_right: padding[3],
        }
    )


def test_tir_func_single_conv2d():
    c_i, c_o, h, w = 3, 16, 224, 224
    strides = [2, 2]
    kernel_size = [7, 7]
    padding = [3, 3, 3, 3]
    dilation = [1, 1]
    primfunc = get_conv2d_primfunc(
        input_shape=[1, c_i, h, w], weight_shape=[c_o, c_i, 7, 7], strides=strides, padding=padding
    )

    def fschedule(attrs, func, target):
        s = EdgexSchedule(func)
        relay_rewrite_mgr = PostScheduleArgumentRewriteManager(s)
        conv = s.get_block("conv")
        schedule_edgex_conv_block(
            s,
            conv,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=1,
            relay_rewrite_mgr=relay_rewrite_mgr,
        )
        return relay_rewrite_mgr.create_annotated_func()

    check_edgex_tir_build("tir_single_conv2d", primfunc, edgex_fschedule=fschedule, output_idx=2)


def test_tir_func_quantized_conv2d():
    @T.prim_func
    def quantized_conv(
        x: T.handle,
        weight: T.handle,
        y: T.handle,
        bias_param: T.handle,
        mul_param: T.handle,
        shift_param: T.handle,
    ) -> None:
        c_i = T.var("int32")
        c_o = T.var("int32")
        h_i = T.var("int32")
        w_i = T.var("int32")
        h_o = T.var("int32")
        w_o = T.var("int32")
        k_h = T.var("int32")
        k_w = T.var("int32")
        Y = T.match_buffer(y, [1, c_o, h_o, w_o], dtype="int8")
        W = T.match_buffer(weight, [c_o, c_i, k_h, k_w], dtype="int8")
        X = T.match_buffer(x, [1, c_i, h_i, w_i], dtype="int8")
        BiasParam = T.match_buffer(bias_param, [Y.shape[1]], dtype="int32")
        MulParam = T.match_buffer(mul_param, [Y.shape[1], 1, 1], dtype="int64")
        ShiftParam = T.match_buffer(shift_param, [Y.shape[1], 1, 1], dtype="int64")
        T_conv = T.alloc_buffer(Y.shape, dtype="int32")
        T_add = T.alloc_buffer(Y.shape, dtype="int32")
        T.evaluate(T.call_extern("conv1", X.data, W.data, T_conv.data, dtype=""))
        T.evaluate(T.call_extern("biasadd", T_conv.data, T_add.data, BiasParam.data, dtype=""))
        T.evaluate(
            T.call_extern("quantize", T_add.data, Y.data, MulParam.data, ShiftParam.data, dtype="")
        )

    c_i, c_o, h, w = 3, 16, 224, 224
    input_shape = [1, c_i, h, w]
    output_shape = [1, c_o, h // 2, w // 2]
    weight_shape = [c_o, c_i, 3, 3]
    strides = [2, 2]
    kernel_size = [3, 3]
    padding = [1, 1, 1, 1]
    dilation = [1, 1]
    conv1 = get_conv2d_primfunc(input_shape, weight_shape, strides=strides, padding=padding)
    func = quantized_conv.specialize(
        {
            quantized_conv.params[0]: tir.decl_buffer(input_shape, "int8"),
            quantized_conv.params[1]: tir.decl_buffer(weight_shape, "int8"),
            quantized_conv.params[2]: tir.decl_buffer(output_shape, "int8"),
        }
    )
    mod = tvm.contrib.edgex.tir.transform.InlinePrimFuncCalls(
        {"conv1": conv1, "biasadd": post_conv_bias_add, "quantize": post_conv_quantize}
    )(tvm.ir.IRModule.from_expr(func))

    def fschedule(attrs, primfunc, target):
        s = EdgexSchedule(primfunc)
        relay_rewrite_mgr = PostScheduleArgumentRewriteManager(s)
        conv = s.get_block("conv")
        schedule_edgex_conv_block(
            s,
            conv,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=1,
            relay_rewrite_mgr=relay_rewrite_mgr,
        )
        scheduled = relay_rewrite_mgr.create_annotated_func()
        return scheduled

    mulnorm_np = np.random.randint(0, 20, [c_o, 1, 1]).astype("int64")
    shiftnorm_np = np.random.randint(0, 5, [c_o, 1, 1]).astype("int64")
    check_edgex_tir_build(
        "tir_quantized_conv2d",
        mod["main"],
        edgex_fschedule=fschedule,
        output_idx=2,
        input_data=[None, None, None, None, mulnorm_np, shiftnorm_np],
    )


def test_consecutive_small_conv_sharing_dm():
    # TODO(fengrong,xinqi): currently we can only test two exactly same conv
    # (1) fix passes which are coded towards single op
    # (2) correct dma order
    @T.prim_func
    def consecutive_small_conv(
        x: T.handle,
        weight1: T.handle,
        weight2: T.handle,
        y: T.handle,
        bias_param1: T.handle,
        mul_param1: T.handle,
        shift_param1: T.handle,
        bias_param2: T.handle,
        mul_param2: T.handle,
        shift_param2: T.handle,
    ) -> None:
        c = T.var("int32")
        h = T.var("int32")
        w = T.var("int32")
        Y = T.match_buffer(y, [1, c, h, w], dtype="int8")
        W1 = T.match_buffer(weight1, [c, c, 3, 3], dtype="int8")
        W2 = T.match_buffer(weight2, [c, c, 3, 3], dtype="int8")
        X1 = T.match_buffer(x, [1, c, h, w], dtype="int8")
        BiasParam1 = T.match_buffer(bias_param1, [Y.shape[1]], dtype="int32")
        Mul1 = T.match_buffer(mul_param1, [Y.shape[1], 1, 1], dtype="int64")
        Shift1 = T.match_buffer(shift_param1, [Y.shape[1], 1, 1], dtype="int64")
        BiasParam2 = T.match_buffer(bias_param2, [Y.shape[1]], dtype="int32")
        Mul2 = T.match_buffer(mul_param2, [Y.shape[1], 1, 1], dtype="int64")
        Shift2 = T.match_buffer(shift_param2, [Y.shape[1], 1, 1], dtype="int64")
        T_conv1 = T.alloc_buffer(Y.shape, dtype="int32")
        T_conv2 = T.alloc_buffer(Y.shape, dtype="int32")
        T_add1 = T.alloc_buffer(Y.shape, dtype="int32")
        T_add2 = T.alloc_buffer(Y.shape, dtype="int32")
        X2 = T.alloc_buffer([1, c, h, w], dtype="int8")
        T.evaluate(T.call_extern("conv1", X1.data, W1.data, T_conv1.data, dtype=""))
        T.evaluate(T.call_extern("biasadd", T_conv1.data, T_add1.data, BiasParam1.data, dtype=""))
        T.evaluate(
            T.call_extern("quantize", T_add1.data, X2.data, Mul1.data, Shift1.data, dtype="")
        )
        T.evaluate(T.call_extern("conv2", X2.data, W2.data, T_conv2.data, dtype=""))
        T.evaluate(T.call_extern("biasadd", T_conv2.data, T_add2.data, BiasParam2.data, dtype=""))
        T.evaluate(T.call_extern("quantize", T_add2.data, Y.data, Mul2.data, Shift2.data, dtype=""))

    c = 16
    h = 8
    w = 8
    data_shape = [1, c, h, w]
    weight_shape = [c, c, 3, 3]
    strides = [1, 1]
    kernel_size = [3, 3]
    padding = [1, 1, 1, 1]
    dilation = [1, 1]
    conv = get_conv2d_primfunc(data_shape, weight_shape, strides=strides, padding=padding)
    func = consecutive_small_conv.specialize(
        {
            consecutive_small_conv.params[0]: tir.decl_buffer(data_shape, "int8"),
            consecutive_small_conv.params[1]: tir.decl_buffer(weight_shape, "int8"),
        }
    )
    mod = tvm.contrib.edgex.tir.transform.InlinePrimFuncCalls(
        {
            "conv1": conv,
            "biasadd": post_conv_bias_add,
            "quantize": post_conv_quantize,
            "conv2": conv,
        }
    )(tvm.ir.IRModule.from_expr(func))

    def fschedule(attrs, primfunc, target):
        s = EdgexSchedule(primfunc)

        # we use PostScheduleArgumentRewriteManager to allow check utility
        # update the input tensor layout for device test
        relay_rewrite_mgr = PostScheduleArgumentRewriteManager(s)

        # schedule the first conv, do not store ddr
        conv = s.get_block("conv")
        cfg = Conv2dScheduleConfig()
        cfg.is_ddr_input = True
        cfg.is_ddr_output = False
        schedule_edgex_conv_block(
            s,
            conv,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=1,
            relay_rewrite_mgr=relay_rewrite_mgr,
            cfg=cfg,
        )

        # schedule the second conv, do not load ddr
        conv = s.get_block("conv_1")
        cfg = Conv2dScheduleConfig()
        cfg.is_ddr_input = False
        cfg.is_ddr_output = True
        schedule_edgex_conv_block(
            s,
            conv,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=1,
            relay_rewrite_mgr=relay_rewrite_mgr,
            cfg=cfg,
        )
        scheduled = relay_rewrite_mgr.create_annotated_func()
        return scheduled

    # share same parameters for two convs now
    bias_np = np.random.randint(-64, 64, [c]).astype("int32")
    mulnorm_np = np.random.randint(0, 16, [c, 1, 1]).astype("int64")
    shiftnorm_np = np.random.randint(1, 10, [c, 1, 1]).astype("int64")
    check_edgex_tir_build(
        "consecutive_small_conv",
        mod["main"],
        edgex_fschedule=fschedule,  # pass device only schedule, because it changes tensor layout
        output_idx=[3],  # specify which tensor is output, thus it can be zero initialized
        input_data=[
            None,
            None,
            None,
            None,
            bias_np,
            mulnorm_np,
            shiftnorm_np,
            bias_np,
            mulnorm_np,
            shiftnorm_np,
        ],
    )


@T.prim_func
def conv2d_NCHWc(
    x: T.handle,
    weight: T.handle,
    y: T.handle,
    stride_h: T.int32,
    stride_w: T.int32,
    pad_top: T.int32,
    pad_left: T.int32,
    pad_bottom: T.int32,
    pad_right: T.int32,
) -> None:
    n = T.var("int32")
    c_i = T.var("int32")
    c_o = T.var("int32")
    c_b = T.var("int32")
    h = T.var("int32")
    w = T.var("int32")
    k_h = T.var("int32")
    k_w = T.var("int32")
    X = T.match_buffer(x, [n, c_i, h, w, c_b], dtype="int8")
    W = T.match_buffer(weight, [c_o, c_i, k_h, k_w, c_b, c_b], dtype="int8")
    Y = T.match_buffer(
        y,
        [
            n,
            c_o,
            (h + pad_top + pad_bottom - k_h) // stride_h + 1,
            (w + pad_left + pad_right - k_w) // stride_w + 1,
            c_b,
        ],
        dtype="int32",
    )

    # pad data first
    Xpad = T.alloc_buffer(
        [n, c_i, h + pad_top + pad_bottom, w + pad_left + pad_right, c_b], dtype="int8"
    )
    for i0, i1, i2, i3, i4 in T.grid(
        n, c_i, h + pad_top + pad_bottom, w + pad_left + pad_right, c_b
    ):
        with T.block("pad_data"):
            ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
            Xpad[ax0, ax1, ax2, ax3, ax4] = T.if_then_else(
                T.likely(pad_top <= ax2, dtype="bool")
                and T.likely(ax2 < h + pad_bottom, dtype="bool")
                and T.likely(pad_left <= ax3, dtype="bool")
                and T.likely(ax3 < w + pad_right, dtype="bool"),
                X[ax0, ax1, ax2 - pad_top, ax3 - pad_left, ax4],
                T.int8(0),
                dtype="int8",
            )

    # compute conv2d_NCHWc
    # nn:1, co:c_o/oc_chunk, oh:o_h, ow:o_w, ic:i_c, kh:k_h, kw:k_w, cb:c_b/oc_block
    for i0, i1, i2, i3, i4, i5, i6, i7 in T.grid(
        n,
        c_o,
        (h + pad_top + pad_bottom - k_h) // stride_h + 1,
        (w + pad_left + pad_right - k_w) // stride_w + 1,
        c_b,
        c_i * c_b,
        k_h,
        k_w,
    ):
        with T.block("conv2d_NCHWc"):
            nn, co, oh, ow, cb, ic, kh, kw = T.axis.remap(
                "SSSSSRRR", [i0, i1, i2, i3, i4, i5, i6, i7]
            )
            with T.init():
                Y[nn, co, oh, ow, cb] = 0
            Y[nn, co, oh, ow, cb] += (
                T.cast(
                    Xpad[
                        nn,
                        T.floordiv(ic, c_b),
                        oh * stride_h + kh,
                        ow * stride_w + kw,
                        T.floormod(ic, c_b),
                    ],
                    "int32",
                )
                * T.cast(W[co, T.floordiv(ic, c_b), kh, kw, T.floormod(ic, c_b), cb], "int32")
            )


def test_tir_func_conv2d_NCHW16c():
    n, c_i, c_o, c_b, h, w = 1, 1, 4, 16, 224, 224
    strides = [2, 2]
    kernel_size = [7, 7]
    padding = [3, 3, 3, 3]
    dilation = [1, 1]
    input_shape = [n, c_i, h, w, c_b]
    weight_shape = [c_o, c_i, *kernel_size, c_b, c_b]
    primfunc = conv2d_NCHWc
    x, weight, _, stride_h, stride_w, pad_top, pad_left, pad_bottom, pad_right = conv2d_NCHWc.params
    primfunc = primfunc.specialize(
        {
            x: tir.decl_buffer(input_shape, "int8"),
            weight: tir.decl_buffer(weight_shape, "int8"),
            stride_h: strides[0],
            stride_w: strides[1],
            pad_top: padding[0],
            pad_left: padding[1],
            pad_bottom: padding[2],
            pad_right: padding[3],
        }
    )

    def fschedule(attrs, func, target):
        s = EdgexSchedule(func)
        relay_rewrite_mgr = PostScheduleArgumentRewriteManager(s)
        conv = s.get_block("conv2d_NCHWc")
        schedule_edgex_conv_block(
            s,
            conv,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=1,
            layout="NCHW16c",
            kernel_layout="OIHW16i16o",
            relay_rewrite_mgr=relay_rewrite_mgr,
        )
        return relay_rewrite_mgr.create_annotated_func()

    check_edgex_tir_build("tir_conv2d_NCHWc", primfunc, edgex_fschedule=fschedule, output_idx=2)


@T.prim_func
def conv2d_NCHWc_with_oihw(
    x: T.handle,
    weight: T.handle,
    y: T.handle,
    c_o: T.int32,
    stride_h: T.int32,
    stride_w: T.int32,
    pad_top: T.int32,
    pad_left: T.int32,
    pad_bottom: T.int32,
    pad_right: T.int32,
) -> None:
    n = T.var("int32")
    c_i = T.var("int32")
    c_b = T.var("int32")
    h = T.var("int32")
    w = T.var("int32")
    k_h = T.var("int32")
    k_w = T.var("int32")
    X = T.match_buffer(x, [n, c_i, h, w, c_b], dtype="int8")
    W = T.match_buffer(weight, [64, 16, k_h, k_w], dtype="int8")
    Y = T.match_buffer(
        y,
        [
            n,
            c_o,
            (h + pad_top + pad_bottom - k_h) // stride_h + 1,
            (w + pad_left + pad_right - k_w) // stride_w + 1,
            c_b,
        ],
        dtype="int32",
    )

    # pad data first
    Xpad = T.alloc_buffer(
        [n, c_i, h + pad_top + pad_bottom, w + pad_left + pad_right, c_b], dtype="int8"
    )
    for i0, i1, i2, i3, i4 in T.grid(
        n, c_i, h + pad_top + pad_bottom, w + pad_left + pad_right, c_b
    ):
        with T.block("pad_data"):
            ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
            Xpad[ax0, ax1, ax2, ax3, ax4] = T.if_then_else(
                T.likely(pad_top <= ax2, dtype="bool")
                and T.likely(ax2 < h + pad_bottom, dtype="bool")
                and T.likely(pad_left <= ax3, dtype="bool")
                and T.likely(ax3 < w + pad_right, dtype="bool"),
                X[ax0, ax1, ax2 - pad_top, ax3 - pad_left, ax4],
                T.int8(0),
                dtype="int8",
            )

    # compute conv2d_NCHWc
    # nn:1, co:c_o/oc_chunk, oh:o_h, ow:o_w, ic:i_c, kh:k_h, kw:k_w, cb:c_b/oc_block
    for i0, i1, i2, i3, i4, i5, i6, i7 in T.grid(
        n,
        c_o,
        (h + pad_top + pad_bottom - k_h) // stride_h + 1,
        (w + pad_left + pad_right - k_w) // stride_w + 1,
        c_b,
        c_i * c_b,
        k_h,
        k_w,
    ):
        with T.block("conv2d_NCHWc_oihw"):
            nn, co, oh, ow, cb, ic, kh, kw = T.axis.remap(
                "SSSSSRRR", [i0, i1, i2, i3, i4, i5, i6, i7]
            )
            with T.init():
                Y[nn, co, oh, ow, cb] = 0
            Y[nn, co, oh, ow, cb] += (
                T.cast(
                    Xpad[
                        nn,
                        T.floordiv(ic, c_b),
                        oh * stride_h + kh,
                        ow * stride_w + kw,
                        T.floormod(ic, c_b),
                    ],
                    "int32",
                )
                * T.cast(W[co * 16 + cb, ic, kh, kw], "int32")
            )


def test_tir_func_conv2d_NCHW16c_with_oihw():
    n, c_i, c_o, c_b, h, w = 1, 1, 4, 16, 224, 224
    strides = [2, 2]
    kernel_size = [7, 7]
    padding = [3, 3, 3, 3]
    dilation = [1, 1]
    input_shape = [n, c_i, h, w, c_b]
    weight_shape = [c_o * c_b, c_i * c_b, *kernel_size]
    primfunc = conv2d_NCHWc_with_oihw
    (
        x,
        weight,
        _,
        co,
        stride_h,
        stride_w,
        pad_top,
        pad_left,
        pad_bottom,
        pad_right,
    ) = conv2d_NCHWc_with_oihw.params
    primfunc = primfunc.specialize(
        {
            x: tir.decl_buffer(input_shape, "int8"),
            weight: tir.decl_buffer(weight_shape, "int8"),
            co: c_o,
            stride_h: strides[0],
            stride_w: strides[1],
            pad_top: padding[0],
            pad_left: padding[1],
            pad_bottom: padding[2],
            pad_right: padding[3],
        }
    )

    def fschedule(attrs, func, target):
        s = EdgexSchedule(func)
        relay_rewrite_mgr = PostScheduleArgumentRewriteManager(s)
        conv = s.get_block("conv2d_NCHWc_oihw")
        schedule_edgex_conv_block(
            s,
            conv,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=1,
            layout="NCHW16c",
            kernel_layout="OIHW",
            relay_rewrite_mgr=relay_rewrite_mgr,
        )
        return relay_rewrite_mgr.create_annotated_func()

    check_edgex_tir_build(
        "tir_conv2d_NCHWc_oihw", primfunc, edgex_fschedule=fschedule, output_idx=2
    )


if __name__ == "__main__":
    test_tir_func_single_conv2d()
    test_tir_func_quantized_conv2d()
    test_consecutive_small_conv_sharing_dm()
    test_tir_func_conv2d_NCHW16c()
    test_tir_func_conv2d_NCHW16c_with_oihw()
