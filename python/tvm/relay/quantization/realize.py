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
# pylint: disable=unused-argument,inconsistent-return-statements,unexpected-keyword-arg,global-at-module-level
"""realize"""

import logging
import math
import numpy
import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprMutator

try:
    from tvm.relay.dataflow_pattern import is_op, wildcard
    from .relay_ops import round_right_shift
except ImportError:
    pass

from .debug import pair_node
from .method_dtype import _get_dtype_info, DataType


LOGGER = logging.getLogger("quantize")

global TARGET_NNP
TARGET_NNP = "nnp400"


def _quantize(node, input_config):
    """quantize"""

    out_scale = relay.const(input_config["scale"])
    out_zero_point = relay.const(input_config["zero_point"])
    new_node = relay.qnn.op.quantize(
        node,
        output_scale=out_scale,
        output_zero_point=out_zero_point,
        axis=input_config["axis"],
        out_dtype=input_config["dtype"],
    )
    return new_node


def _dequantize(node, output_config):
    """dequantize"""

    in_scale = relay.const(output_config["scale"])
    in_zero_point = relay.const(output_config["zero_point"])
    new_node = relay.qnn.op.dequantize(
        node,
        input_scale=in_scale,
        input_zero_point=in_zero_point,
        axis=output_config["axis"],
    )
    return new_node


def eliminate_quantize_dequantize(node):
    """eliminate_quantize_dequantize"""

    quantize = is_op("qnn.quantize")(wildcard(), wildcard(), wildcard())
    dequantize = is_op("qnn.dequantize")(quantize, wildcard(), wildcard())

    if dequantize.match(node):
        pre_arg = node.args[0]
        cond1 = pre_arg.attrs.axis == node.attrs.axis
        cond2 = pre_arg.args[1].data.asnumpy() == node.args[1].data.asnumpy()
        cond3 = pre_arg.args[2].data.asnumpy() == node.args[2].data.asnumpy()
        if cond1 and cond2.all() and cond3.all():
            return pre_arg.args[0]

    return node


def eliminate_dequantize_quantize(node):
    """eliminate_dequantize_quantize"""
    if "ir_pass" not in relay.__dict__:
        dequantize = is_op("qnn.dequantize")(wildcard(), wildcard(), wildcard())
        quantize = is_op("qnn.quantize")(dequantize, wildcard(), wildcard())

        if quantize.match(node):
            pre_arg = node.args[0]
            # cond1 = pre_arg.attrs.axis == node.attrs.axis
            cond2 = pre_arg.args[1].data.asnumpy() == node.args[1].data.asnumpy()
            cond3 = pre_arg.args[2].data.asnumpy() == node.args[2].data.asnumpy()
            if cond2.all() and cond3.all():
                return pre_arg.args[0]
    else:
        # 300 need clip after op
        if (
            isinstance(node, relay.Call)
            and node.op.name == "qnn.quantize"
            and isinstance(node.args[0], relay.Call)
            and node.args[0].op.name == "qnn.dequantize"
        ):
            pre_arg = node.args[0]
            # cond1 = pre_arg.attrs.axis == node.attrs.axis
            cond2 = pre_arg.args[1].data.asnumpy() == node.args[1].data.asnumpy()
            cond3 = pre_arg.args[2].data.asnumpy() == node.args[2].data.asnumpy()
            if cond2.all() and cond3.all():
                ori_call = node.args[0].args[0]

                name = "Constant"
                if isinstance(ori_call, relay.Call) and isinstance(ori_call.op, relay.Function):
                    name = getattr(ori_call.op.attrs, "Composite")
                    if not isinstance(name, str):
                        name = name.value
                elif isinstance(ori_call, relay.Call):
                    name = ori_call.op.name

                if name in [
                    "nn.relu",
                    "nn.max_pool2d",
                    "nn.max_pool3d",
                    "concatenate",
                    "add",
                    "nn.sum_pool2d",
                ]:
                    r_datatype = node.attrs.out_dtype
                    q_max_min = _get_dtype_info(r_datatype)
                    return relay.clip(
                        pre_arg.args[0], q_max_min["qmin"], q_max_min["qmax"], out_dtype=r_datatype
                    )

                return pre_arg.args[0]

    return node


def _realize_core(cls, old_arg, new_arg, vertex_config, o2n_dict):
    """realize_core"""

    output_config = vertex_config[old_arg].output_config
    input_config = cls.input_config[old_arg]
    if (
        output_config["ref_count"] > 1
        and old_arg in o2n_dict
        and (vertex_config[old_arg].quantized or cls.quantized)
    ):
        new_arg = o2n_dict[old_arg]
    else:
        new_arg = operate(input_config["operate"], new_arg, output_config, input_config, True)

    if (
        output_config["ref_count"] > 1
        and old_arg not in o2n_dict
        and input_config["operate"] != "none"
    ):
        o2n_dict[old_arg] = new_arg
    pair_node(old_arg, new_arg, output_config, input_config, o2n_dict, cls.quantized)
    return new_arg


def operate(op_type, node, output_config, input_config, convert, multiplier=0):
    """operate"""
    # multiplier 0: accumulator->int8 / 1:int8-requantize default:0
    if op_type == "quantize":
        node = _quantize(node, input_config)
        LOGGER.debug("[realize]<quantize> scale is:")
        LOGGER.debug(input_config["scale"])
    elif op_type == "dequantize":
        node = _dequantize(node, output_config)
        LOGGER.debug("[realize]<_dequantize> scale is:")
        LOGGER.debug(output_config["scale"])
    elif op_type == "requantize":
        node = _dequantize(node, output_config)
        node = _quantize(node, input_config)
        LOGGER.debug("[realize]<requantize> before scale is:")
        LOGGER.debug(output_config["scale"])
        LOGGER.debug("[realize]<requantize> after scale is:")
        LOGGER.debug(input_config["scale"])
    elif op_type == "none":
        pass

    # node = eliminate_quantize_dequantize(node)
    node = eliminate_dequantize_quantize(node)

    if convert:
        node = convert_operate(op_type, node, output_config, input_config, multiplier)
    return node


def _quantize_shift(node):
    data = node.args[0]
    scale = node.args[1].data.asnumpy()
    zero_point = node.args[2].data.asnumpy()
    dtype = node.attrs.out_dtype
    axis = node.attrs.axis
    q_min_max = _get_dtype_info(dtype)

    # todo support int4?
    if dtype.startswith("int"):
        bits = dtype[3:]
        if int(bits) <= 8:
            dtype = "int8"
        elif int(bits) <= 16:
            dtype = "int16"
        elif int(bits) <= 32:
            dtype = "int32"

    tmp = relay.frontend.common.infer_type(data)
    shape = tmp.checked_type.concrete_shape

    if isinstance(data, (relay.Var, relay.Call, relay.TupleGetItem)):
        scale = (1.0 / scale).astype("float32")
        if axis != -1:
            tmp1 = [1] * len(shape)
            tmp2 = shape[axis]
            tmp1[axis] = tmp2
            scale = scale.reshape(tmp1)
            zero_point = zero_point.reshape(tmp1)

        new_scale = relay.const(scale.astype("float16"))
        new_zero_point = relay.const(zero_point)

        if tmp.checked_type.dtype != "float16":
            data = relay.cast(data, "float16")

        if (zero_point == 0).all():
            data = relay.multiply(data, new_scale)
            data = relay.round(data)
            data = relay.clip(data, q_min_max["qmin"], q_min_max["qmax"])
            data = relay.cast(data, dtype)
        else:
            data = relay.multiply(data, new_scale)
            data = relay.round(data)
            data = relay.cast(data, "int32")
            data = relay.add(data, new_zero_point)
            data = relay.clip(data, q_min_max["qmin"], q_min_max["qmax"])
            data = relay.cast(data, dtype)

    elif isinstance(data, relay.Constant):
        data = data.data.asnumpy()

        if axis > 10:
            LOGGER.info("[realize] quantize conv2d_transpose_weight...")
            output_oc = scale.shape[0]
            i_c, o_c, k_h, k_w = data.shape
            groups = output_oc // o_c
            data = numpy.reshape(data, (groups, i_c // groups, o_c, k_h * k_w))
            data = numpy.transpose(data, (0, 2, 1, 3))
            data = numpy.reshape(data, (o_c * groups, i_c // groups, k_h, k_w))

            tmp1 = [1] * len(data.shape)
            tmp2 = output_oc
            tmp1[0] = tmp2
            scale = scale.reshape(tmp1)
            zero_point = zero_point.reshape(tmp1)

            data = data / scale + zero_point

            data = data.round()
            data = numpy.clip(data, q_min_max["qmin"], q_min_max["qmax"])
            data = data.astype(dtype)

            data = numpy.reshape(data, (groups, o_c, i_c, k_h * k_w))
            data = numpy.transpose(data, (0, 2, 1, 3))
            data = numpy.reshape(data, (i_c, o_c, k_h, k_w))

            data = relay.const(data)
        else:
            if axis != -1:
                tmp1 = [1] * len(data.shape)
                tmp2 = data.shape[axis]
                tmp1[axis] = tmp2
                scale = scale.reshape(tmp1)
                zero_point = zero_point.reshape(tmp1)

            # todo modify this for identity to nnp300, in fact can just use the next line
            if TARGET_NNP.startswith("nnp4"):
                data = data / scale + zero_point
            # simulate tvm 300 round, farward +-inf
            else:
                if len(data.shape) == 1:
                    data = data.astype("float32") / scale.astype("float32") + zero_point.astype(
                        "float32"
                    )
                else:
                    data = data.astype("float32") * (1.0 / scale.astype("float32")).astype(
                        "float32"
                    ) + zero_point.astype("float32")
                if data.size > 1:
                    data = data.astype("float64")
                    plus_mask = numpy.where(data > 0)
                    data[plus_mask] += 0.00000001
                    minus_mask = numpy.where(data < 0)
                    data[minus_mask] -= 0.00000001
            # simulate end

            data = data.round()
            data = numpy.clip(data, q_min_max["qmin"], q_min_max["qmax"])
            data = data.astype(dtype)
            data = relay.const(data)
    else:
        raise NotImplementedError
    return data


def _dequantize_shift(node):
    data = node.args[0]
    scale = node.args[1].data.asnumpy().astype(numpy.float16)
    zero_point = node.args[2].data.asnumpy()
    axis = node.attrs.axis
    tmp = relay.frontend.common.infer_type(data)
    shape = tmp.checked_type.concrete_shape

    if axis != -1:
        tmp1 = [1] * len(shape)
        tmp2 = shape[axis]
        tmp1[axis] = tmp2
        scale = scale.reshape(tmp1)
        zero_point = zero_point.reshape(tmp1)

    new_scale = relay.const(scale)
    new_zero_point = relay.const(zero_point)

    if (zero_point == 0).all():
        data = relay.cast(data, "float16")
        data = relay.multiply(data, new_scale)
    else:
        data = relay.cast(data, "int32")
        data = relay.subtract(data, new_zero_point)
        data = relay.cast(data, "float16")
        data = relay.multiply(data, new_scale)
    return data


def print_name_help(node, multiplier):
    "just print helper"
    ori_call = node.args[0].args[0]
    name = "Constant"
    if isinstance(ori_call, relay.Call) and isinstance(ori_call.op, relay.Function):
        name = getattr(ori_call.op.attrs, "Composite")
        if not isinstance(name, str):
            name = name.value
    elif isinstance(ori_call, relay.Call):
        name = ori_call.op.name

    if multiplier == 0:
        print("multiplier: ", multiplier)
        print("quantize input is ", name)


def _requantize_shift(node, multiplier):
    """_requantize_shift"""
    # multiplier 0: accumulator->int8
    #            1:int8-requantize default:0
    dequantize = node.args[0]
    data = dequantize.args[0]

    # print_name_help(node, multiplier)

    d_scale = dequantize.args[1].data.asnumpy()
    d_zero = dequantize.args[2].data.asnumpy()
    d_axis = dequantize.attrs.axis
    tmp = relay.frontend.common.infer_type(data)
    shape = tmp.checked_type.concrete_shape

    q_scale = node.args[1].data.asnumpy()
    q_zero = node.args[2].data.asnumpy()
    q_dtype = node.attrs.out_dtype
    q_axis = node.attrs.axis
    q_min_max = _get_dtype_info(q_dtype)

    # ex q_dtype int24 -> int32
    if q_dtype.startswith("int"):
        bits = q_dtype[3:]
        if int(bits) <= 8:
            q_dtype = "int8"
        elif int(bits) <= 16:
            q_dtype = "int16"
        elif int(bits) <= 32:
            q_dtype = "int32"

    # todo shift_coef_max confirm
    mul_coef_max = 255
    shift_coef_max = 32
    if (
        isinstance(node.args[0].args[0], relay.Call)
        and (
            node.args[0].args[0].op.name
            in ["nn.bias_add", "nn.conv2d", "nn.dense", "nn.batch_matmul"]
        )
        and multiplier == 0
        and TARGET_NNP == "nnp400"
    ) or (multiplier == 0 and TARGET_NNP == "nnp320"):
        mul_coef_max = 32767
        shift_coef_max = 39
    elif multiplier == 0 and TARGET_NNP.startswith("nnp3"):
        mul_coef_max = 255
        shift_coef_max = 31
    elif multiplier == 1 and TARGET_NNP.startswith("nnp3"):
        mul_coef_max = 127
        shift_coef_max = 15
    elif multiplier == 1 and TARGET_NNP.startswith("nnp4"):
        mul_coef_max = 255
        # todo must confirm!!!
        shift_coef_max = 32

    if d_axis != -1:
        tmp1 = [1] * (len(shape) - d_axis)
        tmp2 = shape[d_axis]
        tmp1[0] = tmp2
        if d_scale.size > 1:
            d_scale = d_scale.reshape(tmp1)
            d_zero = d_zero.reshape(tmp1)

    if q_axis != -1:
        tmp1 = [1] * (len(shape) - q_axis)
        tmp2 = shape[q_axis]
        tmp1[0] = tmp2
        q_scale = q_scale.reshape(tmp1)
        q_zero = q_zero.reshape(tmp1)

    if (d_zero == 0).all() and (q_zero == 0).all():
        scale = d_scale / q_scale

        # if multiplier == 0:
        #     print("before scale size: ", d_scale.flatten().size)
        #     print("before scale:", d_scale.flatten())
        #     print("after scale:", q_scale.flatten())

        if scale.size > 1:
            scale[numpy.where(scale < 1e-10)] = 1e-10

        s_shape = scale.shape
        scale = scale.reshape(-1)
        new_scale = []
        all_new_b = []
        pos_val = []
        neg_val = []

        for one_s in scale:
            bit = 0
            v = one_s

            while v < mul_coef_max and bit <= shift_coef_max:
                v = v * 2
                bit = bit + 1

            bit = bit - 1
            if bit == -1:
                new_a = mul_coef_max
                bit = 0
                pos_val.append(0)
                neg_val.append(0)
            else:
                new_a = math.floor(one_s * (2 ** bit) + 0.5)
                pos_val.append(2 ** (bit - 1))
                neg_val.append(2 ** (bit - 1) - 1)

            new_scale.append(new_a)
            all_new_b.append(bit)

        # todo now nnp400 only support int64
        if "ir_pass" not in relay.__dict__ and TARGET_NNP == "nnp400":
            new_scale = numpy.array(new_scale, "int64").reshape(s_shape)
            new_scale = relay.const(new_scale, "int64")
            all_new_b = numpy.array(all_new_b, "int64").reshape(s_shape)
            all_new_b = relay.const(all_new_b, "int64")

            data = relay.cast(data, "int64")
            data = relay.multiply(data, new_scale)

            rounding = "UPWARD"

            if rounding == "TONEAREST":
                all_ones_int64 = numpy.ones(shape=shape, dtype="int64")
                pos_val = numpy.array(pos_val, "int64").reshape(s_shape)
                pos_val = all_ones_int64 * pos_val
                pos_val = relay.const(pos_val, "int64")

                neg_val = numpy.array(neg_val, "int64").reshape(s_shape)
                neg_val = all_ones_int64 * neg_val
                neg_val = relay.const(neg_val, "int64")

                zeros = relay.zeros(shape=shape, dtype="int64")
                cond = relay.greater_equal(data, zeros)
                where = relay.where(cond, pos_val, neg_val)
                add = relay.add(data, where)
                data = relay.right_shift(add, all_new_b)

                data = relay.clip(data, q_min_max["qmin"], q_min_max["qmax"])
                data = relay.cast(data, q_dtype)

            elif rounding == "UPWARD":
                data = round_right_shift(data, all_new_b)
                data = relay.clip(data, q_min_max["qmin"], q_min_max["qmax"])
                data = relay.cast(data, q_dtype)
            else:
                raise NotImplementedError

        elif (
            "ir_pass" in relay.__dict__
            and TARGET_NNP.startswith("nnp3")
            and TARGET_NNP != "nnp320"
            and multiplier == 0
        ):
            new_scale = numpy.array(new_scale).reshape(s_shape)
            new_scale = relay.const(new_scale.astype("uint8"))
            all_new_b = numpy.array(all_new_b).reshape(s_shape)
            all_new_b = relay.const(all_new_b.astype("int32"))
            data = relay.multiply(data, new_scale, out_dtype="int32")
            data = relay.round_right_shift(data, all_new_b, out_dtype="int32")
            data = relay.clip(data, q_min_max["qmin"], q_min_max["qmax"], out_dtype=q_dtype)

        elif "ir_pass" in relay.__dict__ and TARGET_NNP == "nnp320" and multiplier == 0:
            new_scale = numpy.array(new_scale).reshape(s_shape)
            new_scale = relay.const(new_scale.astype("int16"))
            all_new_b = numpy.array(all_new_b).reshape(s_shape)
            all_new_b = relay.const(all_new_b.astype("int32"))
            data = relay.multiply(data, new_scale, out_dtype="int64")
            data = relay.round_right_shift(data, all_new_b, out_dtype="int64")
            data = relay.clip(data, q_min_max["qmin"], q_min_max["qmax"], out_dtype=q_dtype)

        elif "ir_pass" in relay.__dict__ and TARGET_NNP.startswith("nnp3") and multiplier == 1:
            new_scale = numpy.array(new_scale).reshape(s_shape)
            new_scale = relay.const(new_scale.astype("int8"))
            all_new_b = numpy.array(all_new_b).reshape(s_shape)
            all_new_b = relay.const(all_new_b.astype("int32"))
            data = relay.multiply(data, new_scale, out_dtype="int32")
            data = relay.round_right_shift(data, all_new_b, out_dtype="int32")
            data = relay.clip(data, q_min_max["qmin"], q_min_max["qmax"], out_dtype=q_dtype)

        elif "ir_pass" not in relay.__dict__ and TARGET_NNP.startswith("nnp3"):
            assert 0, "new tvm no support detvm requantize"

        # if multiplier == 0:
        #     print("new_scale:", new_scale.data.asnumpy().flatten()[:])
        #     print("new_b:", all_new_b.data.asnumpy().flatten()[:])

    else:
        raise NotImplementedError
    return data


def convert_operate(op_type, node, output_config, input_config, multiplier):
    """convert_operate"""
    if "ir_pass" not in relay.__dict__:
        quantize = is_op("qnn.quantize")(wildcard(), wildcard(), wildcard())
        dequantize = is_op("qnn.dequantize")(wildcard(), wildcard(), wildcard())
        requantize = is_op("qnn.quantize")(dequantize, wildcard(), wildcard())

        if requantize.match(node):
            node = _requantize_shift(node, multiplier)
        elif quantize.match(node):
            node = _quantize_shift(node)
        elif dequantize.match(node):
            node = _dequantize_shift(node)
        else:
            pass
    else:
        if (
            isinstance(node, relay.Call)
            and node.op.name == "qnn.quantize"
            and isinstance(node.args[0], relay.Call)
            and node.args[0].op.name == "qnn.dequantize"
        ):
            node = _requantize_shift(node, multiplier)

        elif isinstance(node, relay.Call) and node.op.name == "qnn.quantize":
            node = _quantize_shift(node)

        elif isinstance(node, relay.Call) and node.op.name == "qnn.dequantize":
            node = _dequantize_shift(node)
        else:
            pass
    return node


class Realize(ExprMutator):
    """realize"""

    def __init__(self, mod, vertex_config):
        super().__init__()
        self.mod = mod
        self.vertex_config = vertex_config
        self.new2old = {}
        self.idx = -1
        if not isinstance(self.mod, relay.Function):
            func = self.visit(self.mod["main"])
            quantized_mod = tvm.IRModule.from_expr(func)
            self.quantized_mod = relay.transform.InferType()(quantized_mod)
        else:
            func = self.visit(self.mod)
            self.quantized_mod = func

    def visit_call(self, call):
        new_args = [self.visit(arg) for arg in call.args]
        # compatible with nnp300
        if "ir_pass" not in relay.__dict__:
            new_call = relay.Call(call.op, new_args, call.attrs, call.type_args, call.span)
        else:
            new_call = relay.Call(call.op, new_args, call.attrs, call.type_args)

        if isinstance(call.op, relay.Function):
            name = getattr(call.op.attrs, "Composite")
        else:
            name = call.op.name

        self.idx = self.idx + 1

        LOGGER.info("[realize] idx is %d << %s >> ", self.idx, name)

        new_call = self.vertex_config[call].realize(
            call, new_call, self.vertex_config, self.new2old
        )
        return new_call

    def visit_tuple(self, tup):
        new_tup = super().visit_tuple(tup)

        self.idx = self.idx + 1

        config = self.vertex_config[tup]
        LOGGER.info(
            "[realize]...idx is %d tuplenode... and quantized is %d", self.idx, config.quantized
        )

        realized_args = []
        for old_arg, new_arg in zip(tup.fields, new_tup.fields):

            new_arg = _realize_core(config, old_arg, new_arg, self.vertex_config, self.new2old)

            realized_args.append(new_arg)

        # if the input with same dtype and no_quantized
        # do nothing, ex all int32 centernet
        dtype_list = []
        for arg_ in realized_args:
            dtype_list.append(relay.frontend.common.infer_type(arg_).checked_type.dtype)

        realized_args_new = []
        if not config.quantized and len(set(dtype_list)) != 1:
            for old_arg, new_arg in zip(tup.fields, realized_args):
                tmp = relay.frontend.common.infer_type(new_arg)
                if isinstance(new_arg, relay.Constant) and tmp.checked_type.dtype != "float16":
                    new_arg = relay.const(new_arg.data.asnumpy().astype("float16"))
                elif tmp.checked_type.dtype in ["int8", "int16"]:
                    new_arg = operate("dequantize", new_arg, config.input_config[old_arg], {}, True)
                elif tmp.checked_type.dtype != "float16" and config.output_config["ref_count"] > 0:
                    # output node no need to cast
                    new_arg = relay.cast(new_arg, "float16")
                realized_args_new.append(new_arg)

            if "ir_pass" not in relay.__dict__:
                new_tup = relay.Tuple(realized_args_new, tup.span)
            else:
                new_tup = relay.Tuple(realized_args_new)

        else:
            if "ir_pass" not in relay.__dict__:
                new_tup = relay.Tuple(realized_args, tup.span)
            else:
                new_tup = relay.Tuple(realized_args)

        return new_tup

    def visit_function(self, fn):
        new_fn = super().visit_function(fn)

        config = self.vertex_config[fn.body]
        if config.quantized and isinstance(fn.body, relay.Call):

            final_config = {}
            if config.output_config["quantized_axis"] != "none":
                final_config["scale"] = config.output_config["quantized_scale"]
                final_config["zero_point"] = config.output_config["quantized_zero_point"]
                final_config["axis"] = config.output_config["quantized_axis"]
                final_config["dtype"] = DataType.Int8
                # config.output_config['operate'] may be none
                # config.output_config['operate'] = "dequantize"
                new_body = operate(
                    config.output_config["operate"],
                    new_fn.body,
                    config.output_config,
                    final_config,
                    True,
                )
            else:
                new_body = new_fn.body
                final_config["scale"] = config.output_config["scale"]
                final_config["zero_point"] = config.output_config["zero_point"]
                final_config["axis"] = config.output_config["axis"]

            new_body = operate("dequantize", new_body, final_config, {}, True)
            pair_node(fn.body, new_body, {}, {"operate": "none"}, self.new2old, False)

        elif config.quantized and isinstance(fn.body, relay.Tuple):
            tup_new_fields = []
            for arg, new_arg in zip(fn.body.fields, new_fn.body.fields):
                final_config = {}

                final_config["scale"] = config.input_config[arg]["scale"]
                final_config["zero_point"] = config.input_config[arg]["zero_point"]
                final_config["axis"] = config.input_config[arg]["axis"]

                new_arg = operate("dequantize", new_arg, final_config, {}, True)
                pair_node(arg, new_arg, {}, {"operate": "none"}, self.new2old, False)

                tup_new_fields.append(new_arg)

            if "ir_pass" not in relay.__dict__:
                new_body = relay.Tuple(tup_new_fields, fn.body.span)
            else:
                new_body = relay.Tuple(tup_new_fields)

            pair_node(fn.body, new_body, {}, {"operate": "none"}, self.new2old, False)
        else:

            new_body = new_fn.body
            # todo tuple uncomment this, yolov5 run similarity assert
            # pair_node(fn.body, new_body, {}, {"operate": "none"}, self.new2old, False)

        new_body = relay.frontend.common.infer_type(new_body)

        if "analysis" in relay.__dict__:
            new_params = relay.analysis.free_vars(new_body)
        else:
            new_params = relay.ir_pass.free_vars(new_body)

        new_fn = relay.Function(
            new_params, new_body, new_body.checked_type, new_fn.type_params, new_fn.attrs
        )

        return new_fn


def realize_graph(cls):
    LOGGER.info("[realize] start......")
    global TARGET_NNP
    if "target" in cls.config:
        TARGET_NNP = cls.config["target"]

    realize = Realize(cls.pre_processed_mod, cls.vertex_config)
    LOGGER.info("[realize] finish ")
    cls.post_processed_mod = realize.quantized_mod
    cls.new2old = realize.new2old
