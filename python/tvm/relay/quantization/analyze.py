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
# pylint: disable=unused-argument,inconsistent-return-statements
"""analyze"""

import logging
import collections
from tvm import relay
from tvm.relay.expr_functor import ExprVisitor
from tvm._ffi import runtime_ctypes
from .op_config import OPCONFIGS
from .method_dtype import _get_dtype_info, DataType, Method
from .threshold import Threshold
from .realize import operate

LOGGER = logging.getLogger("quantize")


CONV_COUNTER = 0


def _conv_counter():
    """Get the global counter for conv2d."""
    return CONV_COUNTER


def _set_conv_counter(n):
    """Set the value of the global conv2d counter."""
    global CONV_COUNTER
    CONV_COUNTER = n


# todo add more ops
# only support per-tensor
DISABLE_PERCHANNEL_OP = [
    "identity_op",
    "expand_dims",
    "reshape",
    "squeeze",
    "transpose",
    "nn.batch_flatten",
    "yolo_reorg",
    "tile",
    "split",
    "reverse",
    "take",
]

# the input and output same dtype
IDENTITY_INPUT_DTYPE_OP = [
    "expand_dims",
    "reshape",
    "squeeze",
    "transpose",
    "nn.batch_flatten",
    "yolo_reorg",
    "tile",
    "split",
    "reverse",
    "take",
    "nn.pad",
    "image.resize",
    "nn.leaky_relu",
    "nn.prelu",
]

IDENTITY_OP_LIST = [
    "expand_dims",
    "reshape",
    "squeeze",
    "transpose",
    "nn.batch_flatten",
    "yolo_reorg",
    "tile",
    "reverse",
]

# only support fp16
FLOAT_OP_LIST = [
    "sigmoid",
    "nn.softmax",
    "erf",
    "log",
    "sqrt",
    "exp",
    "tanh",
    "mean",
    "divide",
    "floor_divide",
    "mod",
    "floor_mod",
    "power",
    "variance",
    "power",
]

# fixed-point, two inputs identity scale
FIXED_OP_TWOARGS_LIST = [
    "add",
    "subtract",
    "maximum",
    "minimum",
]


def _quantized_judge(vertex_config, node, input_axis, quantized, config):
    """quantized judge"""

    vertex_config[node].output_config["ref_count"] = (
        vertex_config[node].output_config["ref_count"] + 1
    )

    # todo consider more!!
    # arg not quantized and self not quantized: dtype:float16
    if vertex_config[node].quantized or not vertex_config[node].quantized and quantized:
        dtype = config["dtype"]
    else:
        dtype = DataType.Float16

    input_config = {
        # "in_dtype": vertex_config[node].output_config["dtype"],  # TODO can del?
        # "in_axis": vertex_config[node].output_config["axis"],
        "dtype": dtype,  # todo quantized 'dtype' according to bits-info
        "axis": input_axis,
        "method": None,
        "threshold": None,
        "operate": "none",
    }

    if isinstance(node, relay.Var) and vertex_config[node].output_config["net_in_dtype"] == "uint8":
        input_config.update({"dtype": DataType.UInt8})

    # make sure the identity dtype op can deal
    if (
        input_axis == -1
        and isinstance(node, relay.Call)
        and not isinstance(node.op, relay.Function)
        and node.op.name in IDENTITY_INPUT_DTYPE_OP
        and vertex_config[node.args[0]].output_config["quantized_axis"] != "none"
        and vertex_config[node.args[0]].output_config["quantized_axis"] > -1
    ):
        vertex_config[node.args[0]].output_config["quantized_axis"] = -1

    # three coditions do quantize
    # ----1, int32 input
    # ----2, fp16 to int8(as int8 to fp16, int8 must have his own scale)
    # ----3, perchannel != pertensor and not identity_dtype
    # then update the quantized_axis
    if (
        vertex_config[node].output_config["dtype"] == DataType.Int32
        or (not vertex_config[node].quantized and quantized)
        or (
            vertex_config[node].quantized
            and quantized
            and vertex_config[node].output_config["axis"] != input_axis
            and node.op.name not in IDENTITY_INPUT_DTYPE_OP
        )
    ):
        # check if input is already quantized(split-node may quantized already)
        # --the quantized_axis should be the smallest to all-possible axis
        # --notice! when do 'threshold' the input_axis is not really use axis,
        # ----------------  finally use the output_config['quantized_axis']
        if vertex_config[node].output_config["quantized_axis"] == "none":
            input_config.update(
                {
                    "method": config["method"],
                    "threshold": config["threshold"](node, input_axis, config),
                }
            )

            tmp = {}
            tmp["operate"] = "requantize"
            if not vertex_config[node].quantized and quantized:
                tmp["operate"] = "quantize"
            input_config.update(tmp)

            vertex_config[node].output_config["quantized_axis"] = input_axis

        # rule: node only quantized once
        # example:
        # --if the existed quantized-axis is perchannel and current conv2d input axis is pertensor
        # --then modify quantized_axis to input_axis(-1)
        elif vertex_config[node].output_config["quantized_axis"] > input_axis:
            vertex_config[node].output_config["quantized_axis"] = input_axis

    tvm_dtype = runtime_ctypes.DataType(input_config["dtype"])
    tvm_type = tvm_dtype.CODE2STR[tvm_dtype.type_code]

    if tvm_type in ["int", "uint"]:
        input_config.update(_get_dtype_info(input_config["dtype"]))

    return input_config


def oneargdeal(cls, node, vertex_config, ci0):
    """OneArgDeal get inputconfig outputconfig"""
    if isinstance(node.op, relay.Function):
        name = getattr(node.op.attrs, "Composite")
    else:
        name = node.op.name
    LOGGER.debug("[analyze] %s start...", name)

    arg = node.args[0]

    # get input0_config
    input0_axis = -1
    # todo consider if arg.quantize and not cls.quantized, axis set to -1
    if vertex_config[arg].quantized:
        input0_axis = vertex_config[arg].output_config["axis"]
    if (
        node.attrs is not None
        and "layout" in node.attrs.keys()
        and node.attrs.layout in ["NCHW", "NHWC"]
    ):
        input0_axis = node.attrs.layout.find("C")

    if cls.name in DISABLE_PERCHANNEL_OP:
        input0_axis = -1

    if cls.name == "nn.pad" and node.attrs.pad_mode == "constant" and node.attrs.pad_value != 0:
        input0_axis = -1

    # todo perch more consider (input_axis -> output_axis)
    if cls.name in FLOAT_OP_LIST:
        input0_axis = -1

    input0_config = _quantized_judge(vertex_config, node.args[0], input0_axis, cls.quantized, ci0)
    cls.input_config = {arg: input0_config}

    # get output0_config
    output0_config = {
        "dtype": DataType.Int32 if cls.quantized else DataType.Float16,
        "axis": input0_config["axis"],
        "quantized_axis": "none",
        "ref_count": 0,
    }
    if cls.name in IDENTITY_INPUT_DTYPE_OP:
        output0_config.update(
            {
                "dtype": input0_config["dtype"] if cls.quantized else DataType.Float16,
            }
        )

    cls.output_config = output0_config


class Common:
    """Common"""

    def __init__(self, node, vertex_config, config):
        self.quantized = False

        self.input_config = {}
        for arg in node.args:
            self.input_config[arg] = {}
            tmp1 = {}
            tmp1.update({"in_dtype": vertex_config[arg].output_config["dtype"]})
            tmp1.update({"in_axis": vertex_config[arg].output_config["axis"]})
            if vertex_config[arg].quantized:
                tmp2 = {"operate": "dequantize"}
            else:
                tmp2 = {"operate": "none"}
            tmp1.update(tmp2)
            self.input_config[arg].update(tmp1)

        self.output_config = {
            "dtype": node.checked_type.dtype,
            "axis": -1,
        }

    def realize(self, old_node, new_node, vertex_config, n2o):
        """realize"""
        realized_args = []

        for old_arg, new_arg in zip(old_node.args, new_node.args):
            output_config = vertex_config[old_arg].output_config
            input_config = self.input_config[old_arg]
            new_arg = operate(input_config["operate"], new_arg, output_config, input_config, True)
            realized_args.append(new_arg)

        new_node = relay.Call(
            old_node.op, realized_args, new_node.attrs, new_node.type_args, new_node.span
        )
        return new_node


class Var:
    def __init__(self, node, net_in_dtype="uint8"):
        self.quantized = False

        self.output_config = {
            "dtype": node.checked_type.dtype,
            "axis": -1,
            "quantized_axis": "none",
            "ref_count": 0,
            "net_in_dtype": net_in_dtype,
        }


class Constant:
    def __init__(self, node):
        self.quantized = False

        self.output_config = {
            "dtype": node.checked_type.dtype,
            "axis": -1,
            "quantized_axis": "none",
            "ref_count": 0,
        }


class Tuple:
    """Tuple"""

    def __init__(self, node, vertex_config, config):
        LOGGER.debug("[analyze] Tuplenode start")
        _dtype = []
        _axis = []
        _quantized = []

        for arg in node.fields:
            _dtype.append(vertex_config[arg].output_config["dtype"])
            _axis.append(vertex_config[arg].output_config["axis"])
            _quantized.append(vertex_config[arg].quantized)

        if all(_quantized):
            self.quantized = True
        # elif all(i is False for i in _quantized):
        #     self.quantized = False
        else:
            self.quantized = False

        output_axis = max(_axis)

        # todo need more consider!
        # tuple alse need to consider split node
        len_arg = len(node.fields)
        self.input_config = {}
        for arg_dix in range(len_arg):
            input_config = _quantized_judge(
                vertex_config,
                node.fields[arg_dix],
                _axis[arg_dix],
                self.quantized,
                config["input" + str(arg_dix)],
            )
            self.input_config.update({node.fields[arg_dix]: input_config})

        # set output0_config
        output0_config = {
            "dtype": config["input0"]["dtype"] if self.quantized else DataType.Float16,
            "axis": output_axis,
            "quantized_axis": "none",
            "ref_count": 0,
        }

        if output0_config["dtype"] not in [DataType.Float16]:
            output0_config.update(_get_dtype_info(output0_config["dtype"]))  # todo modify this
        self.output_config = output0_config
        LOGGER.debug("[analyze] Tuplenode end")


class TupleGetitem:
    """TupleGetitem"""

    def __init__(self, node, vertex_config):
        arg = node.tuple_value

        # todo consider more
        self.quantized = vertex_config[arg].quantized
        input_axis = vertex_config[arg].output_config["axis"]

        vertex_config[arg].output_config["ref_count"] = (
            vertex_config[arg].output_config["ref_count"] + 1
        )

        # todo consider more!!
        if isinstance(arg, relay.Call) and arg.op.name != "split":
            assert 0, "meet tupleGetitem no support, call yhh"
        if self.quantized:
            dtype = DataType.Int8
        else:
            dtype = DataType.Float16

        input_config = {
            # "in_dtype": vertex_config[arg].output_config["dtype"],
            # "in_axis": vertex_config[arg].output_config["axis"],
            "dtype": dtype,
            "bits": 8,
            "axis": input_axis,
            "method": None,
            "threshold": None,
            "operate": "none",
        }
        self.input_config = {arg: input_config}

        # get output0_config
        output0_config = {
            "dtype": dtype,
            "axis": input_axis,
            "quantized_axis": "none",
            "ref_count": 0,
        }

        # if output0_config["dtype"] not in [DataType.Float16]:
        #     output0_config.update(_get_dtype_info(output0_config["dtype"]))  # todo modify this
        self.output_config = output0_config


class AnalyzeGraph(ExprVisitor):
    """analyze graph"""

    def __init__(self, mod, config, node_id, net_in_dtype="uint8"):
        super().__init__()
        self.config = config
        self.idx = -1
        self.node_id = node_id
        self.net_in_dtype = net_in_dtype
        self.collect_node = set()
        self.vertex_config = collections.OrderedDict()
        self.collect_result = collections.OrderedDict()
        self.visit(mod["main"])

    def visit_call(self, call):
        for arg in call.args:
            self.visit(arg)

        if isinstance(call.op, relay.Function):
            name = getattr(call.op.attrs, "Composite")
        else:
            name = call.op.name

        self.idx = self.idx + 1
        LOGGER.info("[analyze] idx is %d ....", self.idx)

        config = self.config[self.node_id[call]]
        if "skip_conv_layers" in self.config:
            config["skip_conv_layers"] = self.config["skip_conv_layers"]

        if name in FIXED_OP_TWOARGS_LIST:
            self.vertex_config[call] = OPCONFIGS["FixedOpTwoArgs"](call, self.vertex_config, config)
        elif name in IDENTITY_OP_LIST:
            self.vertex_config[call] = OPCONFIGS["identity_op"](call, self.vertex_config, config)
        elif name in OPCONFIGS:
            self.vertex_config[call] = OPCONFIGS[name](call, self.vertex_config, config)
        else:
            self.vertex_config[call] = OPCONFIGS["float_op"](call, self.vertex_config, config)

        tmp = []
        for arg in call.args:
            # if self.vertex_config[call].quantized:
            if self.vertex_config[call].input_config[arg]["threshold"] is not None:
                if isinstance(arg, relay.Constant):
                    self.collect_result[arg] = arg.data.asnumpy()
                else:
                    tmp.append(arg)

        # no need to add call, ex: split
        # if self.vertex_config[call].quantized:
        #   self.collect_node.update(tmp + [call])
        self.collect_node.update(tmp)

    def visit_var(self, var):
        self.vertex_config[var] = Var(var, self.net_in_dtype)

    def visit_constant(self, const):
        self.vertex_config[const] = Constant(const)

    def visit_tuple(self, tup):
        for x in tup.fields:
            self.visit(x)

        config = self.config[self.node_id[tup]]

        self.vertex_config[tup] = Tuple(tup, self.vertex_config, config)

        tmp = []
        for arg in tup.fields:
            if self.vertex_config[tup].input_config[arg]["threshold"] is not None:
                if isinstance(arg, relay.Constant):
                    self.collect_result[arg] = arg.data.asnumpy()
                else:
                    tmp.append(arg)
        self.collect_node.update(tmp)

    def visit_tuple_getitem(self, t):
        self.visit(t.tuple_value)

        self.vertex_config[t] = TupleGetitem(t, self.vertex_config)

    def visit_function(self, fn):
        super().visit_function(fn)

        # support final node int32 to int8
        if isinstance(fn.body, relay.Call):
            assert (
                fn.body in self.vertex_config
            ), "fn body is call should already in self.vertex_config"
            self.vertex_config[fn.body].output_config["is_fn_body"] = True
            # should consider int8 input
            # todo more configuration!!
            if self.vertex_config[fn.body].output_config["dtype"] == DataType.Int32:
                self.vertex_config[fn.body].output_config["threshold"] = Threshold.RelativeEntropy(
                    fn.body, -1, {}
                )
                self.collect_node.update([fn.body])
                self.vertex_config[fn.body].output_config["method"] = Method.Symmetry
                self.vertex_config[fn.body].output_config["operate"] = "requantize"
                self.vertex_config[fn.body].output_config["qmax"] = 127
            else:
                self.vertex_config[fn.body].output_config["threshold"] = None
                self.vertex_config[fn.body].output_config["method"] = None
                self.vertex_config[fn.body].output_config["operate"] = "none"

        elif isinstance(fn.body, relay.Tuple):
            pass
            # assert 0, "no support f.body is tuple, please call yhh!!!"


def analyze_graph(cls):
    tmp = AnalyzeGraph(cls.pre_processed_mod, cls.config, cls.node_id, cls.net_in_dtype)
    cls.vertex_config = tmp.vertex_config
    cls.collect_node = tmp.collect_node
    cls.collect_result = tmp.collect_result
