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
# pylint: disable=unused-argument,inconsistent-return-statements,bad-continuation,too-many-function-args,arguments-differ,simplifiable-if-statement
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

REF_CNT_G = {}

# todo add more ops
# only support per-tensor
DISABLE_PERCHANNEL_OP = [
    "identity_op",
    "strided_slice",
    "expand_dims",
    "reshape",
    "squeeze",
    "transpose",
    "nn.batch_flatten",
    "vision.yolo_reorg",
    "tile",
    "split",
    "reverse",
    "take",
    "gather_nd",
    "nn.depth_to_space",
    "nn.space_to_depth",
    "nn.pad",
]

# the input and output same dtype
IDENTITY_INPUT_DTYPE_OP = [
    "identity_op",
    "strided_slice",
    "expand_dims",
    "reshape",
    "squeeze",
    "transpose",
    "nn.batch_flatten",
    "vision.yolo_reorg",
    "tile",
    "reverse",
    "zeros_like",
    "ones_like",
    "broadcast_to",
    "take",
    "gather_nd",
    "slice_like",
    "nn.pad",
    "image.resize",
    "image.resize2d",
    "nn.leaky_relu",
    "nn.prelu",
    "nn.upsampling",
    "nn.unmaxpool_upsample",
    "split",
    "nn.depth_to_space",
    "nn.space_to_depth",
]

# the in-out axis should be same and nodo requantize
IDENTITY_AXIS_OP = [
    "nn.leaky_relu",
    "nn.relu",
    "nn.prelu",
    "clip",
    "nn.upsampling",
    "nn.unmaxpool_upsample",
    "strided_slice",
    "nn.pad",
    "image.resize",
    "image.resize2d",
    "contrib.adaptive_max_pool2d",
    "contrib.adaptive_avg_pool2d",
]

# only shape change
IDENTITY_OP_LIST = [
    "expand_dims",
    "reshape",
    "squeeze",
    "transpose",
    "nn.batch_flatten",
    "vision.yolo_reorg",
    "tile",
    "reverse",
    "zeros_like",
    "ones_like",
    "broadcast_to",
    "nn.depth_to_space",
    "nn.space_to_depth",
]

# can't use int8
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
    "maximum",
    "minimum",
]

# if fixed-point, two inputs identity scale
FIXED_OP_TWOARGS_LIST = [
    "add",
    "subtract",
    "nn.bias_add",
]


def recursive_identity_axis(input_axis, node, vertex_config):
    """recursive_identity_axis"""
    # for maxpool index is true
    if isinstance(node, relay.TupleGetItem):
        node = node.tuple_value

    if (
        isinstance(node, relay.Call)
        and not isinstance(node.op, relay.Function)
        and vertex_config[node].quantized
    ):

        if (
            node.op.name == "nn.max_pool2d" and REF_CNT_G[node.args[0]] == 1
        ) or node.op.name in IDENTITY_AXIS_OP:
            if vertex_config[node].output_config["axis"] not in [input_axis, -1]:
                vertex_config[node].output_config["axis"] = input_axis
                vertex_config[node].input_config[node.args[0]]["axis"] = input_axis

                if vertex_config[node.args[0]].quantized and vertex_config[
                    node.args[0]
                ].output_config["quantized_axis"] not in ["none", input_axis]:
                    vertex_config[node.args[0]].output_config["quantized_axis"] = input_axis

                recursive_identity_axis(input_axis, node.args[0], vertex_config)


def _quantized_judge(vertex_config, node, input_axis, quantized, config):
    """quantized judge"""

    # dtype set
    # arg not quantized and self not quantized: dtype:float16
    cond1 = (
        isinstance(node, relay.Var) and "dtype" in config and config["dtype"] in ["uint8", "int16"]
    )
    if cond1 or vertex_config[node].quantized or not vertex_config[node].quantized and quantized:
        dtype = config["dtype"]
    else:
        dtype = DataType.Float16

    input_config = {
        "dtype": dtype,
        "axis": input_axis,
        "method": None,
        "threshold": None,
        "operate": "none",
    }

    # three coditions do quantize
    # ----1, int32 input and quantized. some int32 case no need to do, ex cast.
    # ----2, fp16 to int8(as int8 to fp16, int8 must have his own scale)
    if (
        (
            vertex_config[node].output_config["dtype"] == DataType.Int32
            and vertex_config[node].quantized
        )
        or (not vertex_config[node].quantized and quantized)
        or cond1
    ):
        # check if input is already quantized(split-node may quantized already)
        # --the quantized_axis should be the smallest to all-possible axis
        # --notice! when do 'threshold' the input_axis is not really use axis,
        # ----------------  finally use the output_config['quantized_axis']

        vertex_config[node].output_config["ref_count"] = (
            vertex_config[node].output_config["ref_count"] + 1
        )

        if vertex_config[node].output_config["quantized_axis"] == "none":
            input_config.update(
                {
                    "method": config["method"],
                    "threshold": config["threshold"](node, input_axis, config),
                }
            )

            tmp = {}
            tmp["operate"] = "requantize"
            if not vertex_config[node].quantized and quantized or cond1:
                tmp["operate"] = "quantize"
            input_config.update(tmp)

            vertex_config[node].output_config["quantized_axis"] = input_axis

        # rule: node only quantized once
        # example:
        # --if the existed quantized-axis is perchannel and current conv2d input axis is pertensor
        # --then modify quantized_axis to input_axis(-1)
        elif vertex_config[node].output_config["quantized_axis"] > input_axis:
            vertex_config[node].output_config["quantized_axis"] = input_axis

    recursive_identity_axis(input_axis, node, vertex_config)

    if "DataType" in runtime_ctypes.__dict__:
        tvm_dtype = runtime_ctypes.DataType(input_config["dtype"])
    else:
        tvm_dtype = runtime_ctypes.TVMType(input_config["dtype"])
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

    # pool can use prechannel
    if (
        name.split("_")[-1].startswith("pool")
        and "layout" in node.attrs.keys()
        and node.attrs.layout in ["NCHW", "NHWC"]
    ):
        input0_axis = node.attrs.layout.find("C")

    if cls.name in DISABLE_PERCHANNEL_OP:
        input0_axis = -1

    if cls.name == "nn.pad" and node.attrs.pad_mode == "constant" and node.attrs.pad_value != 0:
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
    """var analyze"""

    def __init__(self, node, net_in_dtype):
        self.quantized = False
        if (
            isinstance(net_in_dtype, dict)
            and node.name_hint in net_in_dtype
            and net_in_dtype[node.name_hint] in ["uint8", "int16"]
        ) or (isinstance(net_in_dtype, str) and net_in_dtype in ["uint8", "int16"]):
            self.quantized = True

        self.output_config = {
            "dtype": node.checked_type.dtype,
            "axis": -1,
            "quantized_axis": "none",
            "ref_count": 0,
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
        else:
            self.quantized = False
            for arg in node.fields:
                if isinstance(arg, relay.Tuple) and vertex_config[arg].quantized:
                    vertex_config[arg].quantized = False

        if "quantized" in config:
            self.quantized = config["quantized"]

        output_axis = max(_axis)

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

            # for case concat(%1, %1, %1, %1)
            if node.fields[arg_dix] in self.input_config:
                continue

            self.input_config.update({node.fields[arg_dix]: input_config})

        # set output0_config
        output0_config = {
            "ref_count": 0,
            "dtype": config["input0"]["dtype"] if self.quantized else DataType.Float16,
            "axis": output_axis,
            "quantized_axis": "none",
        }

        if output0_config["dtype"].startswith("int"):
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
        if isinstance(arg, relay.Call) and arg.op.name not in [
            "split",
            "topk",
            "vision.multibox_transform_loc",
            "nn.max_pool2d",
        ]:
            LOGGER.info("[analyze] TupleGetitem %s ", arg.op.name)
        if self.quantized:
            dtype = DataType.Int8
        else:
            dtype = DataType.Float16

        input_config = {
            "dtype": dtype,
            "axis": input_axis,
            "method": None,
            "threshold": None,
            "operate": "none",
        }
        self.input_config = {arg: input_config}

        out_dtype = dtype
        if arg.op.name in ["nn.max_pool2d"] and self.quantized:
            out_dtype = "int32"

        # get output0_config
        output0_config = {
            "dtype": out_dtype,
            "axis": input_axis,
            "quantized_axis": "none",
            "ref_count": 0,
        }

        # if output0_config["dtype"] not in [DataType.Float16]:
        #     output0_config.update(_get_dtype_info(output0_config["dtype"]))  # todo modify this
        self.output_config = output0_config


class AnalyzeGraph(ExprVisitor):
    """analyze graph"""

    def __init__(self, mod, config, node_id, calibrate_num, net_in_dtype="uint8"):
        super().__init__()
        self.config = config
        self.idx = -1
        self.node_id = node_id
        self.calibrate_num = calibrate_num
        self.net_in_dtype = net_in_dtype
        self.collect_node = set()
        self.vertex_config = collections.OrderedDict()
        self.collect_result = collections.OrderedDict()
        if isinstance(mod, relay.Function):
            self.visit(mod)
        else:
            self.visit(mod["main"])

    def visit_call(self, call):
        for arg in call.args:
            self.visit(arg)

        if isinstance(call.op, relay.Function):
            name = getattr(call.op.attrs, "Composite")
            if not isinstance(name, str):
                name = name.value
        else:
            name = call.op.name

        self.idx = self.idx + 1
        LOGGER.info("[analyze] idx is %d << %s >> ", self.idx, name)

        config = self.config[self.node_id[call]]

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
            if (
                self.vertex_config[fn.body].output_config["dtype"] == DataType.Int32
                and self.vertex_config[fn.body].quantized
            ):
                threshold = Threshold.Percentile
                new_arg = {"calibrate_num": self.calibrate_num}
                for one_arg in threshold.args:
                    new_arg[one_arg["name"]] = one_arg["default"]
                config = {"threshold_arg": new_arg, "method": Method.Symmetry, "dtype": "int8"}
                self.vertex_config[fn.body].output_config["threshold"] = threshold(
                    fn.body, -1, config
                )
                self.collect_node.update([fn.body])
                self.vertex_config[fn.body].output_config["method"] = Method.Symmetry
                self.vertex_config[fn.body].output_config["operate"] = "requantize"
                self.vertex_config[fn.body].output_config["qmin"] = -128
                self.vertex_config[fn.body].output_config["qmax"] = 127
            else:
                self.vertex_config[fn.body].output_config["threshold"] = None
                self.vertex_config[fn.body].output_config["method"] = None
                self.vertex_config[fn.body].output_config["operate"] = "none"

        elif isinstance(fn.body, relay.Tuple):
            pass
            # assert 0, "no support f.body is tuple, please call yhh!!!"


class GetExprRefCount(ExprVisitor):
    """GetExprRefCount"""

    def __init__(self, mod):
        super().__init__()
        self.ret_ref_cnt = {}
        if isinstance(mod, relay.Function):
            self.visit(mod)
        else:
            self.visit(mod["main"])

    def visit_call(self, call):
        for arg in call.args:
            if arg in self.ret_ref_cnt:
                self.ret_ref_cnt[arg] = self.ret_ref_cnt[arg] + 1
            else:
                self.ret_ref_cnt[arg] = 1
            self.visit(arg)

    def visit_tuple(self, tup):
        for x in tup.fields:
            if x in self.ret_ref_cnt:
                self.ret_ref_cnt[x] = self.ret_ref_cnt[x] + 1
            else:
                self.ret_ref_cnt[x] = 1
            self.visit(x)

    def visit_tuple_getitem(self, t):
        if t in self.ret_ref_cnt:
            self.ret_ref_cnt[t] = self.ret_ref_cnt[t] + 1
        else:
            self.ret_ref_cnt[t] = 1
        self.visit(t.tuple_value)


def analyze_graph(cls):
    """analyze_graph"""
    ref_cnt = GetExprRefCount(cls.pre_processed_mod)
    global REF_CNT_G
    REF_CNT_G = ref_cnt.ret_ref_cnt

    tmp = AnalyzeGraph(
        cls.pre_processed_mod, cls.config, cls.node_id, cls.calibrate_num, cls.net_in_dtype
    )
    cls.vertex_config = tmp.vertex_config
    cls.collect_node = tmp.collect_node
    cls.collect_result = tmp.collect_result
