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
# pylint: disable=unused-argument,inconsistent-return-statements,bad-continuation,arguments-differ
"""config"""

import logging
import collections
from tvm import relay
from tvm.relay.expr_functor import ExprVisitor
from .threshold import Threshold
from .method_dtype import Method, DataType
from .op_config import OPCONFIGS

LOGGER = logging.getLogger("quantize")

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
]

# if fixed-point, two inputs identity scale
FIXED_OP_TWOARGS_LIST = [
    "add",
    "subtract",
    "maximum",
    "minimum",
    "nn.bias_add",
]


def helper_get_call_name(call_expr):
    if isinstance(call_expr.op, relay.Function):
        name = getattr(call_expr.op.attrs, "Composite")
        if not isinstance(name, str):
            name = name.value
    else:
        name = call_expr.op.name
    return name


class ConfigSpace(ExprVisitor):
    """ConfigSpace"""

    def __init__(self, mod, node_id, quantize_config, net_in_dtype):
        super().__init__()
        self.node_id = node_id
        self.config = collections.OrderedDict()
        self.all_op = []
        self.exp_ref = {}
        self.idx = -1
        self.quantize_config = {} if quantize_config is None else quantize_config
        self.net_in_dtype = net_in_dtype
        self.op_like_conv_count = 0
        # compatible with nnp300
        if not isinstance(mod, relay.Function):
            self.visit(mod["main"])
        else:
            self.visit(mod)

    @staticmethod
    def get_constant_arg_config(arg_idx, name, quantize_config):
        """get constant config"""
        dtype_tmp = DataType.Int8
        cond1 = name == "nn.bias_add" or arg_idx == 2
        cond2 = (
            name == "nn.bias_add"
            or arg_idx == 2
            and "target" in quantize_config
            and quantize_config["target"].startswith("nnp3")
        )

        if cond1:
            dtype_tmp = DataType.Int32
        if cond2:
            dtype_tmp = DataType.Int24

        arg_dict = {
            "default_config": {
                "threshold": Threshold.MinMax,
                "method": Method.Symmetry,
                "dtype": dtype_tmp,
            }
        }

        return arg_dict

    @staticmethod
    def get_call_arg_config(arg, arg_id, net_in_dtype, config, quantize_config):
        """get call config"""
        # first get from op config
        if isinstance(arg, relay.Call):
            arg_name = helper_get_call_name(arg)
            if arg_name in FIXED_OP_TWOARGS_LIST:
                tmp_dict = OPCONFIGS["FixedOpTwoArgs"].get_config(arg, config)
            elif arg_name in IDENTITY_OP_LIST:
                tmp_dict = OPCONFIGS["identity_op"].get_config(arg, config)
            elif arg_name in OPCONFIGS:
                tmp_dict = OPCONFIGS[arg_name].get_config(arg, config)
            else:
                tmp_dict = OPCONFIGS["float_op"].get_config(arg, config)

        else:
            tmp_dict = {
                "default_config": {
                    "threshold": Threshold.L2Norm,
                    "method": Method.Symmetry,
                    "dtype": DataType.Int8,
                }
            }

        if isinstance(arg, relay.Var) and net_in_dtype != "float32":
            tmp_dict["default_config"]["dtype"] = net_in_dtype

        map_dict = {
            "min_max": Threshold.MinMax,
            "percentile": Threshold.Percentile,
            "l2norm": Threshold.L2Norm,
            "kld": Threshold.KLDAbs,
        }
        if "calib_method" in quantize_config:
            if quantize_config["calib_method"].startswith("percentile"):
                tmp_dict["default_config"]["threshold"] = quantize_config["calib_method"]
            else:
                tmp_dict["default_config"]["threshold"] = map_dict[quantize_config["calib_method"]]

        # then "op_type_config"/"op_idx_config"/"op_name_config"
        if "op_idx_config" in quantize_config and str(arg_id) in quantize_config:
            tmp_dict["defalut_config"]["threshold"] = map_dict[
                quantize_config[str(arg_id)]["calib_method"]
            ]

        return tmp_dict

    def get_whether_quantized(self, quantize_config, name, call_id, tmp, call, config):
        """get whether quantized"""
        skip_conv_layers_name = [
            "nn.conv2d",
            "conv2d_bias_add",
            "nn.conv2d_transpose",
            "conv2d_transpose_bias_add",
            "nn.contrib_conv2d_winograd1d_without_weight_transform",
            "conv2d_winograd1d_bias_add",
            "nn.conv3d",
            "conv3d_bias_add",
        ]

        # float list
        cond1 = False
        if "float_list" in quantize_config:
            for value in quantize_config["float_list"]:
                if (isinstance(value, str) and name == value) or (
                    isinstance(value, int) and value == call_id
                ):
                    cond1 = True
                elif isinstance(value, list) and (name in value or call_id in value):
                    cond1 = True
        if cond1:
            config[tmp]["quantized"] = False

        # skip_conv_layers
        if (
            "skip_conv_layers" in quantize_config
            and quantize_config["skip_conv_layers"] is not None
        ):
            cnt = self.op_like_conv_count

            if name in skip_conv_layers_name:
                LOGGER.info("can config skip_conv_layers, the call is %s, cnt is %d", name, cnt)
                LOGGER.info("      weightshape is %s", str(call.args[1].data.shape))
                if cnt in quantize_config["skip_conv_layers"]:
                    config[tmp]["quantized"] = False
                self.op_like_conv_count = self.op_like_conv_count + 1
            elif cnt - 1 in quantize_config["skip_conv_layers"]:
                config[tmp]["quantized"] = False

        # nnp3xx multiply use fp16
        if (
            "target" in quantize_config
            and quantize_config["target"].startswith("nnp3")
            and name == "multiply"
        ):
            config[tmp]["quantized"] = False

        # equal use fp16
        if name == "equal":
            config[tmp]["quantized"] = False
            for arg in call.args:
                tmp_arg = self.node_id[arg]
                config[tmp_arg]["quantized"] = False

    def visit_call(self, call):
        for arg in call.args:
            self.visit(arg)

        name = helper_get_call_name(call)

        self.idx = self.idx + 1
        LOGGER.debug("[config] call %s %d", name, self.idx)
        if name not in self.all_op:
            self.all_op.append(name)

        tmp = self.node_id[call]
        call_id_prefix = int(tmp.split("_")[0])
        self.config[tmp] = {"valid_config": {}, "default_config": {}}

        arg_idx = -1
        for arg in call.args:
            arg_node_id = self.node_id[arg]

            arg_idx = arg_idx + 1
            arg_key = "input" + str(arg_idx)

            if isinstance(arg, relay.Constant):
                tmp_arg_dict = self.get_constant_arg_config(arg_idx, name, self.quantize_config)
            elif isinstance(arg, (relay.Call, relay.Var, relay.TupleGetItem)):
                tmp_arg_dict = self.get_call_arg_config(
                    arg, arg_node_id, self.net_in_dtype, self.config, self.quantize_config
                )
            else:
                tmp_arg_dict = {
                    "default_config": {
                        "threshold": Threshold.L2Norm,
                        "method": Method.Symmetry,
                        "dtype": DataType.Int8,
                    }
                }
            self.config[tmp]["valid_config"].update(
                {arg_key: tmp_arg_dict["default_config"] if "valid_config" in tmp_arg_dict else {}}
            )
            self.config[tmp]["default_config"].update({arg_key: tmp_arg_dict["default_config"]})

        self.get_whether_quantized(
            self.quantize_config, name, call_id_prefix, tmp, call, self.config
        )

    def visit_tuple(self, tup):

        for arg in tup.fields:
            self.visit(arg)

        tmp = self.node_id[tup]
        name = "Tuple"
        tuple_id_prefix = int(tmp.split("_")[0])

        self.idx = self.idx + 1
        LOGGER.debug("[config] tuple %d", self.idx)
        self.config[tmp] = {"valid_config": {}, "default_config": {}}

        arg_idx = -1
        for arg in tup.fields:

            arg_node_id = self.node_id[arg]

            arg_idx = arg_idx + 1
            arg_key = "input" + str(arg_idx)

            if isinstance(arg, relay.Constant):
                tmp_arg_dict = self.get_constant_arg_config(arg_idx, name, self.quantize_config)
            elif isinstance(arg, (relay.Call, relay.Var, relay.TupleGetItem)):
                tmp_arg_dict = self.get_call_arg_config(
                    arg, arg_node_id, self.net_in_dtype, self.config, self.quantize_config
                )
            else:
                tmp_arg_dict = {
                    "default_config": {
                        "threshold": Threshold.L2Norm,
                        "method": Method.Symmetry,
                        "dtype": DataType.Int8,
                    }
                }
            self.config[tmp]["valid_config"].update(
                {arg_key: tmp_arg_dict["default_config"] if "valid_config" in tmp_arg_dict else {}}
            )
            self.config[tmp]["default_config"].update({arg_key: tmp_arg_dict["default_config"]})

        if "float_list" in self.quantize_config:
            self.get_whether_quantized(
                self.quantize_config, name, tuple_id_prefix, tmp, None, self.config
            )


def config_space(cls):
    tmp = ConfigSpace(cls.pre_processed_mod, cls.node_id, cls.quantize_config, cls.net_in_dtype)
    cls.config_space = tmp.config
    cls.all_op = tmp.all_op
