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


class ConfigSpace(ExprVisitor):
    """ConfigSpace"""

    def __init__(self, mod, node_id, quantize_config):
        super().__init__()
        self.node_id = node_id
        self.config = collections.OrderedDict()
        self.all_op = []
        self.exp_ref = {}
        self.idx = -1
        self.quantize_config = {} if quantize_config is None else quantize_config
        # compatible with nnp300
        if not isinstance(mod, relay.Function):
            self.visit(mod["main"])
        else:
            self.visit(mod)

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
        LOGGER.debug("[config] call %s %d", name, self.idx)
        if name not in self.all_op:
            self.all_op.append(name)

        tmp = self.node_id[call]

        self.config[tmp] = {"valid_config": {}, "default_config": {}}

        arg_idx = -1
        for arg in call.args:
            id_node = self.node_id[arg]

            id_node_prefix = str(id_node.split("_")[0])

            arg_idx = arg_idx + 1
            arg_key = "input" + str(arg_idx)
            arg_dict = {"valid_config": {}, "default_config": {}}

            if id_node not in self.exp_ref:
                self.exp_ref[id_node] = [tmp + "-" + arg_key]
            else:
                self.exp_ref[id_node].append(tmp + "-" + arg_key)

            # todo support 3-10
            if id_node_prefix in self.quantize_config:
                arg_dict["default_config"] = {arg_key: {}}
                arg_dict["default_config"][arg_key].update(self.quantize_config[id_node_prefix])

            elif isinstance(arg, relay.Var) and "all" in self.quantize_config:
                arg_dict["default_config"] = {arg_key: {}}
                arg_dict["default_config"][arg_key].update(self.quantize_config["all"])

            elif isinstance(arg, relay.Var):
                arg_dict["valid_config"] = {
                    arg_key: {
                        "threshold": (
                            Threshold.MinMax,
                            Threshold.Percentile,
                            Threshold.MovingAverageMinMax,
                            Threshold.L2Norm,
                            Threshold.RelativeEntropy,
                        ),
                        "method": (Method.Symmetry, Method.Asymmetry),
                        "dtype": (DataType.UInt8, DataType.Int8),
                    }
                }

                arg_dict["default_config"] = {
                    arg_key: {
                        "threshold": Threshold.L2Norm,
                        "method": Method.Symmetry,
                        "dtype": DataType.Int8,
                    }
                }

            elif isinstance(arg, relay.Constant) and "constant" in self.quantize_config:
                arg_dict["default_config"] = {arg_key: {}}
                arg_dict["default_config"][arg_key].update(self.quantize_config["constant"])

            elif isinstance(arg, relay.Constant):
                if arg_idx != 2:
                    arg_dict["valid_config"] = {
                        arg_key: {
                            "threshold": (
                                Threshold.MinMax,
                                Threshold.Percentile,
                                Threshold.MovingAverageMinMax,
                                Threshold.L2Norm,
                                Threshold.RelativeEntropy,
                            ),
                            "method": (Method.Symmetry, Method.Asymmetry),
                            "dtype": (DataType.Int8,),
                        }
                    }
                    arg_dict["default_config"] = {
                        arg_key: {
                            "threshold": Threshold.MinMax,
                            "method": Method.Symmetry,
                            "dtype": DataType.Int8,
                        }
                    }
                else:
                    # must be bias!
                    arg_dict["valid_config"] = {
                        arg_key: {
                            "threshold": (
                                Threshold.MinMax,
                                Threshold.Percentile,
                                Threshold.MovingAverageMinMax,
                                Threshold.L2Norm,
                                Threshold.RelativeEntropy,
                            ),
                            "method": (Method.Symmetry, Method.Asymmetry),
                            "dtype": (DataType.Int32,),
                        }
                    }
                    arg_dict["default_config"] = {
                        arg_key: {
                            "threshold": Threshold.MinMax,
                            "method": Method.Symmetry,
                            "dtype": DataType.Int32,
                        }
                    }

            elif isinstance(arg, relay.Call) and "call" in self.quantize_config:

                arg_dict["default_config"] = {arg_key: {}}
                arg_dict["default_config"][arg_key].update(self.quantize_config["call"])

            elif isinstance(arg, relay.Call) and "all" in self.quantize_config:
                arg_dict["default_config"] = {arg_key: {}}
                arg_dict["default_config"][arg_key].update(self.quantize_config["all"])

            elif isinstance(arg, relay.Call):
                if isinstance(arg.op, relay.Function):
                    name = getattr(arg.op.attrs, "Composite")
                else:
                    name = arg.op.name

                tmp_dict = {}
                if name in FIXED_OP_TWOARGS_LIST:
                    tmp_dict = OPCONFIGS["FixedOpTwoArgs"].get_config(call, self.config)
                elif name in IDENTITY_OP_LIST:
                    tmp_dict = OPCONFIGS["identity_op"].get_config(call, self.config)
                elif name in OPCONFIGS:
                    tmp_dict = OPCONFIGS[name].get_config(call, self.config)
                else:
                    tmp_dict = OPCONFIGS["float_op"].get_config(call, self.config)

                arg_dict["default_config"] = {arg_key: tmp_dict["default_config"]}
                arg_dict["valid_config"] = {arg_key: tmp_dict["valid_config"]}

            else:
                arg_dict = {
                    "valid_config": {
                        arg_key: {
                            "threshold": (
                                Threshold.MinMax,
                                Threshold.Percentile,
                                Threshold.MovingAverageMinMax,
                                Threshold.L2Norm,
                                Threshold.RelativeEntropy,
                            ),
                            "method": (Method.Symmetry, Method.Asymmetry),
                            "dtype": (DataType.Int8, DataType.Int16),
                        }
                    },
                    "default_config": {
                        arg_key: {
                            "threshold": Threshold.L2Norm,
                            "method": Method.Symmetry,
                            "dtype": DataType.Int8,
                        }
                    },
                }

            # int16 config must from outside
            # only nu [conv2d dense] input can support fp16
            # conv2d + relu clip can support "int16"
            # todo maxpool + relu only support int8
            # todo only support conv2d_bias int16
            if isinstance(arg, relay.Call) and isinstance(arg.op, relay.Function):
                arg_name = getattr(arg.op.attrs, "Composite")
            elif isinstance(arg, relay.Call):
                arg_name = arg.op.name
            if (
                arg_dict["default_config"][arg_key]["dtype"] == "int16"
                and (
                    name
                    not in [
                        "conv2d_bias_add",
                        "nn.conv2d",
                        "nn.dense",
                        "nn.batch_matmul",
                        "nn.relu",
                        "nn.leaky_relu",
                        "nn.prelu",
                        "clip",
                    ]
                    and isinstance(arg, relay.Call)
                )
                or (
                    isinstance(arg, relay.Call)
                    and (
                        arg_name
                        in [
                            "add",
                            "multiply",
                            "nn.sum_pool2d",
                            "nn.sum_pool3d",
                            "nn.max_pool2d",
                            "nn.max_pool3d",
                        ]
                    )
                )
            ):

                arg_dict["default_config"][arg_key]["dtype"] = "int8"

                call_id = self.node_id[arg]

                if len(self.exp_ref[id_node]) > 1:
                    for mem in self.exp_ref[id_node]:
                        if mem.split("-")[0] != tmp:
                            self.config[mem.split("-")[0]]["default_config"][mem.split("-")[1]][
                                "dtype"
                            ] = "int8"

                if (
                    arg_name in ["nn.relu", "nn.leaky_relu", "nn.prelu", "clip"]
                    and self.config[call_id]["default_config"]["input0"]["dtype"] == "int16"
                ):

                    self.config[call_id]["default_config"]["input0"]["dtype"] = "int8"

            self.config[tmp]["default_config"].update(arg_dict["default_config"])
            self.config[tmp]["valid_config"].update(arg_dict["valid_config"])

    def visit_tuple(self, tup):

        for arg in tup.fields:
            self.visit(arg)

        tmp = self.node_id[tup]
        self.idx = self.idx + 1
        LOGGER.debug("[config] tuple %d", self.idx)
        self.config[tmp] = {"valid_config": {}, "default_config": {}}

        arg_idx = -1
        for arg in tup.fields:
            id_node = self.node_id[arg]
            id_node_prefix = str(id_node.split("_")[0])

            arg_idx = arg_idx + 1
            arg_key = "input" + str(arg_idx)
            arg_dict = {"valid_config": {}, "default_config": {}}
            if id_node_prefix in self.quantize_config:
                arg_dict["default_config"] = {arg_key: self.quantize_config[id_node_prefix]}

            elif isinstance(arg, relay.Var) and "all" in self.quantize_config:
                arg_dict["default_config"] = {arg_key: self.quantize_config["all"]}

            elif isinstance(arg, relay.Var):
                arg_dict["valid_config"] = {
                    arg_key: {
                        "threshold": (
                            Threshold.MinMax,
                            Threshold.Percentile,
                            Threshold.MovingAverageMinMax,
                            Threshold.L2Norm,
                            Threshold.RelativeEntropy,
                        ),
                        "method": (Method.Symmetry, Method.Asymmetry),
                        "dtype": (DataType.UInt8, DataType.Int8),
                    }
                }

                arg_dict["default_config"] = {
                    arg_key: {
                        "threshold": Threshold.L2Norm,
                        "method": Method.Symmetry,
                        "dtype": DataType.Int8,
                    }
                }

            elif isinstance(arg, relay.Constant) and "constant" in self.quantize_config:
                arg_dict["default_config"] = {arg_key: self.quantize_config["constant"]}

            elif isinstance(arg, relay.Constant):
                if arg_idx != 2:
                    arg_dict["valid_config"] = {
                        arg_key: {
                            "threshold": (
                                Threshold.MinMax,
                                Threshold.Percentile,
                                Threshold.MovingAverageMinMax,
                                Threshold.L2Norm,
                                Threshold.RelativeEntropy,
                            ),
                            "method": (Method.Symmetry, Method.Asymmetry),
                            "dtype": (DataType.Int8,),
                        }
                    }

                    arg_dict["default_config"] = {
                        arg_key: {
                            "threshold": Threshold.MinMax,
                            "method": Method.Symmetry,
                            "dtype": DataType.Int8,
                        }
                    }
                else:
                    arg_dict["valid_config"] = {
                        arg_key: {
                            "threshold": (
                                Threshold.MinMax,
                                Threshold.Percentile,
                                Threshold.MovingAverageMinMax,
                                Threshold.L2Norm,
                                Threshold.RelativeEntropy,
                            ),
                            "method": (Method.Symmetry, Method.Asymmetry),
                            "dtype": (DataType.Int32,),
                        }
                    }
                    # must be bias!
                    arg_dict["default_config"] = {
                        arg_key: {
                            "threshold": Threshold.MinMax,
                            "method": Method.Symmetry,
                            "dtype": DataType.Int32,
                        }
                    }

            elif isinstance(arg, relay.Call) and "call" in self.quantize_config:

                arg_dict["default_config"] = {arg_key: self.quantize_config["call"]}

            elif isinstance(arg, relay.Call) and "all" in self.quantize_config:

                arg_dict["default_config"] = {arg_key: self.quantize_config["all"]}

            elif isinstance(arg, relay.Call):
                if isinstance(arg.op, relay.Function):
                    name = getattr(arg.op.attrs, "Composite")
                else:
                    name = arg.op.name

                tmp_dict = {}
                if name in FIXED_OP_TWOARGS_LIST:
                    tmp_dict = OPCONFIGS["FixedOpTwoArgs"].get_config(arg, self.config)
                elif name in IDENTITY_OP_LIST:
                    tmp_dict = OPCONFIGS["identity_op"].get_config(arg, self.config)
                elif name in OPCONFIGS:
                    tmp_dict = OPCONFIGS[name].get_config(arg, self.config)
                else:
                    tmp_dict = OPCONFIGS["float_op"].get_config(arg, self.config)

                arg_dict["default_config"] = {arg_key: tmp_dict["default_config"]}
                arg_dict["valid_config"] = {arg_key: tmp_dict["valid_config"]}

            else:
                arg_dict = {
                    "valid_config": {
                        arg_key: {
                            "threshold": (
                                Threshold.MinMax,
                                Threshold.Percentile,
                                Threshold.MovingAverageMinMax,
                                Threshold.L2Norm,
                                Threshold.RelativeEntropy,
                            ),
                            "method": (Method.Symmetry, Method.Asymmetry),
                            "dtype": (DataType.Int8, DataType.Int16),
                        }
                    },
                    "default_config": {
                        arg_key: {
                            "threshold": Threshold.L2Norm,
                            "method": Method.Symmetry,
                            "dtype": DataType.Int8,
                        }
                    },
                }

            self.config[tmp]["default_config"].update(arg_dict["default_config"])
            self.config[tmp]["valid_config"].update(arg_dict["valid_config"])


def config_space(cls):
    tmp = ConfigSpace(cls.pre_processed_mod, cls.node_id, cls.quantize_config)
    cls.config_space = tmp.config
    cls.all_op = tmp.all_op
