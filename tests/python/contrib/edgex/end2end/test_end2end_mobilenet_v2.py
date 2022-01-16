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
"""Unit tests for mobilenet_v2"""
import json
import os
import pytest
import tvm
from tvm import relay
from tvm.contrib.edgex.relay.transform import ConvertDepthwiseConv2DToConv2D
from tvm.contrib.edgex.testing import (
    check_edgex_relay_build,
    get_fused_functions,
    TempOpStrategy,
)
from tvm.contrib.edgex.relay.op.strategy import fschedule_general_vu
import numpy as np


class UnsupportedDetector(relay.ExprVisitor):
    """Detect fused op currently not supported"""

    def __init__(self):
        super().__init__()
        self.has_conv = False

    def __call__(self, func):
        self.match = False
        self.visit_function(func)
        return self.match

    def visit_call(self, call):
        if isinstance(call.op, tvm.ir.Op):
            if (
                call.op.name == "cast"
                and isinstance(call.args[0], relay.Var)
                and call.args[0].type_annotation.dtype == "int32"
            ):
                self.match = True
        if isinstance(call.attrs, relay.op.op_attrs.Conv2DAttrs):
            self.match = True
        super().visit_call(call)


class OffloadUnsupported(relay.ExprMutator):
    """Offload unsupported ops to cpu"""

    def __init__(self):
        super().__init__()
        self.offload_cpu = False
        self.detector = UnsupportedDetector()

    def visit_call(self, call):
        if isinstance(call.op, relay.Function):
            args = [self.visit(x) for x in call.args]
            if self.detector(call.op):
                new_f = relay.Function(call.op.params, call.op.body, call.op.ret_type)
                new_f = new_f.with_attr("Primitive", 1)
                new_f = new_f.with_attr("DeviceType", 1)
                return relay.annotation.on_device(relay.Call(new_f, args), "cpu")
            return relay.annotation.on_device(relay.Call(call.op, args), "edgex")
        return super().visit_call(call)


def get_mobilenet_v2():
    """helper to get mobilenet_v2 network"""
    mod_file = (
        os.getenv("EDGEX_MODELS_DIR", "/tmp") + "/pytorch/mobilenet_v2/quantized/mobilenet_v2.json"
    )
    params_file = (
        os.getenv("EDGEX_MODELS_DIR", "/tmp")
        + "/pytorch/mobilenet_v2/quantized/mobilenet_v2.params"
    )
    assert os.path.exists(mod_file) and os.path.exists(params_file)
    with open(mod_file, "r") as fi:
        mod = tvm.ir.load_json(json.load(fi))
    with open(params_file, "rb") as fi:
        params = relay.load_param_dict(fi.read())

    # fix norm value range temporarily
    params = dict(params.items())
    for k in params:
        if k.find("round_right_shift") >= 0:
            norm = params[k].asnumpy()
            norm = np.minimum(norm, 24)
            norm = np.maximum(norm, 1)
            params[k] = tvm.nd.array(norm)
        elif k.find("multiply") >= 0:
            norm = params[k].asnumpy()
            norm = np.minimum(norm, 127)
            params[k] = tvm.nd.array(norm)
        elif k.find("bias_add") >= 0:
            norm = params[k].asnumpy()
            norm = np.maximum(np.minimum(norm, 2 ** 20), -(2 ** 20))
            params[k] = tvm.nd.array(norm)

    mod, params = ConvertDepthwiseConv2DToConv2D(params).run(mod)
    mod = relay.transform.InferType()(mod)
    return mod, params


@pytest.mark.skip("skip vu vacc i64 not support.")
def test_mobilenet_v2_per_op():
    """test mobilenet_v2 all ops, run each fused graph independently"""
    mod, params = get_mobilenet_v2()
    detector = UnsupportedDetector()
    with TempOpStrategy(
        [x for x in tvm.ir.Op.list_op_names() if x != "nn.conv2d"],
        "edgex",
        fschedule=fschedule_general_vu,
    ):
        functions = get_fused_functions(mod, params)
        for name in functions:
            func, sub_params = functions[name]
            if detector(func):
                continue
            print(func.astext(False))
            print("op name: {}".format(name))
            check_edgex_relay_build(
                func,
                params=sub_params,
                check_cpu=True,
                test_fused=True,
                cpu_use_tir=False,
            )


@pytest.mark.skip("skip vu vacc i64 not support.")
def test_mobilenet_v2_end2end():
    """test mobilenet_v2 end2end"""
    mod, params = get_mobilenet_v2()
    if UnsupportedDetector()(mod["main"]):
        # if has unsupported op, offload them to cpu temporarily
        mod = tvm.IRModule.from_expr(OffloadUnsupported().visit_function(mod["main"]))
        mod = relay.transform.InferType()(mod)
    with TempOpStrategy(
        [x for x in tvm.ir.Op.list_op_names() if x != "nn.conv2d"],
        "edgex",
        fschedule=fschedule_general_vu,
    ):
        check_edgex_relay_build(
            mod,
            params=params,
            check_cpu=True,
            test_fused=True,
            cpu_use_tir=False,
        )


if __name__ == "__main__":
    test_mobilenet_v2_per_op()
    test_mobilenet_v2_end2end()
