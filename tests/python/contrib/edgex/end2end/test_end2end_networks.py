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
"""Unit tests for resnet50"""
import pytest
import tvm
from tvm import relay
from tvm.contrib.edgex.testing import (
    UnsupportedDetector,
    OffloadUnsupported,
    check_edgex_relay_build,
    get_fs_fused_workload,
    get_fused_functions,
    TempOpStrategy,
)
from tvm.contrib.edgex.relay.op.strategy import (
    fschedule_general_vu,
    SPECIFIED_FSCHEDULE_OPS,
)


def verify_network_per_op(net):
    """test network's all ops, run each fused graph independently"""
    mod, params = get_fs_fused_workload(net, fix_norm=True)
    with TempOpStrategy(
        [x for x in tvm.ir.Op.list_op_names() if x not in SPECIFIED_FSCHEDULE_OPS],
        "edgex",
        fschedule=fschedule_general_vu,
    ):
        functions = get_fused_functions(mod, params)
        for name in functions:
            func, sub_params = functions[name]
            if UnsupportedDetector()(func):
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


def verify_network_end2end(net):
    """test network end2end"""
    mod, params = get_fs_fused_workload(net, fix_norm=True)
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


@pytest.mark.edgex_slow
@pytest.mark.parametrize("net", ["resnet50"])
def test_network_per_op(net):
    verify_network_per_op(net)


@pytest.mark.edgex_slow
@pytest.mark.parametrize("net", ["resnet50"])
def test_network_end2end(net):
    verify_network_end2end(net)


if __name__ == "__main__":
    verify_network_per_op("resnet50")
    verify_network_end2end("resnet50")
