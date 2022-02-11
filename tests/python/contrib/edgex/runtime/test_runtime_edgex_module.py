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

import numpy as np
import pytest
import tvm
from tvm import relay, te
from tvm.ir.module import IRModule
from tvm.contrib import utils
from tvm.contrib.edgex.edgex import build_config_nnp
from tvm.contrib.edgex.relay.op.strategy import fschedule_general_vu
from tvm.contrib.edgex.testing import *
from tvm.relay.build_module import bind_params_by_name


def verify_mod_export(single_op=False):
    # prepare model and data
    mod, params = get_fs_fused_workload("resnet50", fix_norm=True)
    func = mod["main"]
    if single_op:
        functions = get_fused_functions(mod, params)
        # todo(bxq): fix fused function name
        func, params = functions["fused_0_72"]
    input_shape = [int(x) for x in func.params[0].type_annotation.shape]
    input_dtype = func.params[0].type_annotation.dtype
    if input_dtype.startswith("i") or input_dtype.startswith("u"):
        data = np.random.randint(-128, 127, size=input_shape).astype(input_dtype)
    else:
        data = np.random.uniform(-128, 127, size=input_shape).astype(input_dtype)

    if UnsupportedDetector()(func):
        # if has unsupported op, offload them to cpu temporarily
        mod = tvm.IRModule.from_expr(OffloadUnsupported().visit_function(func))
        mod = relay.transform.InferType()(mod)
        func = mod["main"]

    # build an edgex model
    with TempOpStrategy(
        [x for x in tvm.ir.Op.list_op_names() if x != "nn.conv2d"],
        "edgex",
        fschedule=fschedule_general_vu,
    ):
        pass_ctx = build_config_nnp()
        edgex_target = tvm.target.edgex()
        cpu_target = tvm.target.Target("llvm")
        plan_device_cfg = get_edgex_plan_device_config(pass_ctx)
        edgex_mod = IRModule.from_expr(func)
        edgex_mod = relay.transform.InferType()(edgex_mod)
        edgex_mod = relay.transform.PlanDevices(plan_device_cfg)(edgex_mod)
        if params is not None:
            func_with_params = bind_params_by_name(edgex_mod["main"], params)
            edgex_mod = tvm.ir.IRModule.from_expr(func_with_params)
        function = RelayToTIRAnnotator().visit(edgex_mod["main"])
        edgex_mod = IRModule.from_expr(function)
        print(edgex_mod["main"])
        with pass_ctx:
            targets = edgex_target if single_op else {"edgex": edgex_target, "cpu": cpu_target}
            lib = relay.build_module.build(edgex_mod, target=targets, params=params)

    # export model
    temp = utils.tempdir()
    path_lib = temp.relpath("edgex_deploy_lib.so")
    lib.export_library(path_lib)

    # load model
    loaded_lib = tvm.runtime.load_module(path_lib)
    os.remove(path_lib)

    # graph executor wrapper
    ctxs = [tvm.edgex()] if single_op else [tvm.edgex(), tvm.cpu()]
    expect = get_graph_runtime_output(lib, data, ctxs)
    outs = get_graph_runtime_output(loaded_lib, data, ctxs)
    tvm.testing.assert_allclose(outs, expect, atol=1e-5)


def test_edgex_add():
    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
    s = te.create_schedule(C.op)
    fadd = tvm.build(s, [A, B, C], "edgex", name="edgex_add")

    ctx = tvm.edgex(0)
    shape = (1024,)
    a = tvm.nd.array(np.random.uniform(size=shape).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=shape).astype(B.dtype), ctx)
    expect = np.add(a.numpy(), b.numpy())
    c = tvm.nd.empty(shape, C.dtype, ctx)
    fadd(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), expect, rtol=1e-4)

    # export and load
    temp = utils.tempdir()
    path_lib = temp.relpath("edgex_add_lib.so")
    fadd.export_library(path_lib)
    loaded_m = tvm.runtime.load_module(path_lib)
    os.remove(path_lib)
    d = tvm.nd.empty(shape, C.dtype, ctx)
    loaded_m(a, b, d)
    tvm.testing.assert_allclose(d.numpy(), expect, rtol=1e-4)


def test_op_mod_export():
    verify_mod_export(single_op=True)


@pytest.mark.edgex_slow
def test_mod_export():
    verify_mod_export()


if __name__ == "__main__":
    test_edgex_add()
    test_op_mod_export()
    test_mod_export()
