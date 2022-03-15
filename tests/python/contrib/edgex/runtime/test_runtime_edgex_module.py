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
from tvm.relay.quantization.relay_ops import round_right_shift


def verify_mod_export(func, params):
    input_shape = [int(x) for x in func.params[0].type_annotation.shape]
    input_dtype = func.params[0].type_annotation.dtype
    if input_dtype.startswith("i") or input_dtype.startswith("u"):
        data = np.random.randint(-128, 127, size=input_shape).astype(input_dtype)
    else:
        data = np.random.uniform(-128, 127, size=input_shape).astype(input_dtype)

    edgex_target = tvm.target.edgex()
    cpu_target = tvm.target.Target("llvm")
    targets = edgex_target
    ctxs = [tvm.edgex()]
    if UnsupportedDetector()(func):
        # if has unsupported op, offload them to cpu temporarily
        mod = tvm.IRModule.from_expr(OffloadUnsupported().visit_function(func))
        mod = relay.transform.InferType()(mod)
        func = mod["main"]
        targets = {"edgex": edgex_target, "cpu": cpu_target}
        ctxs = [tvm.edgex(), tvm.cpu()]

    # build an edgex model
    with TempOpStrategy(
        [x for x in tvm.ir.Op.list_op_names() if x != "nn.conv2d"],
        "edgex",
        fschedule=fschedule_general_vu,
    ):
        pass_ctx = build_config_nnp()
        plan_device_cfg = get_edgex_plan_device_config(pass_ctx)
        edgex_mod = IRModule.from_expr(func)
        edgex_mod = relay.transform.InferType()(edgex_mod)
        edgex_mod = relay.transform.PlanDevices(plan_device_cfg)(edgex_mod)
        if params is not None:
            func_with_params = bind_params_by_name(edgex_mod["main"], params)
            edgex_mod = tvm.ir.IRModule.from_expr(func_with_params)
        function = RelayToTIRAnnotator().visit(edgex_mod["main"])
        edgex_mod = IRModule.from_expr(function)
        with pass_ctx:
            lib = relay.build_module.build(edgex_mod, target=targets, params=params)

    # export model
    temp = utils.tempdir()
    path_lib = temp.relpath("edgex_deploy_lib.so")
    lib.export_library(path_lib)

    # load model
    loaded_lib = tvm.runtime.load_module(path_lib)
    os.remove(path_lib)

    # graph executor wrapper
    expect = get_graph_runtime_output(lib, data, ctxs)
    outs = get_graph_runtime_output(loaded_lib, data, ctxs)
    tvm.testing.assert_allclose(outs, expect, atol=1e-5)


def verify_op_mod_export():
    # prepare model and data
    mod, params = get_fs_fused_workload("resnet50", fix_norm=True)
    functions = get_fused_functions(mod, params)
    # todo(bxq): fix fused function name
    func, params = functions["fused_0_72"]
    verify_mod_export(func, params)


def verify_icache():
    iss_debug_mode = os.getenv("EDGEX_DEBUG_ISS", "off")
    os.environ["EDGEX_DEBUG_ISS"] = "off"
    input_shape = [1, 192, 28, 28]
    input_dtype = "int8"
    x = relay.var("input", shape=input_shape, dtype=input_dtype)
    mulnorm = relay.var("mulnorm", shape=[1, 1, 1, 1], dtype="int64")
    shiftnorm = relay.var("shiftnorm", shape=[1, 1, 1, 1], dtype="int64")
    y = relay.nn.max_pool2d(x, pool_size=(3, 3), padding=(1, 1))
    y = relay.cast(y, dtype="int64")
    y = relay.multiply(y, mulnorm)
    y = round_right_shift(y, shiftnorm)
    y = relay.clip(y, -128, 127)
    y = relay.cast(y, "int8")
    attrs = tvm.ir.make_node("DictAttrs", **{"Primitive": 1})
    fused_func = relay.Function([x, mulnorm, shiftnorm], y, attrs=attrs)

    def wrap_relay_fused_function(relay_function):
        new_args = [relay.Var(p.name_hint, p.type_annotation) for p in relay_function.params]
        return relay.Function(new_args, relay.Call(relay_function, new_args))

    function = wrap_relay_fused_function(fused_func)
    mod = tvm.IRModule.from_expr(function)
    mod = relay.transform.InferType()(mod)

    params = {}
    params["mulnorm"] = np.random.randint(0, 128, [1, 1, 1, 1]).astype("int64")
    params["shiftnorm"] = np.random.randint(0, 9, [1, 1, 1, 1]).astype("int64")

    verify_mod_export(mod["main"], params)
    os.environ["EDGEX_DEBUG_ISS"] = iss_debug_mode


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


def test_op_mod_export_using_asm():
    verify_op_mod_export()


def test_op_mod_export_using_obj():
    f_type = os.getenv("EDGEX_LLVM_OUTPUT_FILETYPE", "")
    os.environ["EDGEX_LLVM_OUTPUT_FILETYPE"] = "obj"
    verify_op_mod_export()
    os.environ["EDGEX_LLVM_OUTPUT_FILETYPE"] = f_type


@pytest.mark.edgex_slow
def test_mod_export():
    # prepare model and data
    mod, params = get_fs_fused_workload("resnet50", fix_norm=True)
    verify_mod_export(mod["main"], params)


@pytest.mark.skip
def test_icache_using_asm():
    verify_icache()


@pytest.mark.skip
def test_icache_using_obj():
    f_type = os.getenv("EDGEX_LLVM_OUTPUT_FILETYPE", "")
    os.environ["EDGEX_LLVM_OUTPUT_FILETYPE"] = "obj"
    verify_icache()
    os.environ["EDGEX_LLVM_OUTPUT_FILETYPE"] = f_type


if __name__ == "__main__":
    pytest.main([__file__])
