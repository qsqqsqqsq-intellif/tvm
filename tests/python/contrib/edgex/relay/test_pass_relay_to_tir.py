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
import sys
import pytest
import tvm
import numpy as np
from tvm import relay
from tvm._ffi.base import TVMError
from tvm._ffi.runtime_ctypes import DataType
from tvm.contrib.edgex.edgex import build_config_nnp
from tvm.contrib.edgex.testing import (
    check_edgex_relay_build,
    TempOpStrategy,
    get_edgex_plan_device_config,
)
from tvm.contrib.edgex.relay.transform import EdgeXRelayToTIR
from tvm.contrib.edgex.relay.op.strategy import fschedule_general_vu
from tvm.ir.type import PrimType


def do_test_primitive(inputs, output, params=None):
    func = relay.Function(inputs, output)
    func = func.with_attr("Primitive", 1)
    new_args = [relay.Var(p.name_hint, p.type_annotation) for p in inputs]
    func = relay.Function(new_args, relay.Call(func, new_args))
    mod = tvm.IRModule.from_expr(func)
    check_edgex_relay_build(mod, params, legacy_lower=True, test_fused=True)
    check_edgex_relay_build(mod, params, legacy_lower=False, test_fused=True)


def test_compile_add():
    a = relay.var("a", dtype="int32", shape=[128])
    b = relay.var("b", dtype="int32", shape=[128])
    c = a + b
    with TempOpStrategy("add", "edgex", fschedule=fschedule_general_vu):
        do_test_primitive([a, b], c)


def test_compile_conv():
    x = relay.var("x", dtype="int8", shape=[1, 3, 28, 28])
    w = relay.var("w", dtype="int8", shape=[16, 3, 3, 3])
    w_data = np.random.randint(-4, 4, [16, 3, 3, 3]).astype("int8")
    y = relay.nn.conv2d(x, w, kernel_size=[3, 3], out_dtype="int32")
    do_test_primitive([x, w], y, params={"w": w_data})


def get_fused_mod(func):
    pass_ctx = build_config_nnp()
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.FuseOps(0)(mod)
    mod = relay.transform.PlanDevices(get_edgex_plan_device_config(pass_ctx))(mod)
    return mod


def test_rewrite_with_constant():
    def renamer(name):
        return "my_namespace_" + name

    a = relay.var("a", dtype="int32", shape=[128])
    b = (a + relay.const(1, dtype="int32")) - relay.const(2, dtype="int32")
    func = relay.Function([a], b)
    mod = get_fused_mod(func)
    with TempOpStrategy(["add", "subtract"], "edgex", fschedule=lambda attr, f, tgt: f):
        mod = EdgeXRelayToTIR(renamer=renamer)(mod)
    assert isinstance(mod["my_namespace_edgex_fused_add"], tvm.tir.PrimFunc)
    assert isinstance(mod["my_namespace_edgex_fused_subtract"], tvm.tir.PrimFunc)


def test_rewrite_with_identity():
    a = relay.var("a", dtype="int32", shape=[128])
    params = [a]
    func = relay.Function([a], a)
    func = func.with_attr({"Primitive": 1})
    params = [relay.var("arg_" + p.name_hint, p.type_annotation) for p in params]
    func = relay.Function(params, relay.Call(func, params))
    mod = get_fused_mod(func)
    with pytest.raises(TVMError) as e:
        mod = EdgeXRelayToTIR()(mod)
    assert str(e).find("Aliased param buffer") >= 0


def test_relay_annotation_on_tir_block():
    x = relay.var("x", dtype="int8", shape=[1, 3, 28, 28])
    w = relay.var("w", dtype="int8", shape=[16, 3, 3, 3])
    w_data = np.random.randint(-4, 4, [16, 3, 3, 3]).astype("int8")
    y = relay.nn.conv2d(x, w, kernel_size=[3, 3], out_dtype="int32")
    func = relay.Function([x, w], y)
    mod = get_fused_mod(func)
    mod = EdgeXRelayToTIR(post_schedule_rewrite=False)(mod)
    primfunc = mod["edgex_fused_nn_conv2d"]
    s = tvm.tir.schedule.Schedule(primfunc)
    block_stmt = s.get_sref(s.get_child_blocks(s.get_block("root"))[1]).stmt
    assert block_stmt.annotations["relay_op_name"] == "nn.conv2d"
    assert block_stmt.annotations["relay_op_attrs.kernel_size"][0] == 3
    assert block_stmt.annotations["relay_op_attrs.kernel_size"][1] == 3
    assert block_stmt.annotations["relay_op_attrs.out_dtype"] == PrimType("int32")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
