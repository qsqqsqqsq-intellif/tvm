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
import tvm
from tvm import relay
from tvm.contrib.edgex.arith.nlfc import extract_nlfc_params
import tvm.testing
from tvm import te
from tvm import topi
from tvm.contrib.edgex.topi import naive_vu_schedule
from tvm.contrib.edgex.tir.transform import ConvertFpToNlfc
from tvm.contrib.edgex.testing import (
    check_edgex_relay_build,
    check_edgex_tir_build,
    wrap_relay_fused_function,
    TempOpStrategy,
)
from tvm.tir.function import PrimFunc


def check_nlfc(name, primfunc, data_range):
    """schedule check helper for nlfc testcase"""
    # convert arithmetic op to it's nlfc counterpart op
    edgex_func = ConvertFpToNlfc()(tvm.IRModule.from_expr(primfunc))["main"]
    nlfc_buffers = None
    nlfc_params, nlfc_tables, edgex_func = extract_nlfc_params(edgex_func)

    # make nlfc tables as fixed input
    input_data = []
    if nlfc_tables:
        input_data.extend([_.numpy() for _ in nlfc_tables])
    for _ in primfunc.params:
        input_data.append(None)

    # make cpu primfunc, add dummy nlfc args
    cpu_params = list(primfunc.params)
    cpu_bufs = dict(primfunc.buffer_map)
    if nlfc_params:
        nlfc_buffers = set()
        cpu_params = list(nlfc_params) + cpu_params
        for p in nlfc_params:
            nlfc_buf = edgex_func.buffer_map[p]
            cpu_bufs[p] = nlfc_buf
            nlfc_buffers.add(nlfc_buf)
    cpu_func = PrimFunc(
        cpu_params,
        primfunc.body,
        ret_type=primfunc.ret_type,
        buffer_map=cpu_bufs,
        attrs=primfunc.attrs,
    )

    # checking
    check_edgex_tir_build(
        name,
        edgex_func,
        edgex_fschedule=lambda attrs, f, tgt: naive_vu_schedule(
            f, enable_relay_rewrite=True, nlfc_buffers=nlfc_buffers
        ),
        cpu_prim_func=cpu_func,
        check_cpu=True,
        input_data=input_data,
        data_range=data_range,
        atol=1e-3,
        rtol=1e-3,
    )


def test_sigmoid_tir():
    def do_test(shape, dtype):
        x = te.placeholder(shape, dtype, "x")
        y = topi.sigmoid(x)
        f = te.create_prim_func([x, y])
        check_nlfc(
            f"nlfc_sigmoid_{dtype}_{'_'.join([str(_) for _ in shape])}", f, data_range=[-16, 16]
        )

    do_test([16], "float16")
    do_test([31], "float16")
    do_test([128], "float16")


def test_relay_nlfc_rewrite():
    x = relay.var("x", shape=[16], dtype="float16")
    y = relay.sigmoid(x)
    f = relay.Function([x], y)
    f = wrap_relay_fused_function(f)
    mod = relay.transform.InferType()(tvm.IRModule.from_expr(f))

    def fschedule(attrs, primfunc, target):
        if target.kind.name != "edgex":
            return primfunc
        return naive_vu_schedule(primfunc, enable_relay_rewrite=True)

    with TempOpStrategy("sigmoid", "edgex", fschedule=fschedule):
        check_edgex_relay_build(
            mod,
            test_fused=True,
            data_range=[-16, 16],
            atol=1e-3,
            rtol=1e-3,
        )


if __name__ == "__main__":
    test_sigmoid_tir()
    test_relay_nlfc_rewrite()
