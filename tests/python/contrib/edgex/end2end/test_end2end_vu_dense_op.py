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
import pytest
import tvm
import tvm.testing
from tvm import relay
from tvm.contrib.edgex.topi import naive_vu_schedule
from tvm.contrib.edgex.testing import (
    TempOpStrategy,
    check_edgex_relay_build,
    wrap_relay_fused_function,
)
from tvm.contrib.edgex.topi import naive_vu_schedule


def fschedule(attrs, primfunc, target):
    if target.kind.name != "edgex":
        return primfunc
    return naive_vu_schedule(primfunc, allow_multi_block=True)


def do_test_vu_dense_op(M, N, K, dtype, out_dtype, has_bias=False):
    x = relay.var("x", dtype=dtype, shape=[M, K])
    w = relay.var("w", dtype=dtype, shape=[N, K])
    y = relay.nn.dense(x, w, units=N, out_dtype=out_dtype)
    if has_bias:
        b = relay.var("b", dtype=out_dtype, shape=[N])
        y = relay.nn.bias_add(y, b)
        relay_func = relay.Function([x, w, b], y)
    else:
        relay_func = relay.Function([x, w], y)
    relay_func = wrap_relay_fused_function(relay_func)
    mod = relay.transform.InferType()(tvm.IRModule.from_expr(relay_func))

    with TempOpStrategy(
        "nn.dense",
        "edgex",
        fschedule=fschedule,
    ):
        check_edgex_relay_build(mod)


@pytest.mark.skip
def test_vu_dense_op_in_common_backbone():
    # mobilenet_v3
    do_test_vu_dense_op(1, 1024, 576, dtype="int8", out_dtype="int32", has_bias=False)
    do_test_vu_dense_op(1, 1024, 576, dtype="int8", out_dtype="int32", has_bias=True)
    # resnet50
    do_test_vu_dense_op(1, 2048, 1000, dtype="int8", out_dtype="int32", has_bias=False)
    do_test_vu_dense_op(1, 2048, 1000, dtype="int8", out_dtype="int32", has_bias=True)


if __name__ == "__main__":
    test_vu_dense_op_in_common_backbone()
