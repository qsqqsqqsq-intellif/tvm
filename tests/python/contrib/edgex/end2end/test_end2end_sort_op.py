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
import numpy as np
import pytest
from tvm import relay
from tvm.contrib.edgex.testing import (
    check_edgex_relay_build,
    wrap_relay_fused_function,
)


@pytest.mark.parametrize(
    "N, K",
    [
        (1000, 72),  # within vm
    ],
)
@pytest.mark.parametrize("dtype", ["int32", "float16"])
@pytest.mark.parametrize("ret_type", ["both", "values", "indices"])
@pytest.mark.parametrize("is_ascend", [True, False])
def test_topk_1d(N, K, dtype, ret_type, is_ascend):
    x = relay.var("x", shape=[N], dtype=dtype)
    x_data = np.asarray(list(reversed((range(0, N))))).astype(dtype)
    np.random.shuffle(x_data)
    if ret_type == "both":
        y1, y2 = relay.topk(x, K, axis=0, ret_type=ret_type, is_ascend=is_ascend)
        f = wrap_relay_fused_function(relay.Function([x], relay.Tuple([y1, y2])), mark_edgex=True)
    else:
        y = relay.topk(x, K, axis=0, ret_type=ret_type, is_ascend=is_ascend)
        f = wrap_relay_fused_function(relay.Function([x], relay.Tuple([y])), mark_edgex=True)
    check_edgex_relay_build(f, input_data=[x_data])


@pytest.mark.parametrize("shape, K, axis", [([4, 64], 64, 1)])
def test_topk_nd(shape, K, axis):
    dtype = "int32"
    ret_type = "both"
    is_ascend = True
    x = relay.var("x", shape=shape, dtype=dtype)
    data = np.zeros(shape).astype(dtype).reshape([-1])
    data = np.linspace(0, data.shape[0], data.shape[0], dtype=dtype)
    np.random.shuffle(data)
    data = data.reshape(shape)
    if ret_type == "both":
        y1, y2 = relay.topk(x, K, axis=axis, ret_type=ret_type, is_ascend=is_ascend)
        f = wrap_relay_fused_function(relay.Function([x], relay.Tuple([y1, y2])), mark_edgex=True)
    else:
        y = relay.topk(x, K, axis=axis, ret_type=ret_type, is_ascend=is_ascend)
        f = wrap_relay_fused_function(relay.Function([x], relay.Tuple([y])), mark_edgex=True)
    t = check_edgex_relay_build(f, input_data=[data])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
