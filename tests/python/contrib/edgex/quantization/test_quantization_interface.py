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
import tvm
from tvm import relay
import tvm.relay.quantization


def test_quantization_interface():
    data = relay.var("data", shape=[1, 3, 16, 16], dtype="float32")
    weight = relay.var("weight", shape=[16, 3, 3, 3], dtype="float32")
    conv = relay.nn.conv2d(data, weight, padding=[1, 1], kernel_size=[3, 3])
    weight_data = np.random.uniform(-10, 10, [16, 3, 3, 3]).astype("float32")
    mod = tvm.IRModule.from_expr(relay.Function([data, weight], conv))
    params = {"weight": weight_data}
    quant_mod, quant_params = relay.quantization.run_quantization(
        "testconv", mod, params=params, fast_mode=True
    )
    for param in quant_mod["main"].params:
        name = param.name_hint
        if name == "data":
            continue
        data = quant_params[name]
        assert data.dtype == param.checked_type.dtype
        assert len(data.shape) == len(param.checked_type.shape)
        for x, y in zip(data.shape, param.checked_type.shape):
            assert x == int(y)


if __name__ == "__main__":
    test_quantization_interface()
