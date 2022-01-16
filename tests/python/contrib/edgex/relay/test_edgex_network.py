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
import os
import tvm
import pytest
import json
from tvm import testing
from tvm import relay
from tvm.contrib.edgex.testing import get_graph_runtime_output
from tvm.contrib.debugger import debug_runtime
import numpy as np


@pytest.mark.skip("for debug use")
def test_get_layers_output():
    dtype = "float32"
    target = "llvm"
    ctx = tvm.context(target, 0)
    shape = (1, 3, 224, 224)

    mod, params = testing.resnet.get_workload()
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target, params=params)
    data = np.random.uniform(-1, 1, size=shape).astype("float32")

    # raw api
    try:
        gmod = lib["debug_create"]("default", ctx)
    except:
        print("Skip because debug graph_runtime not enabled")
        return
    set_input = gmod["set_input"]
    run = gmod["run"]
    get_output = gmod["get_output"]
    get_output_by_layer = gmod["get_output_by_layer"]
    set_input("data", tvm.nd.array(data))
    run()
    out_r = get_output(0).asnumpy()

    # debug graph runtime wrapper
    debug_m = debug_runtime.GraphModuleDebug(
        lib["debug_create"]("default", ctx), [ctx], lib.get_json(), "./build/layer_results"
    )
    debug_m.set_input("data", tvm.nd.array(data.astype(dtype)))
    debug_m.run()
    out_d = debug_m.get_output(0, tvm.nd.empty((1, 1000), dtype))

    # compare each layer
    for i, node in enumerate(debug_m.debug_datum.get_graph_nodes()):
        num_outputs = 1 if node["op"] == "param" else int(node["attrs"]["num_outputs"])
        for j in range(num_outputs):
            np.testing.assert_allclose(
                get_output_by_layer(i, j).asnumpy(),
                debug_m.debug_get_output(i + j).asnumpy(),
                rtol=1e-4,
                atol=1e-4,
            )


@pytest.mark.parametrize("net", ["resnet50", "mobilenet_v2"])
def test_networks(net):
    """test network from quantized mode and fusion stitching pass to tir expression"""
    # get quant mod and params
    mod_file = os.getenv("EDGEX_MODELS_DIR", "/tmp") + "/pytorch/%s/quantized/%s.json" % (
        net,
        net,
    )
    params_file = os.getenv("EDGEX_MODELS_DIR", "/tmp") + "/pytorch/%s/quantized/%s.params" % (
        net,
        net,
    )
    assert os.path.exists(mod_file) and os.path.exists(params_file)
    with open(mod_file, "r") as fi:
        mod = tvm.ir.load_json(json.load(fi))
    with open(params_file, "rb") as fi:
        params = relay.load_param_dict(fi.read())

    # build and run and compare
    print(mod["main"])
    input_shape = [int(x) for x in mod["main"].params[0].type_annotation.shape]
    input_dtype = mod["main"].params[0].type_annotation.dtype
    if input_dtype.startswith("i") or input_dtype.startswith("u"):
        data = np.random.randint(-128, 127, size=input_shape).astype(input_dtype)
    else:
        data = np.random.uniform(-128, 127, size=input_shape).astype(input_dtype)
    lib_std = relay.build(mod, target="llvm", params=params)
    out_std = get_graph_runtime_output(lib_std, data).numpy()
    with tvm.transform.PassContext(config={"relay.backend.use_meta_schedule": True}):
        lib_stg = relay.build(mod, target="llvm", params=params)
    out_stg = get_graph_runtime_output(lib_stg, data).numpy()
    np.testing.assert_allclose(out_stg, out_std, rtol=1e-4, atol=1e-4)


@pytest.mark.edgex_slow
@pytest.mark.parametrize("net", ["inception_v1", "inception_v4", "densenet"])
def test_more_networks(net):
    test_networks(net)


if __name__ == "__main__":
    pytest.main([__file__])
