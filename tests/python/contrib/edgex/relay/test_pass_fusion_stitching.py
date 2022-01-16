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
from tvm import relay
from tvm.contrib.edgex.relay.transform import FusionStitch


def get_fusion_stitch_mod(net, save=False):
    mod_file = os.getenv("EDGEX_MODELS_DIR") + "/pytorch/%s/quantized/ori/%s.json" % (net, net)
    assert os.path.exists(mod_file)
    with open(mod_file, "r") as fi:
        mod = tvm.ir.load_json(json.load(fi))
    mod = relay.transform.InferType()(mod)
    mod = FusionStitch()(mod)

    if save:
        mod_file = os.getenv("EDGEX_MODELS_DIR") + "/pytorch/%s/quantized/%s.json" % (net, net)
        with open(mod_file, "w") as fo:
            json.dump(tvm.ir.save_json(mod), fo)
    return mod


@pytest.mark.parametrize("net", ["resnet50", "inception_v1"])
def test_fusion_stitching(net):
    mod_file = os.getenv("EDGEX_MODELS_DIR", "/tmp") + "/pytorch/%s/quantized/%s.json" % (net, net)
    assert os.path.exists(mod_file)
    with open(mod_file, "r") as fi:
        expected_mod = tvm.ir.load_json(json.load(fi))

    mod = get_fusion_stitch_mod(net)
    assert tvm.ir.structural_equal(mod, expected_mod)


@pytest.mark.edgex_slow
@pytest.mark.parametrize("net", ["mobilenet_v2", "mobilenet_v2_qat", "inception_v4", "densenet"])
def test_more_fusion_stitching(net):
    test_fusion_stitching(net)


if __name__ == "__main__":
    pytest.main([__file__])
