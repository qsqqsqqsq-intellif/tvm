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
import tvm.testing
from tvm.contrib.edgex.testing import TempOpStrategy, check_edgex_relay_build


def test_heterogeneous_execution():
    a = relay.var("a", dtype="int32", shape=[128])
    b = relay.var("b", dtype="int32", shape=[128])
    c = relay.annotation.on_device(a + b, "edgex")
    d = relay.annotation.on_device(c * relay.const(2), "cpu")
    mod = tvm.IRModule.from_expr(relay.Function([a, b], d))
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.FuseOps(fuse_opt_level=0)(mod)
    with TempOpStrategy(["multiply", "add"], ["edgex", "llvm"]):
        check_edgex_relay_build(mod, None, check_cpu=True, test_fused=True)


if __name__ == "__main__":
    test_heterogeneous_execution()
