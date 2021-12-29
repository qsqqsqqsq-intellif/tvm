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
"""TVM EdgeX"""
# pylint: disable-msg=C0103,W0401,W0614
import os
import tvm
from tvm.contrib.edgex.tir.transform import *


def pass_debug_wrapper(passfunc):
    """Print tir before and after pass"""

    def _func(mod, ctx):
        print(ctx)
        print("Before pass: %s" % mod.script())
        result = passfunc(mod)
        print("After pass: %s" % result.script())
        return result

    return tvm.transform.module_pass(_func, opt_level=2)


def build_config_nnp(extra_config=None, extra_disabled_pass=None, opt_level=2):
    """Add nnp lower pass.

    Returns
    -------
    ret : tvm.transform.PassContext
        The pass context contains the nnp lower pass.
    """

    pass_list = []
    pass_list.append((2, InjectDmaIntrin()))
    pass_list.append((2, InjectCalculatedIsa()))
    pass_list.append((2, InjectHandShakeIntrin()))
    pass_list.append((2, RewriteVcuOps()))
    pass_list.append((2, FlatStorageConstraintHandler()))
    pass_list.append((2, StorageRewriteNNP400()))
    pass_list.append((2, SplitVcuControlFlow()))
    pass_list.append((3, tvm.tir.transform.DecorateDeviceScope()))
    pass_list.append((3, LiftGlobalAllocation()))

    # lowered tir dumping support
    dump_tir = os.environ.get("EDGEX_DEBUG_DUMP_TIR", None) is not None
    use_existing_tir = os.environ.get("EDGEX_DEBUG_USE_EXISTING_TIR", None) is not None
    working_dir = os.environ.get("EDGEX_DEBUG_WORKING_DIR", None)
    if dump_tir or use_existing_tir:
        if working_dir is None:
            raise ValueError("Please specify EDGEX_DEBUG_WORKING_DIR for tir debug purpose")
        pass_list.append((3, DumpOrReuseLoweredTIR(working_dir, use_existing_tir)))

    config = {
        "tir.add_lower_pass": pass_list,
        "relay.backend.use_meta_schedule": True,
        "relay.fallback_device_type": 16,
        "relay.backend.use_multitarget_pass_context": True,
    }
    if extra_config is not None:
        config.update(extra_config)

    disabled_pass = [
        # we will use tvm_access_ptr() in codegen and do not lower it to address_of()
        "tir.LowerDeviceStorageAccessInfo",
        "tir.StorageRewrite",
    ]
    if extra_disabled_pass is not None:
        if not isinstance(extra_disabled_pass, list):
            extra_disabled_pass = [extra_disabled_pass]
        for pass_obj in extra_disabled_pass:
            if isinstance(pass_obj, tvm.transform.Pass):
                pass_name = pass_obj.info.name
            elif isinstance(pass_obj, str):
                pass_name = pass_obj
            else:
                raise ValueError("pass in `extra_disabled_pass` should be string or Pass")
            disabled_pass.append(pass_name)
    return tvm.transform.PassContext(
        config=config, disabled_pass=disabled_pass, opt_level=opt_level
    )
