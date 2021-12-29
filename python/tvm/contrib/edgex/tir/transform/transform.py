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
"""Wrapping existing edgex transformations."""
# pylint: disable=invalid-name
import os
import tvm
from . import _ffi_api


def InjectCalculatedIsa():
    """Inject calculated isa value using existed isa's value.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectCalculatedIsa()


def InjectHandShakeIntrin():
    """Inject hand shake intrinsic for nnp.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectHandShakeIntrin()


def FlatStorageConstraintHandler():
    """Handle the flat storage address or memory size
    according to the constraint.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.FlatStorageConstraintHandler()


def StorageRewriteNNP400():
    """Rewrite storage allocation pattern.

    Moves the allocation to outer most possible scope.
    Trying to share space between allocations to make
    a static allocation plan when possible.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.StorageRewriteNNP400()


def SplitVcuControlFlow():
    """Split vcu/cu control flow into different branches.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.SplitVcuControlFlow()


def InjectDmaIntrin():
    """Rewrite specific buffer access patterns to dma intrinsic.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectDmaIntrin()


def LiftGlobalAllocation():
    """Move all global scope memory allocation out of device_scope annotation.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LiftGlobalAllocation()


def RewriteVcuOps():
    """Optimize vectorized computation in vcu units.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.RewriteVcuOps()


def InlinePrimFuncCalls(extern_primfuncs=None):
    """Inline calls to primfuncs.

    Parameters
    ----------
    extern_primfuncs : dict
        Mapping from extern name to PrimFunc

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InlinePrimFuncCalls(extern_primfuncs)


def DumpOrReuseLoweredTIR(working_dir, reuse_existing_tir=False):
    """Dump current tir or reuse tir script from previous dump.
    TODO(bxq): Since currently low level tir scripting is not fully
    supported, we use this pass at final lower phase for debug purpose.

    Parameters
    ----------
    reuse_existing_tir : bool
        Try load tir script from previous dump.
    """

    def _func(mod: tvm.IRModule, _):
        for gv in mod.functions:
            func = mod.functions[gv]
            kernel_name = gv.name_hint + "_kernel0"
            tir_path = os.path.join(working_dir, kernel_name, "tir", kernel_name + ".tir.py")
            if reuse_existing_tir:
                if not os.path.isfile(tir_path):
                    raise ValueError(f"Can not find existing tir script file {tir_path}")
                with open(tir_path) as input_file:
                    obj = tvm.script.from_source(input_file.read(), tir_prefix="T")
                if not isinstance(obj, tvm.tir.PrimFunc):
                    raise ValueError(f"The dumped object is not PrimFunc in {tir_path}")
                mod.update_func(gv, obj)
            else:
                tir_dir = os.path.dirname(tir_path)
                if not os.path.isdir(tir_dir):
                    os.makedirs(tir_dir)
                with open(tir_path, "w") as output_file:
                    output_file.write(func.script())
        return mod

    return tvm.transform.module_pass(_func, opt_level=2)
