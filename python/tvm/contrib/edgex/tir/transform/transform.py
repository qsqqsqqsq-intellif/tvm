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


def StorageConstraintHandler():
    """Handle the storage address or memory size
    according to the constraint.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.StorageConstraintHandler()


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
