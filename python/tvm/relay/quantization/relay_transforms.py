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
# pylint: disable=invalid-name, unused-argument, missing-docstring, unused-import
"""
Relay pass transformation infrastructure.
"""
import types
import inspect
import functools
import warnings

import tvm.ir
from tvm import te
from tvm.runtime import ndarray as _nd

from tvm import relay
from tvm.relay.transform import _ffi_api


def InsertNorm(mean, scale):
    """Insert batchnorm in front of graph.

    Returns
    -------
    ret: tvm.transform.Pass
        The registered pass to insert norm.

    """
    return _ffi_api.InsertNorm(mean, scale)


def FuseAdd():
    """Fuse add op into biasadd or convert add which behind conv2d or dense to bias.

    Returns
    -------
    ret: tvm.transform.Pass
        The registered pass to perform operator simplification.

    """
    return _ffi_api.FuseAdd()


def FuseReshapeSqueeze():
    """Fuse multiply to conv/dense.

    Parameters
    ----------
    e: tvm.relay.Expr
        The input Expression

    Returns
    -------
    result: tvm.relay.Expr
    """
    return _ffi_api.FuseReshapeSqueeze()


def ConvertAvgpoolToSumpool():
    """Convert avg_pool2d/global_avg_pool2d op to sum_pool2d/global_avg_sum_pool2d + multiply.

    Returns
    -------
    ret: tvm.transform.Pass
        The registered pass to convert avg_pool2d/global_avg_pool2d op.

    """
    return _ffi_api.ConvertAvgpoolToSumpool()


def ConvertAdaptivepoolToNormpool():
    """convert_adaptivepool_to_normpool_ops special operators to basic operators.

    Parameters
    ----------
    e: tvm.relay.Expr
        The input Expression

    Returns
    -------
    result: tvm.relay.Expr
    """
    return _ffi_api.ConvertAdaptivepoolToNormpool()


def FuseMultiplyToConv():
    """Fuse multiply to conv/dense.

    Parameters
    ----------
    e: tvm.relay.Expr
        The input Expression

    Returns
    -------
    result: tvm.relay.Expr
    """
    return _ffi_api.FuseMultiplyToConv()
