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
"""Edgex extension of topi operators"""
from __future__ import absolute_import as _abs
import tvm.topi.cpp as _cpp


def round_right_shift(lhs, rhs):
    """Round right shift with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.te.Tensor or Expr
        The left operand
    rhs : tvm.te.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.te.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.round_right_shift(lhs, rhs)


def cast_reinterpret(data, dtype):
    """Reinterpret_cast input tensor to data type, same as numpy.ndarray.view.
    Currently support downcast(eg.: int32 to int8) only

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    dtype: str
        The target data type

    Returns
    -------
    result : relay.Expr
        The reinterpreted result.
    """
    return _cpp.cast_reinterpret(data, dtype)
