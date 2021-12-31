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
"""Relay tensor operations for quantitization."""
# pylint: disable=redefined-builtin, unused-argument
from tvm.relay.op import _make
from tvm.target import override_native_generic_func
from tvm.relay.op import op as reg
from tvm.relay import op as _op
import tvm.relay.op.strategy.generic as generic
from tvm import topi


def round_right_shift(lhs, rhs):
    """Round right shift with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.round_right_shift(lhs, rhs)


@override_native_generic_func("round_right_shift_strategy")
def round_right_shift_strategy(attrs, inputs, out_type, target):
    """round_right_shift general strategy"""
    strategy = _op.OpStrategy()

    def fcompute(attrs, inputs, out_dtype):
        return [topi.cpp.round_right_shift(inputs[0], inputs[1])]

    strategy.add_implementation(
        fcompute,
        generic.schedule_injective,
        name="round_right_shift.%s" % target.kind.name,
        plevel=15,
    )
    return strategy


reg.register_strategy("round_right_shift", round_right_shift_strategy, level=15)
