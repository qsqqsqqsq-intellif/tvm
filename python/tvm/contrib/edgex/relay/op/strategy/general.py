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
# pylint: disable=unused-argument
"""Edgex relay op strategies."""

import tvm
import tvm.relay.op.strategy.generic as generic
from tvm.relay import op as _op
from tvm.relay.op import op as reg
from tvm import te
from tvm import topi
from tvm.target import get_native_generic_func, generic_func
import tvm.relay.quantization
from tvm.relay.quantization.relay_ops import round_right_shift_strategy

SPECIFIED_FSCHEDULE_OPS = ["nn.conv2d", "concatenate", "split", "reshape", "transpose"]


def register_edgex_fschedule(op_name):
    """Workaround annotation for fschedule register"""

    def _wrapped_fschedule(fschedule):
        if (
            _op.get(op_name).has_attr("FEdgeXSchedule")
            and _op.get(op_name).get_attr("FEdgeXSchedule") is not None
        ):
            generic_fschedule = _op.get(op_name).get_attr("FEdgeXSchedule")
        else:
            generic_fschedule = get_native_generic_func(op_name + "_edgex_fschedule")
            tvm.ir.register_op_attr(op_name, "FEdgeXSchedule", generic_fschedule)
        generic_fschedule.register(fschedule, "edgex", allow_override=True)
        return generic_fschedule

    return _wrapped_fschedule


def register_edgex_fprimfunc(op_name):
    """Workaround annotation for primfunc register"""

    def _wrapped_fprimfunc(fprimfunc):
        if (
            _op.get(op_name).has_attr("FEdgeXPrimFunc")
            and _op.get(op_name).get_attr("FEdgeXPrimFunc") is not None
        ):
            generic_fprimfunc = _op.get(op_name).get_attr("FEdgeXPrimFunc")
        else:
            generic_fprimfunc = get_native_generic_func(op_name + "_edgex_fprimfunc")
            tvm.ir.register_op_attr(op_name, "FEdgeXPrimFunc", generic_fprimfunc)
        generic_fprimfunc.register(fprimfunc, "edgex", allow_override=True)
        return generic_fprimfunc

    return _wrapped_fprimfunc


def fschedule_general_vu(attrs, prim_func, tgt):
    """general fschedule function for non-conv ops"""
    scheduled_func = tvm.contrib.edgex.topi.naive_vu_schedule(
        prim_func, is_cpu=tgt.kind == "llvm", allow_multi_block=True, enable_relay_rewrite=True
    )
    return scheduled_func


@generic_func
def dummy_schedule(attrs, outs, tgt):
    """Default schedule for edgex."""
    return te.create_schedule([x.op for x in outs])


@generic.schedule_injective.register("edgex")
def schedule_injective_edgex(attrs, outs, target):
    """schedule injective ops for edgex"""
    with target:
        outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    te.schedule.AutoInlineInjective(s)
    return s


@generic.dense_strategy.register("edgex")
def dense_strategy_edgex(attrs, inputs, out_type, target):
    """temporary dense edgex strategy that are schedulable"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        generic.wrap_compute_dense(topi.nn.dense),
        generic.wrap_topi_schedule(topi.generic.schedule_dense),
        name="dense.vu.edgex",
        plevel=10,
    )
    return strategy


@round_right_shift_strategy.register("edgex")
def round_right_shift_strategy_edgex(attrs, inputs, out_type, target):
    """round_right_shift edgex strategy"""
    strategy = _op.OpStrategy()

    def fcompute(attrs, inputs, out_dtype):
        return [topi.cpp.round_right_shift_intrin(inputs[0], inputs[1])]

    strategy.add_implementation(
        fcompute,
        generic.schedule_injective,
        name="round_right_shift.%s" % target.kind.name,
        plevel=15,
    )
    # strategy.add_tir_implementation(
    #    fcompute, tir_schedule_injective, name="round_right_shift.%s" % target.kind.name, plevel=15
    # )
    return strategy


reg.register_schedule("cast_reinterpret", dummy_schedule)


@register_edgex_fschedule("concatenate")
def schedule_concatenate_edgex(attrs, prim_func, target):
    """concatenate edgex schedule"""
    return tvm.contrib.edgex.topi.schedule_memcpy_style_edgex_impl(prim_func, target)


@register_edgex_fschedule("split")
def schedule_split_edgex(attrs, prim_func, target):
    """split edgex schedule"""
    return tvm.contrib.edgex.topi.schedule_memcpy_style_edgex_impl(prim_func, target)


@register_edgex_fschedule("reshape")
def schedule_reshape_edgex(attrs, prim_func, target):
    """reshape edgex schedule"""
    return tvm.contrib.edgex.topi.schedule_memcpy_style_edgex_impl(prim_func, target)


@register_edgex_fschedule("transpose")
def schedule_transpose_edgex(attrs, prim_func, target):
    """transpose edgex schedule"""
    return tvm.contrib.edgex.topi.schedule_memcpy_style_edgex_impl(prim_func, target)


@register_edgex_fprimfunc("topk")
def fprimfunc_topk_edgex(attrs, inputs, out_type):
    """topk edgex fprimfunc"""
    return tvm.contrib.edgex.topi.wrap_fprimfunc_topk(
        tvm.contrib.edgex.topi.sort_fprimfunc_naive_impl
    )(attrs, inputs, out_type)


@register_edgex_fschedule("topk")
def schedule_topk_edgex(attrs, prim_func, target):
    """topk edgex schedule"""
    return prim_func