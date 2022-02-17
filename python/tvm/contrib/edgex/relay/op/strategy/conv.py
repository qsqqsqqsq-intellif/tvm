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

import re
from tvm.relay.op.strategy.generic import (
    wrap_compute_conv2d,
    wrap_topi_schedule,
    conv2d_strategy,
    conv2d_NCHWc_strategy,
)
from tvm.relay import op as _op
from tvm import topi
from tvm.contrib.edgex.topi.conv2d import conv2d_nchw_tir_schedule
from .general import register_edgex_fschedule


@register_edgex_fschedule("nn.conv2d")
def conv2d_fschedule_edgex(attrs, prim_func, target):
    return conv2d_nchw_tir_schedule(attrs, prim_func, target)


@conv2d_NCHWc_strategy.register("edgex")
@conv2d_strategy.register("edgex")
def conv2d_strategy_edgex(attrs, inputs, out_type, target):
    """conv2d edgex strategy"""
    strategy = _op.OpStrategy()
    dilation_h, dilation_w = attrs.get_int_tuple("dilation")
    groups = attrs.groups
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")
    assert layout.startswith("NCHW") and kernel_layout.startswith(
        "OIHW"
    ), r"Only support 'NCHW(\d*c)?' and 'OIHW(\d*i\d*o)?' layout."
    if groups == 1:
        if layout == "NCHW":
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.conv2d_nchw),
                wrap_topi_schedule(topi.generic.schedule_conv2d_nchw),
                name="tir_conv2d_nchw.edgex",
            )
            # strategy.add_tir_implementation(
            #    wrap_compute_conv2d(topi.nn.conv2d_nchw),
            #    conv2d_nchw_tir_schedule,
            #    name="tir_conv2d_nchw.edgex",
            # )
        elif re.match(r"NCHW(\d*)c", layout):
            assert re.match(r"OIHW(\d*i\d*o)", kernel_layout)
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.conv2d_NCHWc, True, True),
                wrap_topi_schedule(topi.generic.schedule_conv2d_NCHWc),
                name="tir_conv2d_NCHWc.edgex",
            )
    else:
        strategy.add_implementation(
            wrap_compute_conv2d(topi.nn.group_conv2d_nchw, has_groups=True),
            wrap_topi_schedule(topi.generic.schedule_conv2d_nchw),
            name="tir_grouped_conv2d_nchw.edgex",
        )
        # strategy.add_tir_implementation(
        #    wrap_compute_conv2d(topi.nn.group_conv2d_nchw, has_groups=True),
        #    conv2d_nchw_tir_schedule,
        #    name="tir_group_conv2d_nchw.edgex",
        # )

    return strategy
