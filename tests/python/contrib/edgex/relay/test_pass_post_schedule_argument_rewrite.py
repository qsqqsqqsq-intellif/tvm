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
import numpy as np
from tvm.contrib.edgex.tir.schedule import EdgexSchedule
from tvm.contrib.edgex.relay.transform import (
    PostScheduleArgumentRewriteManager,
    PostScheduleArgumentRewrite,
)
from tvm.relay.build_module import bind_params_by_name
from tvm.contrib import graph_executor
from tvm.contrib.edgex.testing import TempOpStrategy
from tvm.contrib.edgex.topi.naive_vu_schedule import rewrite_quantize_params_to_u8
from tvm import topi
from tvm import te


s_relay_fused_conv = """
#[version = "0.0.5"]
def @main(%input: Tensor[(1, 3, 224, 224), int8],
          %weight: Tensor[(32, 3, 3, 3), int8],
          %bias: Tensor[(32), int32],
          %unused: Tensor[(32, 3, 3, 3), float32],
          %mulnorm: Tensor[(32, 1, 1), int64],
          %shiftnorm: Tensor[(32, 1, 1), int64]) {
    %7 = fn (%p0: Tensor[(1, 3, 224, 224), int8],
             %p1: Tensor[(32, 3, 3, 3), int8],
             %p2: Tensor[(32), int32],
             %p3: Tensor[(32, 1, 1), int64]
             %p4: Tensor[(32, 1, 1), int64], Primitive=1) {
        %0 = nn.conv2d(%p0, %p1, strides=[2, 2], padding=[1, 1, 1, 1],
                       channels=32, kernel_size=[3, 3], out_dtype="int32", data_layout="NCHW");
        %1 = nn.bias_add(%0, %p2);
        %2 = cast(%1, dtype="int64");
        %3 = multiply(%2, %p3);
        %4 = round_right_shift(%3, %p4);
        %5 = clip(%4, a_min=-128f, a_max=127f);
        %6 = cast(%5, dtype="int8");
        nn.relu(%6)
    };
    %7(%input, %weight, %bias, %mulnorm, %shiftnorm)
}
"""


def rewrite_conv_weight_layout_ochw(
    s: EdgexSchedule,
    block,
    relay_rewrite_mgr: PostScheduleArgumentRewriteManager = None,
):
    """rewrite conv weight layout by edgex cube weight layout convention"""
    axes = s.get_read_buffer_axes(block, 2)
    origin_weight_buffer = s.get_buffer_of(axes[0])

    c_o, c_i = axes[:2]
    delta_times, delta, co_unit_and_alpha = s.split_buffer(c_o, factors=[None, 2, 16])
    epsilon_times, epsilon_ci_part, beta = s.split_buffer(c_i, factors=[None, 1, 16])
    k = s.fuse_buffer(*axes[2:])
    s.reorder_buffer(delta_times, epsilon_times, delta, epsilon_ci_part, k, co_unit_and_alpha, beta)
    epsilon = s.fuse_buffer(epsilon_ci_part, k)
    new_weight_buffer = s.get_buffer_of(epsilon)

    # specify relay transformation for arguments
    def relay_forward_ochw(x):
        x = relay.nn.pad(x, [(0, 0), (0, 13), (0, 0), (0, 0)])
        x = relay.reshape(x, [1, 2, 16, 1, 1, 16, 9])
        x = relay.transpose(x, [0, 3, 1, 4, 6, 2, 5])
        x = relay.reshape(x, [1, 1, 2, 9, 16, 16])
        return x

    def relay_backward_ochw(x):
        x = relay.reshape(x, [1, 2, 1, 1, 9, 16, 16])
        x = relay.transpose(x, [0, 2, 5, 1, 3, 6, 4])
        x = relay.reshape(x, [32, 16, 3, 3])
        x = relay.strided_slice(x, [0, 0, 0, 0], [32, 3, 3, 3], slice_mode="size")
        return x

    if relay_rewrite_mgr is not None:
        relay_rewrite_mgr.trace_update(
            origin_buf=origin_weight_buffer,
            new_buf=new_weight_buffer,
            forward_transform=relay_forward_ochw,
            backward_transform=relay_backward_ochw,
        )


def my_conv_strategy(attrs, inputs, out_type, target):
    strategy = relay.op.OpStrategy()
    strategy.add_implementation(
        relay.op.strategy.generic.wrap_compute_conv2d(topi.nn.conv2d_nchw),
        relay.op.strategy.wrap_topi_schedule(topi.generic.schedule_conv2d_nchw),
    )
    return strategy


def conv_schedule_with_simu_layout_rewrite(attrs, func, target):
    s = EdgexSchedule(func)
    relay_rewrite_mgr = PostScheduleArgumentRewriteManager(s)
    rewrite_quantize_params_to_u8(s, relay_rewrite_mgr)
    rewrite_conv_weight_layout_ochw(s, s.get_block("compute"), relay_rewrite_mgr)
    return relay_rewrite_mgr.create_annotated_func()


def test_rewrite_nnp_conv_weight():
    """a testcase to check result before/after conv weight rewrite"""
    relay_mod = tvm.parser.parse(s_relay_fused_conv)
    x = tvm.nd.array(np.random.randint(0, 64, [1, 3, 224, 224]).astype("int8"))

    # must bind relay params to enable fold and rewrite global params
    relay_params = {}
    for p in relay_mod["main"].params[1:]:
        dtype = p.type_annotation.dtype
        shape = [int(x) for x in p.type_annotation.shape]
        arr = tvm.nd.array(np.random.randint(0, 8, shape).astype(dtype))
        relay_params[p.name_hint] = arr

    def get_output(m):
        # either (1) or (2) should be done (or both) after PostScheduleArgumentRewrite
        # to make the relay.build() work

        # (1) optional process step: constant folding
        # this can eliminate extra relay operations transforming the params
        func_with_params = bind_params_by_name(m["main"], relay_params)
        m = tvm.ir.IRModule.from_expr(func_with_params)
        m = relay.transform.FoldConstant()(m)

        # (2) optional process step: default fuse
        # this can wrap every op call into fused call of single op
        # one of the (1) and (2) step should be done
        # to make the relay.build() work
        m = relay.transform.FuseOps(fuse_opt_level=0)(m)

        # we assume PostScheduleArgumentRewrite is after other optimizations
        # thus we set opt_level=0 here
        with tvm.ir.transform.PassContext(config={"relay.backend.use_meta_schedule": True}):
            lib = relay.build(m, target="llvm")

        m = graph_executor.GraphModule(lib["default"](tvm.cpu()))
        m.set_input(0, x)
        m.run()
        return m.get_output(0).numpy()

    # check result without rewrite machanism, defuse to prevent cache not get hit
    defused_mod = relay.transform.DefuseOps()(relay_mod)
    expect = get_output(defused_mod)

    # run PostScheduleArgumentRewrite
    # note that cache may clear after relay.build() call
    from tvm.contrib.edgex.relay.backend import ScheduleCache

    dev = tvm.cpu()
    target = tvm.target.Target("llvm")
    with ScheduleCache():
        with TempOpStrategy(
            "nn.conv2d",
            "llvm",
            fschedule=conv_schedule_with_simu_layout_rewrite,
            fstrategy=my_conv_strategy,
        ):
            with tvm.ir.transform.PassContext(
                config={"relay.backend.use_meta_schedule": True}
            ) as pass_ctx:
                plan_config = tvm.target.make_compilation_config(
                    pass_ctx,
                    {
                        tvm.tir.IntImm("int32", dev.device_type): target,
                    },
                    target,
                )
                relay_mod = relay.transform.PlanDevices(plan_config)(relay_mod)
                updated_mod = PostScheduleArgumentRewrite()(relay_mod)
                y = get_output(updated_mod)
                tvm.testing.assert_allclose(expect, y, rtol=1e-5)


if __name__ == "__main__":
    test_rewrite_nnp_conv_weight()
