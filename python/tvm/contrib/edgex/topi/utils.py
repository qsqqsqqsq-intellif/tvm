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
# pylint: disable=invalid-name
"""Common edgex related utilities"""

from functools import reduce
import tvm
from tvm import tir
from tvm import relay
from tvm.contrib.edgex.tir.schedule import EdgexSchedule
from tvm.contrib.edgex.relay.transform import PostScheduleArgumentRewriteManager
from tvm.contrib.edgex.relay.op import cast_reinterpret


# Typename -> (typeid, bytes) for edgex datatypes
EDGEX_DTYPE_INFO = {
    "int8": (0, 1),
    "uint8": (1, 1),
    "float16": (2, 2),
    "float32": (3, 4),
    "int32": (4, 4),
    "int16": (4, 2),
}


def get_conv_epsilon_delta(
    num_co_group, num_ci_group, kernel_size, alpha=16, beta=16, sparsity_en=0, para_mode=0, pe_num=2
):
    """Calculate the epsilon and delta loop value"""
    # calculate epsilon, epsilon_times and last_epsilon
    if sparsity_en == 1:
        assert num_ci_group % 2 == 0, "The num_ci_group should be divided evenly by 2."
        num_ci_group_sparsity = num_ci_group // 2
        assert (
            num_ci_group_sparsity <= 16535
        ), "The num_ci_group_sparsity should less than or equal to the 16535."
    else:
        num_ci_group_sparsity = num_ci_group
        assert (
            num_ci_group_sparsity <= 32767
        ), "The num_ci_group_sparsity should less than or equal to the 32767."
    # TODO(someone): eps_ci_times default set 1,
    # it can be optimized by winograd enable or not respectively.
    eps_ci_times = 1
    ks = reduce(lambda x, y: x * y, list(kernel_size))
    ci_k_size = (num_ci_group_sparsity + beta - 1) // beta * ks
    # get the epsilon region loop value, to avoid injected bubble in epsilon
    # the last_epsilon must equal to epsilon.
    def _get_epsilon():
        nonlocal eps_ci_times
        epsilon = eps_ci_times * ks
        epsilon_times = (
            (num_ci_group_sparsity + beta - 1) // beta + eps_ci_times - 1
        ) // eps_ci_times
        if epsilon_times == 1:
            last_epsilon = epsilon
        else:
            last_epsilon = epsilon if (ci_k_size % epsilon) == 0 else (ci_k_size % epsilon)
        if last_epsilon != epsilon and epsilon_times > 1:
            eps_ci_times += 1
            return _get_epsilon()
        assert (
            last_epsilon == epsilon and epsilon_times > 0
        ), "Only support last_epsilon equal to epsilon."
        return epsilon, epsilon_times, eps_ci_times, last_epsilon

    epsilon_region_loops = _get_epsilon()
    # calculate delta, delta_times and last_delta
    # dense*zeta*delta*(winograd_en?16:1)*ALPHA*GAMMA<=16*256*4
    # assume max zeta is 16, max delta is 4, dense=1.
    delta = 4
    # TODO(someone): delta can be optimized in valid range
    max_delta = (num_co_group + alpha - 1) // alpha
    # get the delta region loop value, to avoid injected bubble in delta
    # the last_delta must equal to delta.
    def _get_delta():
        nonlocal delta
        if para_mode == 0:
            # tile para
            delta_thd = (num_co_group + alpha - 1) // alpha
            while delta > delta_thd:
                delta -= 1
            delta_times = ((num_co_group + alpha - 1) // alpha + delta - 1) // delta
        else:
            # co para
            delta_thd = ((num_co_group + alpha - 1) // alpha + pe_num - 1) // pe_num
            while delta > delta_thd:
                delta -= 1
            delta_times = ((num_co_group + alpha - 1) // alpha + delta * pe_num - 1) // (
                delta * pe_num
            )
        if delta_times == 1:
            last_delta = delta
        else:
            if para_mode == 0:
                # tile para
                last_delta = delta if max_delta % delta == 0 else max_delta % delta
            else:
                # co para
                last_delta = (
                    delta
                    if ((max_delta + pe_num - 1) // pe_num) % delta == 0
                    else ((max_delta + pe_num - 1) // pe_num) % delta
                )
        if last_delta != delta and delta > 1:
            delta -= 1
            return _get_delta()
        assert last_delta == delta and delta > 0, "Only support last_epsilon equal to epsilon."
        return delta, delta_times, last_delta

    delta_region_loops = _get_delta()
    return epsilon_region_loops, delta_region_loops


class PostConvOpMatcher:
    """A matcher to find ops after nu conv/matmul operation"""

    def __init__(self, s: tir.Schedule, channel_index: int):
        self.sched = s
        self.channel_index = channel_index
        self.bias_add_block = None
        self.pre_quantize_block = None
        self.quantize_multiply_block = None
        self.quantize_shift_block = None
        self.relu_block = None
        self.last_block = None

    def match(self, head_block):
        """match entrance"""
        self.last_block = None
        next_block = self.__match_bias_add(head_block)
        next_block = self.__match_multiply_round_right_shift(next_block)
        next_block = self.__match_relu(next_block)
        return self.last_block is not None

    def __match_bias_add(self, block):
        if block is None:
            return None
        block_stmt = self.sched.get_sref(block).stmt
        if self.__is_elemwise(block_stmt, "per_channel") and self.__is_bias_add(block_stmt):
            self.bias_add_block = block
            self.last_block = block
            return self.__next_block(block)
        return block

    def __match_multiply_round_right_shift(self, input_block):
        if input_block is None:
            return None

        # cast i64
        block_stmt = self.sched.get_sref(input_block).stmt
        if not self.__is_elemwise(block_stmt, "const") or not self.__is_cast(block_stmt, "int64"):
            return input_block

        # multiply
        multiply_block = self.__next_block(input_block)
        if multiply_block is None:
            return input_block
        block_stmt = self.sched.get_sref(multiply_block).stmt
        if not self.__is_elemwise(block_stmt, "per_channel") or not self.__is_multiply(block_stmt):
            return input_block

        # round right shift
        shift_block = self.__next_block(multiply_block)
        if shift_block is None:
            return input_block
        block_stmt = self.sched.get_sref(shift_block).stmt
        if not self.__is_elemwise(block_stmt, "per_channel") or not self.__is_round_right_shift(
            block_stmt
        ):
            return input_block

        # clip
        clip_block = self.__next_block(shift_block)
        if clip_block is None:
            return input_block
        block_stmt = self.sched.get_sref(clip_block).stmt
        if not self.__is_elemwise(block_stmt, "const") or not self.__is_clip_i8(block_stmt):
            return input_block

        # cast i8
        output_block = self.__next_block(clip_block)
        if output_block is None:
            return input_block
        block_stmt = self.sched.get_sref(output_block).stmt
        if not self.__is_elemwise(block_stmt, "const") or not self.__is_cast(block_stmt, "int8"):
            return input_block

        self.pre_quantize_block = input_block
        self.quantize_multiply_block = multiply_block
        self.quantize_shift_block = shift_block
        self.last_block = output_block
        return self.__next_block(output_block)

    def __match_relu(self, block):
        """match relu block"""
        if block is None:
            return None
        block_stmt = self.sched.get_sref(block).stmt
        if self.__is_elemwise(block_stmt, "const") and self.__is_relu(block_stmt):
            self.relu_block = block
            self.last_block = block
            return self.__next_block(block)
        return block

    def __next_block(self, block):
        s = self.sched
        consumers = s.get_consumers(block)
        if len(consumers) != 1:
            return None
        return consumers[0]

    def __is_elemwise(self, block_stmt, typ):
        if len(block_stmt.writes) != 1:
            return False
        if len(block_stmt.reads) > 3:
            return False
        if not isinstance(block_stmt.body, tir.BufferStore):
            return False
        iter_vars = {x.var for x in block_stmt.iter_vars}
        iter_extents = {x.var: x.dom.extent for x in block_stmt.iter_vars}
        store_indices = block_stmt.body.indices
        channel_axis_var = store_indices[self.channel_index] if self.channel_index >= 0 else None
        for expr in store_indices:
            if expr not in iter_vars:
                return False
            iter_vars.remove(expr)
        if len(iter_vars) > 0:
            return False
        valid = True

        def fvisit(obj):
            nonlocal valid
            if isinstance(obj, tir.BufferLoad):
                indices = obj.indices
                if len(indices) == len(store_indices) and all(
                    [x.same_as(y) for x, y in zip(indices, store_indices)]
                ):
                    return
                if typ == "const":
                    valid = False
                elif typ == "per_channel":
                    if any([x != 0 and not isinstance(x, tir.Var) for x in indices]):
                        valid = False
                        return
                    if channel_axis_var is not None:
                        indice_vars = [x for x in indices if isinstance(x, tir.Var)]
                        for v in indice_vars:
                            if iter_extents.get(v, 1) != 1 and not channel_axis_var.same_as(v):
                                valid = False

        tvm.tir.stmt_functor.post_order_visit(block_stmt.body, fvisit)
        return valid

    def __is_bias_add(self, block_stmt):
        expr = block_stmt.body.value
        return (
            isinstance(expr, tir.Add)
            and isinstance(expr.a, tir.BufferLoad)
            and isinstance(expr.b, tir.BufferLoad)
        )

    def __is_relu(self, block_stmt):
        expr = block_stmt.body.value
        return isinstance(expr, tir.Max) and (
            (isinstance(expr.a, tir.BufferLoad) and expr.b == 0)
            or (isinstance(expr.b, tir.BufferLoad) and expr.a == 0)
        )

    def __is_multiply(self, block_stmt):
        expr = block_stmt.body.value
        return (
            isinstance(expr, tir.Mul)
            and isinstance(expr.a, tir.BufferLoad)
            and isinstance(expr.b, tir.BufferLoad)
        )

    def is_elemwise_cast(self, block_stmt, dtype):
        """determine a block is elemwise cast operation"""
        return self.__is_elemwise(block_stmt, "const") and self.__is_cast(block_stmt, dtype)

    def __is_cast(self, block_stmt, dtype):
        """determine a block is cast operation"""
        expr = block_stmt.body.value
        return isinstance(expr, tir.Cast) and expr.dtype == dtype

    def __is_clip_i8(self, block_stmt):
        expr = block_stmt.body.value
        if isinstance(expr, tir.Min):
            if expr.b != 127:
                return False
            expr = expr.a
            return (
                isinstance(expr, tir.Max) and isinstance(expr.a, tir.BufferLoad) and expr.b == -128
            )
        if isinstance(expr, tir.Max):
            if expr.b != -128:
                return False
            expr = expr.a
            return (
                isinstance(expr, tir.Min) and isinstance(expr.a, tir.BufferLoad) and expr.b == 127
            )
        return False

    def __is_round_right_shift(self, block_stmt):
        expr = block_stmt.body.value
        return isinstance(expr, tir.Call) and expr.op.name == "tir.nnp_round_right_shift"


def rewrite_param_to_dtype(
    s: EdgexSchedule,
    producer,
    block,
    dtype: str,
    is_reinterpret: bool,
    relay_rewrite_mgr: PostScheduleArgumentRewriteManager = None,
):
    """Rewrite bias_add/quantize param buffer into specified dtype
    and inline param transformations like expand_dims.
    """

    def is_simple_transform(block_stmt):
        if not isinstance(block_stmt.body, tir.BufferStore):
            return False
        store_indices = block_stmt.body.indices
        load = block_stmt.body.value
        if not isinstance(load, tir.BufferLoad):
            return False
        load_indices = load.indices
        store_vars = [x for x in store_indices if isinstance(x, tir.Var)]
        load_vars = [x for x in load_indices if isinstance(x, tir.Var)]
        return (
            all([x == 0 or isinstance(x, tir.Var) for x in store_indices])
            and all([x == 0 or isinstance(x, tir.Var) for x in load_indices])
            and len(store_vars) == len(set(store_vars))
            and len(load_vars) == len(set(load_vars))
        )

    def inline_simple_transformation(block):
        producers = s.get_producers(block)
        for pb in producers:
            inline_simple_transformation(pb)
        if is_simple_transform(s.get_sref(block).stmt):
            s.compute_inline(block)

    # try inline all simple param transformations
    producer_sref = s.get_sref(producer)
    for pb in s.get_producers(block):
        sref = s.get_sref(pb)
        if sref.same_as(producer_sref):
            continue
        inline_simple_transformation(pb)

    def do_rewrite(buffer):
        shape = list(buffer.shape)
        elembytes = tvm.DataType(buffer.dtype).bits // 8
        new_elembytes = tvm.DataType(dtype).bits // 8
        factor = elembytes // new_elembytes
        if is_reinterpret:
            assert elembytes % new_elembytes == 0
            shape[-1] = shape[-1] * (elembytes // new_elembytes)
        new_buffer = tir.decl_buffer(
            shape,
            dtype,
            name=buffer.name + "_" + dtype,
        )
        new_buffers.append(new_buffer)

        def __load_rewrite(load):
            indices = list(load.indices)
            if is_reinterpret:
                indices[-1] *= factor
            return tir.Cast(load.dtype, tir.BufferLoad(load.buffer, indices))

        def __store_rewrite(store):
            indices = list(store.indices)
            if is_reinterpret:
                indices[-1] *= factor
            return tir.BufferStore(store.buffer, tir.Cast(dtype, store.value), indices)

        def __region_rewrite(ranges):
            ranges = list(ranges)
            ranges[-1] = tvm.ir.Range.from_min_extent(ranges[-1].min * factor, ranges[-1].extent)
            return ranges

        def __relay_forward(x):
            if is_reinterpret:
                return cast_reinterpret(x, dtype)
            return relay.cast(x, dtype)

        def __relay_backward(x):
            if is_reinterpret:
                return cast_reinterpret(x, buffer.dtype)
            return relay.cast(x, buffer.dtype)

        s.replace_buffer(
            block,
            buffer,
            new_buffer,
            load_rewrite=__load_rewrite,
            store_rewrite=__store_rewrite,
            region_rewrite=__region_rewrite if is_reinterpret else None,
        )

        # relay rewrite tracing
        if relay_rewrite_mgr is None:
            return

        relay_rewrite_mgr.trace_update(
            origin_buf=buffer,
            new_buf=new_buffer,
            forward_transform=__relay_forward,
            backward_transform=__relay_backward,
        )

    # detect and rewrite param buffer
    non_param_buffers = {x.buffer for x in producer_sref.stmt.writes}
    block_stmt = s.get_sref(block).stmt
    new_buffers = []
    for _, region in enumerate(block_stmt.reads):
        buffer = region.buffer
        if buffer in non_param_buffers:
            continue
        if buffer.dtype == dtype:
            continue
        do_rewrite(buffer)
    return new_buffers


def get_axes(s, block, buffer):
    """helper function to get buffer axes from block"""
    for i, region in enumerate(s.get_sref(block).stmt.reads):
        if region.buffer.same_as(buffer):
            return s.get_read_buffer_axes(block, i)
    return None


def relay_rewrite_per_channel_bias_only(
    s, block, bias_param_buf, n_channel, relay_rewrite_mgr=None
):
    """rewrite post conv param with only per-channel bias"""
    # Step (2.1): pack bias buffer as 64b per line
    axes = get_axes(s, block, bias_param_buf)
    bias_axis = axes[0] if len(axes) == 1 else s.fuse_buffer(*axes)
    new_axis, _ = s.split_buffer(bias_axis, factor=4 * 16)  # [lines, 64]

    # relay adaption
    def relay_forward_params(b):
        b = relay.reshape(b, [-1])
        b = relay.nn.pad(b, [(0, (16 - n_channel % 16) % 16 * 4)])
        b = relay.reshape(b, [-1, 16 * 4])
        return b

    def relay_backward_params(b):
        b = relay.reshape(b, [-1])
        b = relay.strided_slice(b, [0], [n_channel * 4], slice_mode="size")
        b = relay.reshape(b, [int(_) for _ in bias_param_buf.shape])
        return b

    if relay_rewrite_mgr is not None:
        relay_rewrite_mgr.trace_update(
            bias_param_buf,
            s.get_buffer_of(new_axis),
            forward_transform=relay_forward_params,
            backward_transform=relay_backward_params,
        )


def relay_rewrite_per_channel_bias_and_norm(
    s, block, bias_param_buf, multiply_param_buf, shift_param_buf, n_channel, relay_rewrite_mgr=None
):
    """rewrite post conv param with per-channel bias + norm"""
    # Step (1.1): pack bias buffer as 64b per line
    axes = get_axes(s, block, bias_param_buf)
    bias_axis = axes[0] if len(axes) == 1 else s.fuse_buffer(*axes)
    _, bias_i = s.split_buffer(bias_axis, factor=4 * 16)  # [lines, 64]

    # Step (1.2): stack and interleave mul&shift norm
    axes = get_axes(s, block, multiply_param_buf)
    mul_norm_axis = axes[0] if len(axes) == 1 else s.fuse_buffer(*axes)
    axes = get_axes(s, block, shift_param_buf)
    shift_norm_axis = axes[0] if len(axes) == 1 else s.fuse_buffer(*axes)
    s.stack_buffer(mul_norm_axis, shift_norm_axis)  # [2*co]
    two, co = s.split_buffer(mul_norm_axis, nparts=2)  # [2, co]
    s.reorder_buffer(co, two)
    _, co_i = s.split_buffer(co, factor=16)  # [lines, 16, 2]
    norm_i = s.fuse_buffer(co_i, two)  # [lines, 32]

    # Step (1.3): stack bias and norm params
    s.stack_buffer(bias_i, norm_i)  # [lines, 96]

    # relay adaption
    def relay_forward_params(b, m, s):
        b = relay.reshape(b, [-1])
        b = relay.nn.pad(b, [(0, (16 - n_channel % 16) % 16 * 4)])
        b = relay.reshape(b, [-1, 16 * 4])
        m = relay.reshape(m, [-1])
        m = relay.nn.pad(m, [(0, (16 - n_channel % 16) % 16)])
        s = relay.reshape(s, [-1])
        s = relay.nn.pad(s, [(0, (16 - n_channel % 16) % 16)])
        fused = relay.concatenate([m, s], axis=0)
        fused = relay.reshape(fused, [2, -1])
        fused = relay.transpose(fused, [1, 0])
        fused = relay.reshape(fused, [-1, 16 * 2])
        fused = relay.concatenate([b, fused], axis=1)
        return fused

    def relay_backward_params(fused):
        lines = (n_channel + 15) // 16
        b = relay.strided_slice(fused, [0, 0], [lines, 16 * 4], slice_mode="size")
        b = relay.reshape(b, [-1])
        b = relay.strided_slice(b, [0], [n_channel * 4], slice_mode="size")
        b = relay.reshape(b, [int(_) for _ in bias_param_buf.shape])
        fused = relay.strided_slice(fused, [0, 16 * 4], [lines, 32], slice_mode="size")
        fused = relay.reshape(fused, [-1, 2])
        fused = relay.transpose(fused, [1, 0])
        m = relay.strided_slice(fused, [0, 0], [1, n_channel], slice_mode="size")
        m = relay.reshape(m, [int(_) for _ in multiply_param_buf.shape])
        s = relay.strided_slice(fused, [1, 0], [1, n_channel], slice_mode="size")
        s = relay.reshape(s, [int(_) for _ in shift_param_buf.shape])
        return b, m, s

    if relay_rewrite_mgr is not None:
        relay_rewrite_mgr.trace_update(
            [bias_param_buf, multiply_param_buf, shift_param_buf],
            s.get_buffer_of(bias_i),
            forward_transform=relay_forward_params,
            backward_transform=relay_backward_params,
        )
