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
# pylint: disable=invalid-name, missing-function-docstring, unused-argument, unexpected-keyword-arg
"""Conv2D schedule on edgex"""
import re
from copy import deepcopy
from functools import reduce
import tvm
from tvm import tir, relay
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.utils import get_const_tuple
from tvm.contrib.edgex.tir.schedule import EdgexSchedule
from tvm.contrib.edgex.relay.transform import PostScheduleArgumentRewriteManager
from tvm.contrib.edgex.config import EdgexConfig
from tvm.contrib.edgex.base.edgexlog import EdgexLog as el
from tvm.contrib.edgex.arith.utils import ceiling_align
from .utils import (
    EDGEX_DTYPE_INFO,
    get_conv_epsilon_delta,
    PostConvOpMatcher,
    get_producer_block,
    relay_rewrite_per_channel_bias_and_norm,
    relay_rewrite_per_channel_bias_only,
    rewrite_param_to_dtype,
    get_line_num,
    get_conv_odma_output_bytes,
    get_element_bytes,
)


class Conv2dScheduleConfig:
    """Conv2d schedule configuration"""

    # global hardware configuration
    global_hw_cfg = EdgexConfig.get_current()

    # whether input data need load from DDR
    is_ddr_input: bool = True

    # whether output data need store to DDR
    is_ddr_output: bool = True

    # whether output channel tiling is enabled
    tile_oc: bool = False

    # output channel tiles num
    tile_oc_factor: int = 1

    # whether height tiling is enabled
    tile_h: bool = False

    # height tiles num
    tile_h_factor: int = 1

    # whether width tiling is enabled
    tile_w: bool = False

    # width tiles num
    tile_w_factor: int = 1

    # whether need cb buffer for weight
    cb_buffer: bool = False

    # conv configuration
    has_bias: bool = False
    has_norm: bool = False
    has_relu: bool = False  # for relu and leaky relu

    # cfg for nu isa
    alpha: int = 16
    beta: int = 16
    cube_enable: int = int(global_hw_cfg.PE_NUM) - 1
    epsilon: int = 0
    epsilon_times: int = 0
    eps_ci_times: int = 0
    last_epsilon: int = 0
    delta: int = 0
    delta_times: int = 0
    last_delta: int = 0
    sparsity_en: int = 0
    para_mode: int = 0
    psum_out_en: int = 1
    int_type: int = 0
    bias_mode: int = 0
    relu_mode: int = 0
    relu_round_mode: int = 4  # 0:ceiling; 1:floor; 2:truncate; 3:rounding off; 4:rounding
    norm_coeff_mode: int = 1
    odma_out_elem_bytes: int = 4
    ci_para_lines: int = 16
    co_para_unit_alpha: int = 16  # penult dim for rewrite weight


class Conv2dScheduler:
    """Conv2d schedule class for edgex"""

    def __init__(
        self,
        sch: tir.schedule.Schedule,
        conv_block: tir.schedule.BlockRV,
        input_shape,
        weight_shape,
        output_shape,
        input_dtype,
        weight_dtype,
        output_dtype,
        kernel_size,
        strides,
        padding,
        dilation,
        groups,
        data_layout: str = "NCHW",
        kernel_layout: str = "OIHW",
        relay_rewrite_mgr: PostScheduleArgumentRewriteManager = None,
        cfg: Conv2dScheduleConfig = None,
    ):
        # init schedule state
        self._sch: EdgexSchedule = sch
        self._relay_rewrite_mgr = relay_rewrite_mgr
        if cfg is None:
            cfg = Conv2dScheduleConfig()
        self._cfg = cfg

        # the block in prim_func
        self._conv_block = conv_block
        self._last_block = None

        # bias and quantize param buffer
        self._bias_param_buf = None
        self._multiply_param_buf = None
        self._shift_param_buf = None

        # conv attrs
        self._data_layout = data_layout
        self._input_shape = input_shape
        self._weight_shape = weight_shape
        self._output_shape = output_shape
        self._input_dtype = input_dtype
        self._weight_dtype = weight_dtype
        self._output_dtype = output_dtype
        self._groups = groups
        self._kernel_layout = kernel_layout
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding
        self._dilation = dilation
        self._interleaved_data = re.match(r"NCHW(\d*)c", data_layout)
        self._interleaved_weight = re.match(r"OIHW(\d*i\d*o)", kernel_layout)
        self._input_C = self._input_shape[1] * (
            self._input_shape[-1] if self._interleaved_data else 1
        )
        self._weight_O = self._weight_shape[0] * (
            self._weight_shape[-1] if self._interleaved_weight else 1
        )
        self._weight_I = self._weight_shape[1] * (
            self._weight_shape[-2] if self._interleaved_weight else 1
        )
        self._output_C = self._output_shape[1] * (
            self._output_shape[-1] if self._interleaved_data else 1
        )
        self._num_co_group = self._weight_O // groups
        # the weight ic not contain group usually
        self._num_ci_group = self._input_C // groups  # check if EQ self._weight_I

    def __rewrite_conv2d_weight_layout_oihw(self):
        """Rewrite conv2d weight layout by edgex cube weight layout convention"""
        cfg = self._cfg
        s = self._sch
        conv = self._conv_block
        if self._interleaved_weight:
            co, ci, kh, kw, ib, ob = self._sch.get_read_buffer_axes(conv, 2)
            self._sch.reorder_buffer(co, ob, ci, ib, kh, kw)
            self._sch.fuse_buffer(co, ob)
            self._sch.fuse_buffer(ci, ib)
            if self._relay_rewrite_mgr is not None:
                self._relay_rewrite_mgr.update_func_info()
        # get weight axes
        co, ci, kh, kw = s.get_read_buffer_axes(conv, 2)
        origin_weight_buffer = s.get_buffer_of(co)
        num_co_group_tile = (
            self._num_co_group // cfg.tile_oc_factor if cfg.tile_oc else self._num_co_group
        )
        num_ci_group = self._num_ci_group
        kernel_size = list(self._kernel_size)
        # calculate the injected bubble num.
        co_para_unit_alpha = cfg.co_para_unit_alpha
        add_co = 0
        if (num_co_group_tile % co_para_unit_alpha) != 0:
            add_co = co_para_unit_alpha - (num_co_group_tile % co_para_unit_alpha)
        add_ci = 0
        if (num_ci_group % cfg.beta) != 0:
            add_ci = (num_ci_group + cfg.beta - 1) // cfg.beta * cfg.beta - num_ci_group
        # reshape the weight co*ci*kernel to num_group*num_co_group*ci*kernel
        group_axis, co_group_axis = s.split_buffer(co, factors=[self._groups, self._num_co_group])
        if cfg.tile_oc:
            co_group_o, co_group_i = s.split_buffer(co_group_axis, nparts=cfg.tile_oc_factor)
        else:
            co_group_i = co_group_axis
        shape_post_split_co = [int(x) for x in s.get_buffer_of(group_axis).shape]
        # fill bubble and reshape.
        # step1, reshape at co/ci axis, fill bubble at co_unit_alpha and beta_unit axis,
        #   new_num_co, co_unit_alpha = s.split_buffer(co_group_i,
        #       factors=[None, co_para_unit_alpha])
        #   new_num_ci, beta_unit = s.split_buffer(ci, factors=[None, beta])
        # step2, reshape at new_num_co/new_num_ci axis, fill bubble at delta and epsilon axis,
        #   delta_times, delta = s.split_buffer(new_num_co,
        #       factors=[cfg.delta_times, cfg.delta])
        #   epsilon_times, eps_ci_times = s.split_buffer(new_num_ci,
        #       factors=[cfg.epsilon_times, cfg.eps_ci_times])
        # NOTICE: to avoid injected bubble in delta and epsilon,
        #  the last_delta and last_epsilon must equal to delta and epsilon respectively,
        #  so, step1 and step2 can be simplified as following formula.
        delta_times, delta, co_unit_alpha = s.split_buffer(
            co_group_i, factors=[cfg.delta_times, cfg.delta, co_para_unit_alpha]
        )
        epsilon_times, eps_ci_times, beta_unit = s.split_buffer(
            ci, factors=[cfg.epsilon_times, cfg.eps_ci_times, cfg.beta]
        )
        kernel = s.fuse_buffer(kh, kw)
        shape_pre_transpose = [int(x) for x in s.get_buffer_of(delta_times).shape]
        # transpose
        if cfg.tile_oc:
            s.reorder_buffer(
                co_group_o,
                group_axis,
                delta_times,
                epsilon_times,
                delta,
                eps_ci_times,
                kernel,
                co_unit_alpha,
                beta_unit,
            )
        else:
            s.reorder_buffer(
                delta_times, epsilon_times, delta, eps_ci_times, kernel, co_unit_alpha, beta_unit
            )
        shape_post_transpose = [int(x) for x in s.get_buffer_of(delta_times).shape]

        epsilon = s.fuse_buffer(eps_ci_times, kernel)
        shape_post_layout = [int(x) for x in s.get_buffer_of(epsilon).shape]
        new_weight_buffer = s.get_buffer_of(epsilon)

        # specify relay transformation for arguments
        def relay_forward_ochw(x):
            if self._interleaved_weight:
                x = relay.layout_transform(x, self._kernel_layout, "OIHW")
            x = relay.reshape(x, shape_post_split_co)
            if cfg.tile_oc:
                x = relay.nn.pad(x, [(0, 0), (0, 0), (0, add_co), (0, add_ci), (0, 0), (0, 0)])
            else:
                x = relay.nn.pad(x, [(0, 0), (0, add_co), (0, add_ci), (0, 0), (0, 0)])
            x = relay.reshape(x, shape_pre_transpose)
            if cfg.tile_oc:
                x = relay.transpose(x, [1, 0, 2, 5, 3, 6, 8, 4, 7])
            else:
                x = relay.transpose(x, [0, 1, 4, 2, 5, 7, 3, 6])
            x = relay.reshape(x, shape_post_layout)
            return x

        def relay_backward_ochw(x):
            x = relay.reshape(x, shape_post_transpose)
            if cfg.tile_oc:
                x = relay.transpose(x, [1, 0, 2, 4, 7, 3, 5, 8, 6])
                x = relay.reshape(
                    x,
                    [
                        self._groups,
                        cfg.tile_oc_factor,
                        num_co_group_tile + add_co,
                        num_ci_group + add_ci,
                    ]
                    + kernel_size,
                )
                x = relay.strided_slice(
                    x,
                    [0, 0, 0, 0, 0, 0],
                    [self._groups, cfg.tile_oc_factor, num_co_group_tile, num_ci_group]
                    + kernel_size,
                    slice_mode="size",
                )
            else:
                x = relay.transpose(x, [0, 1, 3, 6, 2, 4, 7, 5])
                x = relay.reshape(
                    x,
                    [self._groups, num_co_group_tile + add_co, num_ci_group + add_ci] + kernel_size,
                )
                x = relay.strided_slice(
                    x,
                    [0, 0, 0, 0, 0],
                    [self._groups, num_co_group_tile, num_ci_group] + kernel_size,
                    slice_mode="size",
                )
            x = relay.reshape(x, self._weight_shape)
            return x

        if self._relay_rewrite_mgr is not None:
            self._relay_rewrite_mgr.trace_update(
                origin_buf=origin_weight_buffer,
                new_buf=new_weight_buffer,
                forward_transform=relay_forward_ochw,
                backward_transform=relay_backward_ochw,
            )

    def tensorize_dma(self, num_outer_loops, dma_block, intrin_name, attrs):
        """Tensorize dma helper"""
        s = self._sch
        if intrin_name in ("nnp_wdma_load", "nnp_bdma_load"):
            axis_idx = num_outer_loops  # pragma at inner batch axis
        else:
            axis_idx = num_outer_loops + 1  # pragma at inner channel axis
        root_loop_sref = s.get_loops(dma_block)[axis_idx]
        sref = s.get_sref(dma_block)
        block_stmt = sref.stmt
        dtype = block_stmt.reads[0].buffer.dtype
        dummy_args = [dtype, 0, 0]
        attr_args = ["%s=%s" % (k, int(v)) for k, v in attrs.items()]
        s.pragma(
            root_loop_sref,
            "nnp_dma_scope",
            tir.Call("", "tir.%s" % intrin_name, dummy_args + attr_args),
        )

    def tensorize_eidma(
        self,
        num_outer_loops,
        eidma_block,
    ):
        """Tensorize eidma block"""
        s = self._sch
        cfg = self._cfg
        buf_axes = s.get_write_buffer_axes(eidma_block, 0)
        n, c, h, w = buf_axes[:4]
        group, ic_group = s.split_buffer(c, factors=[self._groups, None])
        if not self._interleaved_data:
            c1, c0 = s.split_buffer(ic_group, factor=cfg.ci_para_lines)
            s.reorder_buffer(n, group, c1, h, w, c0)

        loop_axes = s.get_loops(eidma_block)
        # NOTICE: a. tile C, split c(align by 16) should use the C dim divided group.
        # b. the C dimension must be divisible by 16/8 if convert to NCHWc in relay.
        # C -> group * ic_group
        group, ic_group = s.split(loop_axes[5], factors=[self._groups, None])
        if not self._interleaved_data:
            # ic_group -> c1 * 16/8
            c1, c0 = s.split(ic_group, factors=[None, cfg.ci_para_lines])
            # reorder to [no, co, ho, wo, ni, group, c1, hi, wi, c0]
            s.reorder(group, c1, loop_axes[6], loop_axes[7], c0)
            if self._input_shape[1] % cfg.ci_para_lines != 0:
                s.loop_partition([c1])
            # eidma_block write dm buffer should align last dimension with 16/8
            s.storage_align(
                eidma_block,
                0,
                len(s.get_sref(eidma_block).stmt.writes[0].buffer.shape) - 2,
                cfg.ci_para_lines,
                0,
            )
        # annotate at inner channel axis.
        s.pragma(s.get_loops(eidma_block)[num_outer_loops + 1], "nnp_dma_scope", "eidma")

    def tensorize_idma(
        self,
        num_outer_loops,
        idma_block,
        kernel_size,
        strides,
        dilation,
    ):
        """Tensorize idma block"""
        s = self._sch
        cfg = self._cfg
        buf_axes = s.get_write_buffer_axes(idma_block, 0)  # read buffer?
        n, c, h, w = buf_axes[:4]
        # C -> group, ic_group
        group, ic_group = s.split_buffer(c, factors=[self._groups, None])
        if not self._interleaved_data:
            c1, c0 = s.split_buffer(ic_group, factor=cfg.ci_para_lines)
            s.reorder_buffer(n, group, c1, h, w, c0)

        loop_axes = s.get_loops(idma_block)
        # C -> group, ic_group
        group, ic_group = s.split(loop_axes[5], factors=[self._groups, None])
        loop_axes = s.get_loops(idma_block)
        if self._interleaved_data:
            # idma data layout [group, ic_group, height, width, c0]
            s.pragma(loop_axes[num_outer_loops + 1], "nnp_data_layout", "GCHWc")
        else:
            # idma_block write dm buffer should align last dimension with 16/8
            s.storage_align(
                idma_block,
                0,
                len(s.get_sref(idma_block).stmt.writes[0].buffer.shape) - 2,
                cfg.ci_para_lines,
                0,
            )
            # idma data layout [group, ic_group, height, width]
            s.pragma(loop_axes[num_outer_loops + 1], "nnp_data_layout", "GCHW")
        # add annotate to prevent optimized if the loop extent is 1 in flatten buffer pass,
        # c must be 16
        s.annotate(loop_axes[num_outer_loops + 1], "preserve_unit_loop", 1)
        s.annotate(loop_axes[num_outer_loops + 2], "preserve_unit_loop", 1)
        s.annotate(loop_axes[num_outer_loops + 3], "preserve_unit_loop", 1)
        s.annotate(loop_axes[num_outer_loops + 4], "preserve_unit_loop", 1)

        dtype_id, _ = EDGEX_DTYPE_INFO[self._input_dtype]
        self.tensorize_dma(
            num_outer_loops,
            idma_block,
            "nnp_idma_load",
            attrs={
                "epsilon_idma": cfg.epsilon,
                "delta_idma": cfg.delta,
                "epsilon_times_idma": cfg.epsilon_times,
                "delta_times_idma": cfg.delta_times,
                "eps_ci_times_idma": cfg.eps_ci_times,
                "last_epsilon_idma": cfg.last_epsilon,
                "last_delta_idma": cfg.last_delta,
                "op_idma": 0,  # 0:conv, 1:matmul
                "para_mode_idma": cfg.para_mode,  # 0:tile para; 1:co para
                "wino_en_idma": 0,  # 0:disable winograd; 1: enable winograd
                "kernel1_speedup_flag_idma": 1 if kernel_size[1] == 1 else 0,  # enable if kw=1
                "sparsity_en_idma": cfg.sparsity_en,  # 0:disabel sparsity mode;
                # 1: enable sparsity mode
                "d_d_idma": 1,
                "d_h_idma": dilation[0],
                "d_w_idma": dilation[1],
                "s_d_idma": 1,
                "s_h_idma": strides[0],
                "s_w_idma": strides[1],
                "k_d_idma": 1,
                "k_h_idma": kernel_size[0],
                "k_w_idma": kernel_size[1],
                "epsilon_rewrite_ibuf_idma": 1,  # NOTE: All rewrite flag need set 1, maybe iss bug.
                "dense_times_rewrite_ibuf_idma": 1,
                "cube_enable_idma": cfg.cube_enable,  # 0:enable cube0; 1:enable cube0/1;
                # 2:enable cube0/1/2
                "data_type_idma": dtype_id,  # idma input data format
                "pad_v_idma": 0,  # padding value
                "num_group_idma": self._groups,
                "num_ci_group_idma": self._num_ci_group,
                "pad_mode_idma": 0,  # padding mode, 0:constant padding(pad_v); 1:edge padding
                "insert_d0_idma": 0,  # deconv dense insert 0 num
                "insert_h0_idma": 0,
                "insert_w0_idma": 0,
                "B_T_idma": 0,  # need config when enable matmul
                "B_dim2_idma": 0,
                "B_dim1_idma": 0,
            },
        )

    def tensorize_ewdma_wdma(self, num_outer_loops, ewdma_block, wdma_block, kernel_size):
        """Tensorize ewdma and wdma blocks"""
        s = self._sch
        cfg = self._cfg
        # process ewdma_block
        s.pragma(s.get_loops(ewdma_block)[num_outer_loops], "nnp_dma_scope", "eidma")
        # ewdma_block/wdma_block write dm buffer should align last dimension with 16/8
        s.storage_align(
            ewdma_block,
            0,
            len(s.get_sref(ewdma_block).stmt.writes[0].buffer.shape) - 2,
            cfg.beta,
            0,
        )
        s.storage_align(
            wdma_block,
            0,
            len(s.get_sref(wdma_block).stmt.writes[0].buffer.shape) - 2,
            cfg.beta,
            0,
        )
        # ewdma_block/wdma_block write dm buffer should align penultimate dimension with 16/32/48
        s.storage_align(
            ewdma_block,
            0,
            len(s.get_sref(ewdma_block).stmt.writes[0].buffer.shape) - 3,
            cfg.co_para_unit_alpha,
            0,
        )
        s.storage_align(
            wdma_block,
            0,
            len(s.get_sref(wdma_block).stmt.writes[0].buffer.shape) - 3,
            cfg.co_para_unit_alpha,
            0,
        )
        # tensorize wdma_block
        dtype_id, _ = EDGEX_DTYPE_INFO[self._weight_dtype]
        self.tensorize_dma(
            num_outer_loops,
            wdma_block,
            "nnp_wdma_load",
            attrs={
                "epsilon_wdma": cfg.epsilon,
                "delta_wdma": cfg.delta,
                "epsilon_times_wdma": cfg.epsilon_times,
                "delta_times_wdma": cfg.delta_times,
                "last_epsilon_wdma": cfg.last_epsilon,
                "last_delta_wdma": cfg.last_delta,
                "sparsity_en_wdma": cfg.sparsity_en,  # 0:disable sparsity mode; 1:enable
                "operation_wdma": 0,  # 0:conv; 1:matmul
                "data_type_wdma": dtype_id,
                "para_mode_wdma": cfg.para_mode,  # 0:tile para; 1:co para
                "cube_enable_wdma": cfg.cube_enable,  # 0:enable cube0; 1:enable cube0/1;
                # 2:enable cube0/1/2
                "rotate_en_wdma": 0,  # 0:disable weight rotate; 1:enable
                "num_group_wdma": self._groups,
                "A_transpose_wdma": 0,  # 0:disable matrix transpose; 1:enable
                "A_dim2_wdma": 0,
                "A_dim1_wdma": 0,
                "wt_addr_wrap_sel2_wdma": 0,
                "wt_addr_wrap_sel1_wdma": 1,
                "wt_st_addr_sel_wdma": 0,
                "st_addr_en_wdma": 1,
                "cb_mode_en_wdma": 1,
                "delta_bubble_inc_addr_wdma": 0,
                "epsilon_bubble_inc_addr_wdma": 0,
                "epsilon_times_rewrite_wbuf_wdma": 1,
                "delta_rewrite_wbuf_wdma": 1,  # NOTE: All rewrite flag need set 1, maybe iss bug.
                "k_size_wdma": kernel_size[0] * kernel_size[1],
                "bubble_insert_en_wdma": 1
                if self._input_dtype in ["int16"] and self._weight_dtype in ["int8", "uint8"]
                else 0,
            },
        )

    def tensorize_ebdma_bdma(self, num_outer_loops, ebdma_block, bdma_block):
        """Tensorize ebdma and bdma blocks"""
        s = self._sch
        cfg = self._cfg
        # process ebdma block
        if ebdma_block is not None:
            s.pragma(s.get_loops(ebdma_block)[num_outer_loops], "nnp_dma_scope", "ewdma")
        # tensorize bdma block
        if bdma_block is not None:
            # some constraints:
            # if delta_rewrite_nbbuf_bdma == 0, delta*4 <= BBUF_SIZE, which is 128
            # dense * zeta * delta * winograd? 16:1 <= OBUF_SIZE, which is 8*256*8
            self.tensorize_dma(
                num_outer_loops,
                bdma_block,
                "nnp_bdma_load",
                attrs={
                    "st_addr_sel": 0,
                    "addr_wrap_sel1": 0,
                    "addr_wrap_sel2": 0,
                    "num_group_bdma": self._groups,
                    "delta_times_bdma": cfg.delta_times,
                    "epsilon_times_bdma": cfg.epsilon_times,
                    "delta_bdma": cfg.delta,
                    "last_delta_bdma": cfg.last_delta,
                    "cube_work_num_bdma": cfg.cube_enable,
                    "parallel_mode_bdma": cfg.para_mode,
                    "bias_en_bdma": int(cfg.has_bias),
                    "norm_en_bdma": int(cfg.has_norm),
                    "delta_rewrite_nbbuf_bdma": 1,
                    "winograd_bdma": 0,
                    "bias_mode_bdma": cfg.bias_mode,
                    "norm_coeff_mode_bdma": cfg.norm_coeff_mode,
                },
            )

    def tensorize_cube(self, num_outer_loops, conv_block, kernel_size):
        """Tensorize compute block"""
        cfg = self._cfg
        input_dtype_id, _ = EDGEX_DTYPE_INFO[self._input_dtype]
        weight_dtype_id, _ = EDGEX_DTYPE_INFO[self._weight_dtype]
        self.tensorize_dma(
            num_outer_loops,
            conv_block,
            "nnp_cube_compute",
            attrs={
                "num_group_cube": self._groups,
                "epsilon_cube": cfg.epsilon,
                "delta_cube": cfg.delta,
                "epsilon_times_cube": cfg.epsilon_times,
                "delta_times_cube": cfg.delta_times,
                "last_epsilon_cube": cfg.last_epsilon,
                "last_delta_cube": cfg.last_delta,
                "k_size_cube": kernel_size[0] * kernel_size[1],  # must set 16, when enable winograd
                "bias_value_cube": 0,  # valid when bias_mode=1
                "cube_work_num_cube": cfg.cube_enable,  # 0:enable cube0; 1:enable cube0/1;
                # 2:enable cube0/1/2
                "winograd_cube": 0,  # 0:disable winograd; 1:enable
                "sparsity_en_cube": cfg.sparsity_en,  # 0:disable sparsity; 1:enable
                "bias_en_cube": int(cfg.has_bias),  # 0:bias disable; 1:enable
                "data_type_cube": input_dtype_id,
                "weight_type_cube": weight_dtype_id,
                # 0:each co bias independent; 1:each co share bias
                "bias_mode_cube": cfg.bias_mode,
                "round_mode_cube": 4,  # 0:ceiling; 1:floor; 2:truncate; 3:rounding off; 4:rounding
                "delta_rewrite_nbbuf_cube": 1,  # NOTE: All rewrite flag need set 1, maybe iss bug.
                "dense_times_rewrite_ibuf_cube": 1,
                "epsilon_rewrite_ibuf_cube": 1,
                "epsilon_times_rewrite_wbuf_cube": 1,
                "delta_rewrite_wbuf_cube": 1,
            },
        )

    def tensorize_odma(self, num_outer_loops, odma_block, strides):
        """Tensorize odma block"""
        s = self._sch
        cfg = self._cfg
        buf_axes = s.get_write_buffer_axes(odma_block, 0)
        n, c, h, w = buf_axes[:4]
        inner_axes = []
        outer_axes = []
        n_o, n_i = s.split_buffer(n, factors=[self._output_shape[0], None])
        inner_axes.append(n_i)
        outer_axes.append(n_o)
        group, oc_group = s.split_buffer(c, factors=[self._groups, None])
        oc_group_o, oc_group_i = s.split_buffer(oc_group, nparts=cfg.tile_oc_factor)
        inner_axes.append(group)
        outer_axes.append(oc_group_o)
        if self._interleaved_data:
            inner_axes.append(oc_group_i)
        else:
            c1, c0 = s.split_buffer(oc_group_i, factor=16)
            inner_axes.append(c1)
        h_o, h_i = s.split_buffer(h, factors=[cfg.tile_h_factor, None])
        inner_axes.append(h_i)
        outer_axes.append(h_o)
        w_o, w_i = s.split_buffer(w, factors=[cfg.tile_w_factor, None])
        inner_axes.append(w_i)
        outer_axes.append(w_o)
        if self._interleaved_data:
            inner_axes.append(buf_axes[-1])
        else:
            inner_axes.append(c0)
        s.reorder_buffer(*(outer_axes + inner_axes))
        if not self._interleaved_data:
            # odma_block write dm buffer should align last dimension with 16
            s.storage_align(
                odma_block, 0, len(s.get_sref(odma_block).stmt.writes[0].buffer.shape) - 2, 16, 0
            )
        # data_layout format contain "NCHW", "NCHWc", "NCDHW", "NCDHWc"
        loop_axes = s.get_loops(odma_block)
        if self._interleaved_data:
            s.pragma(loop_axes[num_outer_loops + 1], "nnp_data_layout", "CHWc")
        else:
            s.pragma(loop_axes[num_outer_loops + 1], "nnp_data_layout", "CHW")
        # add annotate to prevent optimized if the loop extent is 1 in flatten buffer pass,
        # c must be 16
        s.annotate(loop_axes[num_outer_loops + 1], "preserve_unit_loop", 1)
        s.annotate(loop_axes[num_outer_loops + 2], "preserve_unit_loop", 1)
        s.annotate(loop_axes[num_outer_loops + 3], "preserve_unit_loop", 1)

        dtype_id, _ = EDGEX_DTYPE_INFO[self._output_dtype]
        self.tensorize_dma(
            num_outer_loops,
            odma_block,
            "nnp_odma_store",
            attrs={
                "delta_odma": cfg.delta,
                "delta_times_odma": cfg.delta_times,
                "last_delta_odma": cfg.last_delta,
                "num_group_odma": self._groups,
                "addr_wrap_sel2_odma": 0,
                "addr_wrap_sel1_odma": 1,
                "extract_2to1_odma": 1 if strides[1] == 4 else 0,
                "int_type_odma": cfg.int_type,  # 1:output is int16
                "para_mode_odma": cfg.para_mode,  # 0:tile para; 1:co para
                "delta_mode_odma": 0,  # 0:each delta handle 16/32/48co; 1:each delta handle 16co
                "psum_out_en_odma": cfg.psum_out_en,  # 0:disable psum output; 1:enable.
                # psum_out_en must be 0, when winograd_en=1,
                "shiftnorm_odma": 0,
                "mulnorm_odma": 1,
                "norm_coeff_mode_odma": cfg.norm_coeff_mode,  # 0:read from NBUF; 1:fetch from isa
                "xbar_urr_weight_odma": 0,  # bus configuration, iss not simulate it.
                "relu_en_odma": int(cfg.has_relu),
                "round_mode_odma": 4,  # 0:ceiling; 1:floor; 2:truncate; 3:rounding off; 4:rounding
                "delta_rewrite_nbuf_odma": 1,
                "op_odma": 0,
                "wino_en_odma": 0,  # 0:disable winograd; 1:enable
                "cube_enable_odma": cfg.cube_enable,  # 0:enable cube0; 1:enable cube0/1;
                # 2:enable cube0/1/2
                "data_type_odma": dtype_id,  # odma input data format
                "relu_mode_odma": cfg.relu_mode,  # 0:relu; 1:leaky relu
                "leaky_relu_mode_odma": 0,  # 0:relu coeff read from rbuf;
                # 1:relu coeff read from isa
                "bias_mode_odma": cfg.bias_mode,  # 0:bias read from bbuf; 1:bias featch from isa
                "bias_en_odma": 0,  # 0:disable bias add; 1:enable
                "relu_round_mode_odma": cfg.relu_round_mode,  # same as round_mode
                "relu_sftcoeff_odma": 0,  # relu quantize shift coefficient.
                "relu_mulcoeff_odma": 0x1,  # relu quantize mul coefficient.
                "start_state_mode_odma": 1,
                "ub_channel_odma": 0x10,
                "end_state_odma": 1,
                "wo_channel_odma": 0x10,
            },
        )

    def tensorize_eodma(self, num_outer_loops, eodma_block):
        """Tensorize eodma block"""
        s = self._sch
        cfg = self._cfg
        loop_axes = s.get_loops(eodma_block)
        _, oc_group = s.split(loop_axes[5], factors=[self._groups, None])
        if not self._interleaved_data:
            oc_group_tile = self._num_co_group // cfg.tile_oc_factor
            c1, _ = s.split(oc_group, factors=[None, 16])
            if oc_group_tile % 16 != 0:
                s.loop_partition([c1])
        s.pragma(s.get_loops(eodma_block)[num_outer_loops + 1], "nnp_dma_scope", "eodma")

    def __tensorize_dma_intrinsics(
        self,
        num_outer_loops,
        eidma_block,
        ewdma_block,
        ebdma_block,
        idma_block,
        wdma_block,
        bdma_block,
        odma_block,
        eodma_block,
        conv_block,
    ):
        """Conv2d tensorize helper"""
        kernel_size = self._kernel_size
        strides = self._strides
        dilation = self._dilation
        if eidma_block is not None:
            self.tensorize_eidma(num_outer_loops, eidma_block)
        self.tensorize_idma(num_outer_loops, idma_block, kernel_size, strides, dilation)
        self.tensorize_ewdma_wdma(num_outer_loops, ewdma_block, wdma_block, kernel_size)
        if self._cfg.has_bias:
            self.tensorize_ebdma_bdma(num_outer_loops, ebdma_block, bdma_block)
        self.tensorize_cube(num_outer_loops, conv_block, kernel_size)
        self.tensorize_odma(num_outer_loops, odma_block, strides)
        if eodma_block is not None:
            self.tensorize_eodma(num_outer_loops, eodma_block)

    # todo(someone): process all conditions
    def __create_bdma_block(self):
        """Stack the bias and quantize parameter adapt the hardware convention"""
        # DOTO(someone): handle quantize relu parameter(relu multiply and relu shift)
        s = self._sch
        cfg = self._cfg
        # sanity checks
        n_channel = self._weight_shape[0]
        if cfg.has_bias:
            if cfg.has_norm:
                relay_rewrite_per_channel_bias_and_norm(
                    s,
                    self._last_block,
                    self._bias_param_buf,
                    self._multiply_param_buf,
                    self._shift_param_buf,
                    n_channel,
                    self._relay_rewrite_mgr,
                )
            else:
                relay_rewrite_per_channel_bias_only(
                    s, self._last_block, self._bias_param_buf, n_channel, self._relay_rewrite_mgr
                )
        else:
            raise NotImplementedError("not implemented")

    def __swift_tile_cfg(self):
        """Get the tile configuration helper function."""
        cfg = self._cfg
        if cfg.global_hw_cfg is None:
            el.e("Invalid edgex configuration.")
        if not self._data_layout.startswith("NCHW"):
            el.e("Only support 'NCHW(16c)' data layout tile")
        # naively estimate the total size
        def get_total_sizes(oc_factor: int = 1, h_factor: int = 1, w_factor: int = 1):
            # estimate the input size
            input_shape = deepcopy(self._input_shape)
            input_shape[2] = input_shape[2] // h_factor + self._padding[0] + self._padding[2]
            input_shape[3] = input_shape[3] // w_factor + self._padding[1] + self._padding[3]
            if not self._interleaved_data:
                input_shape[1] = ceiling_align(input_shape[1], cfg.ci_para_lines)
            elems = reduce(lambda x, y: x * y, input_shape)
            input_sizes = elems * get_element_bytes(self._input_dtype)
            # estimate the weight size
            weight_shape = deepcopy(self._weight_shape)
            # weight oc align factor can be refined by para mode if need
            weight_shape[0] = weight_shape[0] // oc_factor
            if not self._interleaved_weight:
                weight_shape[0] = ceiling_align(weight_shape[0], 48)
                weight_shape[1] = ceiling_align(weight_shape[1], cfg.ci_para_lines)
            elems = reduce(lambda x, y: x * y, weight_shape)
            weight_sizes = elems * get_element_bytes(self._weight_dtype)
            # estimate the output size
            output_shape = deepcopy(self._output_shape)
            # oc need align by 16
            output_shape[1] = output_shape[1] // oc_factor
            if not self._interleaved_data:
                output_shape[1] = ceiling_align(output_shape[1], 16)
            output_shape[2] = output_shape[2] // h_factor
            output_shape[3] = output_shape[3] // w_factor
            elems = reduce(lambda x, y: x * y, output_shape)
            output_sizes = elems * cfg.odma_out_elem_bytes
            total_sizes = input_sizes + weight_sizes + output_sizes
            return total_sizes

        def do_tiling():
            # TODO(someone): can generate multiple valid factor list,
            # and find the optimal one by cost model.
            # the number greater than the init factor can only become factor possible.
            nonlocal total_sizes
            nonlocal dm_size
            # oc_num = ceiling_align(self._output_shape[1], 16)
            # TODO(someone): inner oc default is 16, group need consider
            oc_factor = 1
            oc_factor_list = list()
            h_factor = 1
            h_factor_list = list()
            w_factor = 1
            w_factor_list = list()
            total_sizes = get_total_sizes(oc_factor=oc_factor, h_factor=h_factor, w_factor=w_factor)
            while total_sizes > dm_size:
                if self._output_shape[1] // oc_factor > 15:
                    oc_factor += 1
                    if (
                        self._output_shape[1] % oc_factor == 0
                        and self._output_shape[1] // oc_factor > 15
                    ):
                        oc_factor_list.append(oc_factor)
                        total_sizes = get_total_sizes(
                            oc_factor=oc_factor, h_factor=h_factor, w_factor=w_factor
                        )
                        if total_sizes < dm_size:
                            break
                if self._output_shape[2] // h_factor > 1:
                    h_factor += 1
                    if self._output_shape[2] % h_factor == 0:
                        h_factor_list.append(h_factor)
                        total_sizes = get_total_sizes(
                            oc_factor=oc_factor, h_factor=h_factor, w_factor=w_factor
                        )
                        if total_sizes < dm_size:
                            break
                if self._output_shape[3] // w_factor > 1:
                    w_factor += 1
                    if self._output_shape[3] % w_factor == 0:
                        w_factor_list.append(w_factor)
                        total_sizes = get_total_sizes(
                            oc_factor=oc_factor, h_factor=h_factor, w_factor=w_factor
                        )
                        if total_sizes < dm_size:
                            break
                if self._output_shape[2] // h_factor < 2 and self._output_shape[3] // w_factor < 2:
                    break
            if len(oc_factor_list) > 0:
                cfg.tile_oc = True
                cfg.tile_oc_factor = oc_factor_list[-1]
            if len(h_factor_list) > 0:
                cfg.tile_h = True
                cfg.tile_h_factor = h_factor_list[-1]
            if len(w_factor_list) > 0:
                cfg.tile_w = True
                cfg.tile_w_factor = w_factor_list[-1]
            return total_sizes < dm_size

        dm_size = cfg.global_hw_cfg.DM_SIZE
        total_sizes = get_total_sizes()
        # do tile
        if total_sizes > dm_size:
            if not do_tiling():
                # TODO(someone): wether use cb_buffer need estimate in detail
                cfg.cb_buffer = True

    def analyze(self):
        """Analyze conv2d prim_func to get the blocks info and the configuration info."""
        s = self._sch
        cfg = self._cfg
        conv = self._conv_block
        # TODO(someone): wether handle "NCHWc"
        channel_index = self._data_layout.find("C")
        matcher = PostConvOpMatcher(s, channel_index=channel_index)
        conv_consumers = s.get_consumers(conv)
        if len(conv_consumers) == 1:
            if matcher.match(conv_consumers[0]):
                # rewrite bias parameter
                if matcher.bias_add_block is not None:
                    (self._bias_param_buf,) = rewrite_param_to_dtype(
                        s,
                        conv,
                        matcher.bias_add_block,
                        dtype="int8",
                        is_reinterpret=True,
                        relay_rewrite_mgr=self._relay_rewrite_mgr,
                    )
                    if self._bias_param_buf.dtype != "int8":
                        el.e(
                            "Rewrite bias param dtype need int8, but get: %s"
                            % self._bias_param_buf.dtype
                        )
                    cfg.has_bias = True
                # rewrite quantize multiply and shift parameter
                if matcher.quantize_multiply_block is not None:
                    assert matcher.quantize_shift_block is not None
                    (self._multiply_param_buf,) = rewrite_param_to_dtype(
                        s,
                        matcher.pre_quantize_block,
                        matcher.quantize_multiply_block,
                        dtype="int8",
                        is_reinterpret=True,
                        pre_cast_dtype="uint16",
                        relay_rewrite_mgr=self._relay_rewrite_mgr,
                    )
                    (self._shift_param_buf,) = rewrite_param_to_dtype(
                        s,
                        matcher.quantize_multiply_block,
                        matcher.quantize_shift_block,
                        dtype="int8",
                        is_reinterpret=False,
                        relay_rewrite_mgr=self._relay_rewrite_mgr,
                    )
                    if (
                        self._multiply_param_buf.dtype != "int8"
                        or self._shift_param_buf.dtype != "int8"
                    ):
                        el.e(
                            "Rewrite multiply and shift param dtype need int8, but get: %s and %s"
                            % (self._multiply_param_buf.dtype, self._shift_param_buf.dtype)
                        )
                    cfg.has_norm = True
                if cfg.has_bias:
                    cfg.bias_mode = 0
                    if cfg.has_norm:
                        cfg.norm_coeff_mode = 0
                        cfg.psum_out_en = 0
                    else:
                        cfg.psum_out_en = 1
                # relu configuration
                if matcher.relu_block is not None:
                    cfg.has_relu = True
                    cfg.relu_mode = 0
                self._last_block = matcher.last_block
        # get other configuration
        cfg.ci_para_lines = get_line_num(self._input_dtype)
        cfg.beta = cfg.ci_para_lines
        cfg.odma_out_elem_bytes = get_conv_odma_output_bytes(
            cfg.psum_out_en, self._output_dtype, cfg.int_type
        )
        co_para_unit = 1 if cfg.para_mode == 0 else cfg.global_hw_cfg.PE_NUM  # use cube enable
        cfg.co_para_unit_alpha = co_para_unit * 16
        # tile configuration
        self.__swift_tile_cfg()
        # epsilon and delta region loops
        num_co_group_tile = (
            self._num_co_group // cfg.tile_oc_factor if cfg.tile_oc else self._num_co_group
        )
        # calculate the loop value.
        (epsilon_region_loops, delta_region_loops,) = get_conv_epsilon_delta(
            num_co_group=num_co_group_tile,
            num_ci_group=self._num_ci_group,
            kernel_size=self._kernel_size,
            alpha=cfg.alpha,
            beta=cfg.beta,
            sparsity_en=cfg.sparsity_en,
            para_mode=cfg.para_mode,
            pe_num=cfg.global_hw_cfg.PE_NUM,
        )
        (
            cfg.epsilon,
            cfg.epsilon_times,
            cfg.eps_ci_times,
            cfg.last_epsilon,
        ) = epsilon_region_loops
        cfg.delta, cfg.delta_times, cfg.last_delta = delta_region_loops

    def preprocess(self):
        """Inline all blocks, create bdma block, and rewrite weight."""
        s = self._sch
        cfg = self._cfg
        last_block = self._last_block
        # try inline post conv ops
        if last_block is not None:
            last_sref = s.get_sref(last_block)
            cur = s.get_consumers(self._conv_block)[0]
            while True:
                cur_sref = s.get_sref(cur)
                if cur_sref.same_as(last_sref):
                    break
                next_block = s.get_consumers(cur)[0]
                s.compute_inline(cur)
                cur = next_block
        # Stack the bias and quantize parameter adapt the hardware convention
        if last_block is not None and cfg.has_bias:
            self.__create_bdma_block()
        # refactor weight layouts adapt the hardware convention.
        self.__rewrite_conv2d_weight_layout_oihw()

    def schedule(self):
        """Conv2d edgex schedule helper"""
        s = self._sch
        cfg = self._cfg
        conv_block = self._conv_block
        pad_block = get_producer_block(s, conv_block, 1)  # read order: [out, in, weight]
        idma_block = s.cache_read(conv_block, 1, "iobuf")
        s.compute_inline(pad_block)
        # if original input is on DDR, create a cache read into DM
        eidma_block = None
        if cfg.is_ddr_input:
            eidma_block = s.cache_read(idma_block, 0, "dm")
        # read weight from ddr -> dm -> wbuf
        wdma_block = s.cache_read(conv_block, 2, "wbuf")
        ewdma_block = s.cache_read(wdma_block, 0, "dm")
        bdma_block = ebdma_block = None
        if cfg.has_bias and self._last_block is not None:
            bdma_block = s.cache_read(self._last_block, 1, "bbuf")
            ebdma_block = s.cache_read(bdma_block, 0, "dm")  # ewdma->bbuf
        odma_block = s.cache_write(conv_block, 0, "cube")
        if self._last_block is not None:
            s.compute_inline(odma_block)
            odma_block = self._last_block
        output_block = odma_block
        # if output will write to ddr, create a cache write from DM
        eodma_block = None
        if cfg.is_ddr_output:
            eodma_block = s.cache_write(odma_block, 0, "dm")
            output_block = eodma_block
        else:
            buf = self._sch.get_sref(odma_block).stmt.writes[0].buffer
            tmp_buf = tvm.tir.decl_buffer(buf.shape, buf.dtype, buf.name, scope="dm")
            self._sch.replace_buffer(odma_block, buf, tmp_buf)

        # NCHW or NCHWc
        loop_axes = s.get_loops(output_block)
        inner_axes = []
        outer_axes = []
        n_o, n_i = s.split(loop_axes[0], factors=[self._output_shape[0], None])
        inner_axes.append(n_i)
        outer_axes.append(n_o)
        c_o, c_i = s.split(loop_axes[1], factors=[cfg.tile_oc_factor, None])
        inner_axes.append(c_i)
        outer_axes.append(c_o)
        h_o, h_i = s.split(loop_axes[2], factors=[cfg.tile_h_factor, None])
        inner_axes.append(h_i)
        outer_axes.append(h_o)
        w_o, w_i = s.split(loop_axes[3], factors=[cfg.tile_w_factor, None])
        inner_axes.append(w_i)
        outer_axes.append(w_o)
        if self._interleaved_data:
            inner_axes.append(loop_axes[-1])
        s.reorder(*(outer_axes + inner_axes))
        s.loop_partition([h_o, w_o], True)
        # replace bias buffer range in odma block
        if self._last_block is not None:
            odma_block_stmt = s.get_sref(odma_block).stmt
            read0_region = odma_block_stmt.reads[0]
            read1_buffer = odma_block_stmt.reads[1].buffer
            read1_region = tir.BufferRegion(
                read1_buffer,
                [odma_block_stmt.reads[1].region[0], tvm.ir.Range.from_min_extent(0, 112)],
            )
            block_body = odma_block_stmt.body
            repl_block = tir.Block(
                odma_block_stmt.iter_vars,
                [read0_region, read1_region],
                [odma_block_stmt.writes[0]],
                odma_block_stmt.name_hint,
                block_body,
            )
            s.state.replace(s.get_sref(odma_block), repl_block, {odma_block_stmt: repl_block})
        # replace weight region, the last 2 dim should be co_para_unit_alpha and beta
        conv_block_stmt = s.get_sref(conv_block).stmt
        read2_region = conv_block_stmt.reads[2]
        read2_buffer = read2_region.buffer
        read2_region = tir.BufferRegion(
            read2_buffer,
            read2_region.region[:-2]
            + [
                tvm.ir.Range.from_min_extent(0, read2_buffer.shape[-2]),
                tvm.ir.Range.from_min_extent(0, read2_buffer.shape[-1]),
            ],
        )
        repl_block = tir.Block(
            conv_block_stmt.iter_vars,
            [conv_block_stmt.reads[0], conv_block_stmt.reads[1], read2_region],
            [conv_block_stmt.writes[0]],
            conv_block_stmt.name_hint,
            conv_block_stmt.body,
            conv_block_stmt.init,
        )
        s.state.replace(s.get_sref(conv_block), repl_block, {conv_block_stmt: repl_block})
        if cfg.is_ddr_output:
            s.compute_at(odma_block, outer_axes[-1], preserve_unit_loops=True)
        s.compute_at(conv_block, outer_axes[-1], preserve_unit_loops=True)
        s.compute_at(idma_block, outer_axes[-1], preserve_unit_loops=True)
        if cfg.is_ddr_input:
            s.compute_at(eidma_block, outer_axes[-1], preserve_unit_loops=True)
        s.compute_at(wdma_block, outer_axes[-1], preserve_unit_loops=True)
        s.compute_at(ewdma_block, outer_axes[-1], preserve_unit_loops=True)
        if cfg.has_bias:
            s.compute_at(bdma_block, outer_axes[-1], preserve_unit_loops=True)
            s.compute_at(ebdma_block, outer_axes[-1], preserve_unit_loops=True)
        s.pragma(
            s.get_loops(odma_block)[4],
            "nnp_num_co",
            self._output_C // cfg.tile_oc_factor,
        )
        self.__tensorize_dma_intrinsics(
            len(outer_axes),
            eidma_block,
            ewdma_block,
            ebdma_block,
            idma_block,
            wdma_block,
            bdma_block,
            odma_block,
            eodma_block,
            conv_block,
        )


def schedule_edgex_conv_block(
    sched,
    conv,
    kernel_size,
    strides,
    padding,
    dilation,
    groups,
    data_layout: str = "NCHW",
    kernel_layout: str = "OIHW",
    relay_rewrite_mgr: PostScheduleArgumentRewriteManager = None,
    cfg: Conv2dScheduleConfig = None,
):
    """schedule edgex convolution on current schedule state"""
    if not (data_layout.startswith("NCHW") and kernel_layout.startswith("OIHW")):
        el.e(r"Only support 'NCHW(16c)?' and 'OIHW(16i16o)?' layout.")
    block_stmt = sched.get_sref(conv).stmt
    output_buffer = block_stmt.reads[0].buffer
    input_buffer = block_stmt.reads[1].buffer
    weight_buffer = block_stmt.reads[2].buffer
    output_shape = list(get_const_tuple(output_buffer.shape))
    input_shape = list(get_const_tuple(input_buffer.shape))
    weight_shape = list(get_const_tuple(weight_buffer.shape))

    # schedule use input shape before padding
    input_shape[2] -= padding[0] + padding[2]
    input_shape[3] -= padding[1] + padding[3]

    input_dtype = input_buffer.dtype
    weight_dtype = weight_buffer.dtype
    output_dtype = output_buffer.dtype
    scheduler = Conv2dScheduler(
        sched,
        conv,
        input_shape=input_shape,
        output_shape=output_shape,
        weight_shape=weight_shape,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        weight_dtype=weight_dtype,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
        relay_rewrite_mgr=relay_rewrite_mgr,
        cfg=cfg,
    )
    # analyze the conv prim_func to get the blocks info and the configuration info.
    scheduler.analyze()
    # inline all blocks, create bdma block, and rewrite the weight.
    scheduler.preprocess()
    # start schedule
    scheduler.schedule()


def conv2d_nchw_tir_schedule(attrs, prim_func, tgt):
    """Conv2d edgex tir schedule"""
    kernel_size = attrs.get_int_tuple("kernel_size")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    padding = attrs.get_int_tuple("padding")
    padding = get_pad_tuple(padding, kernel_size)
    groups = attrs.groups
    data_layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    sched = EdgexSchedule(prim_func)
    relay_rewrite_mgr = PostScheduleArgumentRewriteManager(sched)
    conv_block = sched.get_child_blocks(sched.get_block("root"))[1]
    schedule_edgex_conv_block(
        sched,
        conv_block,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
        relay_rewrite_mgr=relay_rewrite_mgr,
    )
    return relay_rewrite_mgr.create_annotated_func()
