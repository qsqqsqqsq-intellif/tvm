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
"""Conv3D schedule on edgex"""
import re
import tvm
from tvm import tir, relay
from tvm.topi.nn.utils import get_pad_tuple3d
from tvm.topi.utils import get_const_tuple
from tvm.contrib.edgex.tir.schedule import EdgexSchedule
from tvm.contrib.edgex.relay.transform import PostScheduleArgumentRewriteManager
from tvm.contrib.edgex.config import EdgexConfig
from tvm.contrib.edgex.base.edgexlog import EdgexLog as el
from .utils import (
    EDGEX_DTYPE_INFO,
    get_conv_epsilon_delta,
    PostConvOpMatcher,
    get_producer_block,
    relay_rewrite_per_channel_bias_and_norm,
    relay_rewrite_per_channel_bias_only,
    rewrite_param_to_dtype,
    swift_tile_cfg,
    get_line_num,
    get_conv_odma_output_bytes,
)


class Conv3dScheduleConfig:
    """Conv3d schedule configuration"""

    # whether input data need load from DDR
    is_ddr_input: bool = True

    # whether output data need store to DDR
    is_ddr_output: bool = True

    # whether output channel tiling is enabled
    tile_co: bool = False

    # output channel tiles num
    tile_co_num: int = -1


class ScheduleConv3d:
    """Conv3d schedule class for edgex"""

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
        layout,
        kernel_layout,
        relay_rewrite_mgr: PostScheduleArgumentRewriteManager = None,
        cfg: Conv3dScheduleConfig = None,
    ):
        # init schedule state
        self._sch: EdgexSchedule = sch
        self._conv_block = conv_block
        self._relay_rewrite_mgr = relay_rewrite_mgr
        if cfg is None:
            cfg = Conv3dScheduleConfig()
        self._cfg = cfg
        self._global_hw_cfg = EdgexConfig.get_current()
        self._cube_enable = int(self._global_hw_cfg.PE_NUM) - 1

        # conv attrs
        self._input_shape = input_shape
        self._weight_shape = weight_shape
        self._output_shape = output_shape
        self._input_dtype = input_dtype
        self._weight_dtype = weight_dtype
        self._output_dtype = output_dtype
        self._groups = groups
        self._layout = layout
        self._kernel_layout = kernel_layout
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding
        self._dilation = dilation
        self._interleaved_data = re.match(r"NCDHW(\d*)c", layout)
        self._interleaved_weight = re.match(r"OIDHW(\d*i\d*o)", kernel_layout)
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

        # conv internal attrs
        self._epsilon = 0
        self._epsilon_times = 0
        self._eps_ci_times = 0
        self._last_epsilon = 0
        self._delta = 0
        self._delta_times = 0
        self._last_delta = 0
        self._sparsity_en = 0
        self._para_mode = 0
        self._psum_out_en = 1
        self._int_type = 0
        self._has_bias = False
        self._has_norm = False
        self._has_relu = False  # for relu and leaky relu
        self._bias_mode = 0
        self._relu_mode = 0
        self._relu_round_mode = 4  # 0:ceiling; 1:floor; 2:truncate; 3:rounding off; 4:rounding
        self._norm_coeff_mode = 1
        self._odma_out_elem_bytes = 4
        self._ci_para_lines = 16

    def __rewrite_conv3d_weight_layout_oidhw(
        self,
        conv,
    ):
        """Rewrite conv3d weight layout by edgex cube weight layout convention"""
        # schedule use OIDHW weight layout
        if self._interleaved_weight:
            co, ci, kd, kh, kw, ib, ob = self._sch.get_read_buffer_axes(conv, 2)
            self._sch.reorder_buffer(co, ob, ci, ib, kd, kh, kw)
            self._sch.fuse_buffer(co, ob)
            self._sch.fuse_buffer(ci, ib)
            if self._relay_rewrite_mgr is not None:
                self._relay_rewrite_mgr.update_func_info()
        co, ci, kd, kh, kw = self._sch.get_read_buffer_axes(conv, 2)
        origin_weight_buffer = self._sch.get_buffer_of(co)
        num_co_group = self._weight_O // self._groups
        if self._cfg.tile_co:
            num_co_group_tile = num_co_group // self._cfg.tile_co_num
        else:
            num_co_group_tile = num_co_group
        num_ci_group = self._weight_I
        kernel_size = self._weight_shape[2:]
        alpha = 16
        beta = self._ci_para_lines
        # get the loop value.
        (epsilon_region_loops, delta_region_loops,) = get_conv_epsilon_delta(
            num_co_group_tile,
            num_ci_group,
            kernel_size,
            alpha,
            beta,
            self._sparsity_en,
            self._para_mode,
            self._global_hw_cfg.PE_NUM,
        )
        (
            self._epsilon,
            self._epsilon_times,
            self._eps_ci_times,
            self._last_epsilon,
        ) = epsilon_region_loops
        self._delta, self._delta_times, self._last_delta = delta_region_loops
        # calculate the injected bubble num.
        co_para_unit = 1 if self._para_mode == 0 else self._global_hw_cfg.PE_NUM
        co_para_unit_alpha = co_para_unit * 16
        add_co = 0
        if (num_co_group_tile % co_para_unit_alpha) != 0:
            add_co = co_para_unit_alpha - (num_co_group_tile % co_para_unit_alpha)
        add_ci = 0
        if (num_ci_group % beta) != 0:
            add_ci = (num_ci_group + beta - 1) // beta * beta - num_ci_group
        # reshape the weight co*ci*kernel to num_group*num_co_group*ci*kernel
        group_axis, co_group_axis = self._sch.split_buffer(co, factors=[self._groups, num_co_group])
        if self._cfg.tile_co:
            co_group_o, co_group_i = self._sch.split_buffer(
                co_group_axis, nparts=self._cfg.tile_co_num
            )
        else:
            co_group_i = co_group_axis
        shape_post_split_co = [int(x) for x in self._sch.get_buffer_of(group_axis).shape]
        # fill bubble and reshape.
        # step1, reshape at co/ci axis, fill bubble at co_unit_alpha and beta_unit axis,
        #   new_num_co, co_unit_alpha = self._sch.split_buffer(co_group_i,
        #       factors=[None, co_para_unit_alpha])
        #   new_num_ci, beta_unit = self._sch.split_buffer(ci, factors=[None, beta])
        # step2, reshape at new_num_co/new_num_ci axis, fill bubble at delta and epsilon axis,
        #   delta_times, delta = self._sch.split_buffer(new_num_co,
        #       factors=[self._delta_times, self._delta])
        #   epsilon_times, eps_ci_times = self._sch.split_buffer(new_num_ci,
        #       factors=[self._epsilon_times, self._eps_ci_times])
        # NOTICE: to avoid injected bubble in delta and epsilon,
        #  the last_delta and last_epsilon must equal to delta and epsilon respectively,
        #  so, step1 and step2 can be simplified as following formula.
        delta_times, delta, co_unit_alpha = self._sch.split_buffer(
            co_group_i, factors=[self._delta_times, self._delta, co_para_unit_alpha]
        )
        epsilon_times, eps_ci_times, beta_unit = self._sch.split_buffer(
            ci, factors=[self._epsilon_times, self._eps_ci_times, beta]
        )
        kernel = self._sch.fuse_buffer(kd, kh, kw)
        shape_pre_transpose = [int(x) for x in self._sch.get_buffer_of(delta_times).shape]
        # transpose
        if self._cfg.tile_co:
            self._sch.reorder_buffer(
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
            self._sch.reorder_buffer(
                delta_times, epsilon_times, delta, eps_ci_times, kernel, co_unit_alpha, beta_unit
            )
        shape_post_transpose = [int(x) for x in self._sch.get_buffer_of(delta_times).shape]

        epsilon = self._sch.fuse_buffer(eps_ci_times, kernel)
        shape_post_layout = [int(x) for x in self._sch.get_buffer_of(epsilon).shape]
        new_weight_buffer = self._sch.get_buffer_of(epsilon)

        # specify relay transformation for arguments
        def relay_forward_ocdhw(x):
            if self._interleaved_weight:
                x = relay.layout_transform(x, self._kernel_layout, "OIDHW")
            x = relay.reshape(x, shape_post_split_co)
            if self._cfg.tile_co:
                x = relay.nn.pad(
                    x, [(0, 0), (0, 0), (0, add_co), (0, add_ci), (0, 0), (0, 0), (0, 0)]
                )
            else:
                x = relay.nn.pad(x, [(0, 0), (0, add_co), (0, add_ci), (0, 0), (0, 0), (0, 0)])
            x = relay.reshape(x, shape_pre_transpose)
            if self._cfg.tile_co:
                x = relay.transpose(x, [1, 0, 2, 5, 3, 6, 8, 4, 7])
            else:
                x = relay.transpose(x, [0, 1, 4, 2, 5, 7, 3, 6])
            x = relay.reshape(x, shape_post_layout)
            return x

        def relay_backward_ocdhw(x):
            x = relay.reshape(x, shape_post_transpose)
            if self._cfg.tile_co:
                x = relay.transpose(x, [1, 0, 2, 4, 7, 3, 5, 8, 6])
                x = relay.reshape(
                    x,
                    [
                        self._groups,
                        self._cfg.tile_co_num,
                        num_co_group_tile + add_co,
                        num_ci_group + add_ci,
                    ]
                    + kernel_size,
                )
                x = relay.strided_slice(
                    x,
                    [0, 0, 0, 0, 0, 0, 0],
                    [self._groups, self._cfg.tile_co_num, num_co_group_tile, num_ci_group]
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
                    [0, 0, 0, 0, 0, 0],
                    [self._groups, num_co_group_tile, num_ci_group] + kernel_size,
                    slice_mode="size",
                )
            x = relay.reshape(x, self._weight_shape)
            return x

        if self._relay_rewrite_mgr is not None:
            self._relay_rewrite_mgr.trace_update(
                origin_buf=origin_weight_buffer,
                new_buf=new_weight_buffer,
                forward_transform=relay_forward_ocdhw,
                backward_transform=relay_backward_ocdhw,
            )

    def tensorize_dma(self, dma_block, intrin_name, attrs):
        """Tensorize dma helper"""
        if intrin_name in ("nnp_wdma_load", "nnp_bdma_load"):
            axis_idx = 1 if self._cfg.tile_co else 0  # 0 is co_o if is tile_co
        else:
            axis_idx = 2 if self._cfg.tile_co else 1  # 0 is co_o and 1 is batch
        root_loop_sref = self._sch.get_loops(dma_block)[axis_idx]
        sref = self._sch.get_sref(dma_block)
        block_stmt = sref.stmt
        dtype = block_stmt.reads[0].buffer.dtype
        dummy_args = [dtype, 0, 0]
        attr_args = ["%s=%s" % (k, int(v)) for k, v in attrs.items()]
        self._sch.pragma(
            root_loop_sref,
            "nnp_dma_scope",
            tir.Call("", "tir.%s" % intrin_name, dummy_args + attr_args),
        )

    def tensorize_cube(self, cube_block, kernel_size):
        """Tensorize cube helper"""
        input_dtype_id, _ = EDGEX_DTYPE_INFO[self._input_dtype]
        weight_dtype_id, _ = EDGEX_DTYPE_INFO[self._weight_dtype]
        self.tensorize_dma(
            cube_block,
            "nnp_cube_compute",
            attrs={
                "num_group_cube": self._groups,
                "epsilon_cube": self._epsilon,
                "delta_cube": self._delta,
                "epsilon_times_cube": self._epsilon_times,
                "delta_times_cube": self._delta_times,
                "last_epsilon_cube": self._last_epsilon,
                "last_delta_cube": self._last_delta,
                # must set 16, when enable winograd
                "k_size_cube": kernel_size[0] * kernel_size[1] * kernel_size[2],
                "bias_value_cube": 0,  # valid when bias_mode=1
                "cube_work_num_cube": self._cube_enable,  # 0:enable cube0; 1:enable cube0/1;
                # 2:enable cube0/1/2
                "winograd_cube": 0,  # 0:disable winograd; 1:enable
                "sparsity_en_cube": self._sparsity_en,  # 0:disable sparsity; 1:enable
                "bias_en_cube": int(self._has_bias),  # 0:bias disable; 1:enable
                "data_type_cube": input_dtype_id,
                "weight_type_cube": weight_dtype_id,
                # 0:each co bias independent; 1:each co share bias
                "bias_mode_cube": self._bias_mode,
                "round_mode_cube": 4,  # 0:ceiling; 1:floor; 2:truncate; 3:rounding off; 4:rounding
                "delta_rewrite_nbbuf_cube": 1,  # NOTE: All rewrite flag need set 1, maybe iss bug.
                "dense_times_rewrite_ibuf_cube": 1,
                "epsilon_rewrite_ibuf_cube": 1,
                "epsilon_times_rewrite_wbuf_cube": 1,
                "delta_rewrite_wbuf_cube": 1,
            },
        )

    def tensorize_eidma(
        self,
        eidma,
    ):
        """Tensorize eidma block"""
        axes = self._sch.get_write_buffer_axes(eidma, 0)
        n, c, d, h, w = axes
        group, ci_group = self._sch.split_buffer(c, factors=[self._groups, None])
        if not self._interleaved_data:
            c1, c0 = self._sch.split_buffer(ci_group, factor=self._ci_para_lines)
            self._sch.reorder_buffer(n, group, c1, d, h, w, c0)

        if self._cfg.tile_co:
            loops = self._sch.get_loops(eidma)
            n, c, d, h, w = loops[1:6]
        else:
            loops = self._sch.get_loops(eidma)
            n, c, d, h, w = loops
        group, ci_group = self._sch.split(c, factors=[self._groups, None])
        if not self._interleaved_data:
            c1, c0 = self._sch.split(ci_group, factors=[None, self._ci_para_lines])
            self._sch.reorder(n, group, c1, d, h, w, c0)

        if self._input_C % self._ci_para_lines != 0:
            self._sch.loop_partition([c1])

        if self._cfg.tile_co:
            self._sch.pragma(self._sch.get_loops(eidma)[2], "nnp_dma_scope", "eidma")
        else:
            self._sch.pragma(self._sch.get_loops(eidma)[1], "nnp_dma_scope", "eidma")

        # eidma write dm buffer should align last dimension with 16/8
        self._sch.storage_align(
            eidma,
            0,
            len(self._sch.get_sref(eidma).stmt.writes[0].buffer.shape) - 2,
            self._ci_para_lines,
            0,
        )

    def tensorize_idma(
        self,
        idma,
        kernel_size,
        strides,
        dilation,
    ):
        """Tensorize idma block"""
        axes = self._sch.get_write_buffer_axes(idma, 0)
        n, c, d, h, w = axes
        group, ci_group = self._sch.split_buffer(c, factors=[self._groups, None])
        if not self._interleaved_data:
            c1, c0 = self._sch.split_buffer(ci_group, factor=self._ci_para_lines)
            self._sch.reorder_buffer(n, group, c1, d, h, w, c0)

        if self._cfg.tile_co:
            loops = self._sch.get_loops(idma)
            n, c, d, h, w = loops[1:6]
            loop_len = 2  # c_o, n
        else:
            loops = self._sch.get_loops(idma)
            n, c, d, h, w = loops
            loop_len = 1  # n
        group, ci_group = self._sch.split(c, factors=[self._groups, None])
        root_loop_sref = self._sch.get_loops(idma)[loop_len]
        if self._interleaved_data:
            # idma data layout [group, ic_group, depth, height, width, c0]
            if self._groups > 1:
                self._sch.pragma(root_loop_sref, "nnp_data_layout", "GCDHWc")
            else:
                # need double check
                self._sch.pragma(root_loop_sref, "nnp_data_layout", "CDHWc")
        else:
            # idma write dm buffer should align last dimension with 16/8
            self._sch.storage_align(
                idma,
                0,
                len(self._sch.get_sref(idma).stmt.writes[0].buffer.shape) - 2,
                self._ci_para_lines,
                0,
            )
            # idma data layout [group, ic_group, depth, height, width]
            self._sch.pragma(root_loop_sref, "nnp_data_layout", "GCDHW")

        dtype_id, _ = EDGEX_DTYPE_INFO[self._input_dtype]
        self.tensorize_dma(
            idma,
            "nnp_idma_load",
            attrs={
                "epsilon_idma": self._epsilon,
                "delta_idma": self._delta,
                "epsilon_times_idma": self._epsilon_times,
                "delta_times_idma": self._delta_times,
                "eps_ci_times_idma": self._eps_ci_times,
                "last_epsilon_idma": self._last_epsilon,
                "last_delta_idma": self._last_delta,
                "op_idma": 0,  # 0:conv, 1:matmul
                "para_mode_idma": self._para_mode,  # 0:tile para; 1:co para
                "wino_en_idma": 0,  # 0:disable winograd; 1: enable winograd
                "kernel1_speedup_flag_idma": 1 if kernel_size[2] == 1 else 0,  # enable if kw=1
                "sparsity_en_idma": self._sparsity_en,  # 0:disabel sparsity mode;
                # 1: enable sparsity mode
                "d_d_idma": dilation[0],
                "d_h_idma": dilation[1],
                "d_w_idma": dilation[2],
                "s_d_idma": strides[0],
                "s_h_idma": strides[1],
                "s_w_idma": strides[2],
                "k_d_idma": kernel_size[0],
                "k_h_idma": kernel_size[1],
                "k_w_idma": kernel_size[2],
                "epsilon_rewrite_ibuf_idma": 1,  # NOTE: All rewrite flag need set 1, maybe iss bug.
                "dense_times_rewrite_ibuf_idma": 1,
                "cube_enable_idma": self._cube_enable,  # 0:enable cube0; 1:enable cube0/1;
                # 2:enable cube0/1/2
                "data_type_idma": dtype_id,
                "pad_v_idma": 0,  # padding value
                "num_group_idma": self._groups,
                "num_ci_group_idma": self._input_C // self._groups,
                "pad_mode_idma": 0,  # padding mode, 0:constant padding(pad_v); 1:edge padding
                "insert_d0_idma": 0,  # deconv dense insert 0 num
                "insert_h0_idma": 0,
                "insert_w0_idma": 0,
                "B_T_idma": 0,  # need config when enable matmul
                "B_dim2_idma": 0,
                "B_dim1_idma": 0,
            },
        )

    def tensorize_ewdma_wdma(self, ewdma, wdma, kernel_size):
        """Tensorize ewdma and wdma"""
        # process ewdma
        if self._cfg.tile_co:
            self._sch.pragma(self._sch.get_loops(ewdma)[1], "nnp_dma_scope", "eidma")
        else:
            self._sch.pragma(self._sch.get_loops(ewdma)[0], "nnp_dma_scope", "eidma")
        # ewdma/wdma write dm buffer should align last dimension with 16/8
        self._sch.storage_align(
            ewdma,
            0,
            len(self._sch.get_sref(ewdma).stmt.writes[0].buffer.shape) - 2,
            self._ci_para_lines,
            0,
        )
        self._sch.storage_align(
            wdma,
            0,
            len(self._sch.get_sref(wdma).stmt.writes[0].buffer.shape) - 2,
            self._ci_para_lines,
            0,
        )
        # ewdma/wdma write dm buffer should align penultimate dimension with 16
        self._sch.storage_align(
            ewdma, 0, len(self._sch.get_sref(ewdma).stmt.writes[0].buffer.shape) - 3, 16, 0
        )
        self._sch.storage_align(
            wdma, 0, len(self._sch.get_sref(wdma).stmt.writes[0].buffer.shape) - 3, 16, 0
        )
        # tensorize wdma
        dtype_id, _ = EDGEX_DTYPE_INFO[self._weight_dtype]
        self.tensorize_dma(
            wdma,
            "nnp_wdma_load",
            attrs={
                "epsilon_wdma": self._epsilon,
                "delta_wdma": self._delta,
                "epsilon_times_wdma": self._epsilon_times,
                "delta_times_wdma": self._delta_times,
                "last_epsilon_wdma": self._last_epsilon,
                "last_delta_wdma": self._last_delta,
                "sparsity_en_wdma": self._sparsity_en,  # 0:disable sparsity mode; 1:enable
                "operation_wdma": 0,  # 0:conv; 1:matmul
                "data_type_wdma": dtype_id,
                "para_mode_wdma": self._para_mode,  # 0:tile para; 1:co para
                "cube_enable_wdma": self._cube_enable,  # 0:enable cube0; 1:enable cube0/1;
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
                "k_size_wdma": kernel_size[0] * kernel_size[1] * kernel_size[2],
                "bubble_insert_en_wdma": 1
                if self._input_dtype in ["int16"] and self._weight_dtype in ["int8", "uint8"]
                else 0,
            },
        )

    def tensorize_ebdma_bdma(self, ebdma, bdma):
        """Tensorize ebdma and bdma"""
        # process ebdma
        if ebdma is not None:
            if self._cfg.tile_co:
                self._sch.pragma(self._sch.get_loops(ebdma)[1], "nnp_dma_scope", "ewdma")
            else:
                self._sch.pragma(self._sch.get_loops(ebdma)[0], "nnp_dma_scope", "ewdma")
        # tensorize bdma
        if bdma is not None:
            # some constraints:
            # if delta_rewrite_nbbuf_bdma == 0, delta*4 <= BBUF_SIZE, which is 128
            # dense * zeta * delta * winograd? 16:1 <= OBUF_SIZE, which is 8*256*8
            self.tensorize_dma(
                bdma,
                "nnp_bdma_load",
                attrs={
                    "st_addr_sel": 0,
                    "addr_wrap_sel1": 0,
                    "addr_wrap_sel2": 0,
                    "num_group_bdma": self._groups,
                    "delta_times_bdma": self._delta_times,
                    "epsilon_times_bdma": self._epsilon_times,
                    "delta_bdma": self._delta,
                    "last_delta_bdma": self._last_delta,
                    "cube_work_num_bdma": self._cube_enable,
                    "parallel_mode_bdma": self._para_mode,
                    "bias_en_bdma": int(self._has_bias),
                    "norm_en_bdma": int(self._has_norm),
                    "delta_rewrite_nbbuf_bdma": 1,
                    "winograd_bdma": 0,
                    "bias_mode_bdma": self._bias_mode,
                    "norm_coeff_mode_bdma": self._norm_coeff_mode,
                },
            )

    def tensorize_odma(self, odma, strides):
        """Tensorize odma block"""
        axes = self._sch.get_write_buffer_axes(odma, 0)
        n, c, d, h, w = axes
        if self._groups > 1:
            group, co_group = self._sch.split_buffer(c, factors=[self._groups, None])
            if self._cfg.tile_co:
                co_group_o, co_group_i = self._sch.split_buffer(
                    co_group, nparts=self._cfg.tile_co_num
                )
                if self._interleaved_data:
                    self._sch.reorder_buffer(co_group_o, n, group, co_group_i, d, h, w, axes[-1])
                else:
                    c1, c0 = self._sch.split_buffer(co_group_i, factor=16)
                    self._sch.reorder_buffer(co_group_o, n, group, c1, d, h, w, c0)
            else:
                c1, c0 = self._sch.split_buffer(co_group, factor=16)
                self._sch.reorder_buffer(n, group, c1, d, h, w, c0)
        else:
            if self._cfg.tile_co:
                co_o, co_i = self._sch.split_buffer(c, nparts=self._cfg.tile_co_num)
                if self._interleaved_data:
                    self._sch.reorder_buffer(co_o, n, co_i, d, h, w, axes[-1])
                else:
                    c1, c0 = self._sch.split_buffer(co_i, factor=16)
                    self._sch.reorder_buffer(co_o, n, c1, d, h, w, c0)
            else:
                c1, c0 = self._sch.split_buffer(c, factor=16)
                self._sch.reorder_buffer(n, c1, d, h, w, c0)
        # odma write dm buffer should align last dimension with 16
        self._sch.storage_align(
            odma, 0, len(self._sch.get_sref(odma).stmt.writes[0].buffer.shape) - 2, 16, 0
        )
        if self._cfg.tile_co:
            loop_len = 2  # c_o, n
        else:
            loop_len = 1  # n
        # data_layout format contain "NCHW", "NCHWc", "NCDHW", "NCDHWc"
        root_loop_sref = self._sch.get_loops(odma)[loop_len]
        if self._interleaved_data:
            # need double check, the loop will be simplify if extent 1.
            if self._output_shape[2] > 1:
                self._sch.pragma(root_loop_sref, "nnp_data_layout", "CDHWc")
            else:
                self._sch.pragma(root_loop_sref, "nnp_data_layout", "CHWc")
        else:
            if self._output_shape[2] > 1:
                self._sch.pragma(root_loop_sref, "nnp_data_layout", "CDHW")
            else:
                self._sch.pragma(root_loop_sref, "nnp_data_layout", "CHW")

        dtype_id, _ = EDGEX_DTYPE_INFO[self._output_dtype]
        self.tensorize_dma(
            odma,
            "nnp_odma_store",
            attrs={
                "delta_odma": self._delta,
                "delta_times_odma": self._delta_times,
                "last_delta_odma": self._last_delta,
                "num_group_odma": self._groups,
                "addr_wrap_sel2_odma": 0,
                "addr_wrap_sel1_odma": 1,
                "extract_2to1_odma": 1 if strides[2] == 4 else 0,
                "int_type_odma": self._int_type,  # 1:output is int16
                "para_mode_odma": self._para_mode,  # 0:tile para; 1:co para
                "delta_mode_odma": 0,  # 0:each delta handle 16/32/48co; 1:each delta handle 16co
                "psum_out_en_odma": self._psum_out_en,  # 0:disable psum output; 1:enable.
                # psum_out_en must be 0, when winograd_en=1,
                "shiftnorm_odma": 0,
                "mulnorm_odma": 1,
                "norm_coeff_mode_odma": self._norm_coeff_mode,  # 0:read from NBUF; 1:fetch from isa
                "xbar_urr_weight_odma": 0,  # bus configuration, iss not simulate it.
                "relu_en_odma": int(self._has_relu),
                "round_mode_odma": 4,  # 0:ceiling; 1:floor; 2:truncate; 3:rounding off; 4:rounding
                "delta_rewrite_nbuf_odma": 1,
                "op_odma": 0,
                "wino_en_odma": 0,  # 0:disable winograd; 1:enable
                "cube_enable_odma": self._cube_enable,  # 0:enable cube0; 1:enable cube0/1;
                # 2:enable cube0/1/2
                "data_type_odma": dtype_id,
                "relu_mode_odma": self._relu_mode,  # 0:relu; 1:leaky relu
                "leaky_relu_mode_odma": 0,  # 0:relu coeff read from rbuf;
                # 1:relu coeff read from isa
                "bias_mode_odma": self._bias_mode,  # 0:bias read from bbuf; 1:bias featch from isa
                "bias_en_odma": 0,  # 0:disable bias add; 1:enable
                "relu_round_mode_odma": self._relu_round_mode,  # same as round_mode
                "relu_sftcoeff_odma": 0,  # relu quantize shift coefficient.
                "relu_mulcoeff_odma": 0x1,  # relu quantize mul coefficient.
                "start_state_mode_odma": 1,
                "ub_channel_odma": 0x10,
                "end_state_odma": 1,
                "wo_channel_odma": 0x10,
            },
        )

    def tensorize_eodma(self, eodma):
        """Tensorize eodma block"""
        if self._cfg.tile_co:
            loops = self._sch.get_loops(eodma)
            c = loops[2]
            co_group_tile = self._output_C // self._groups // self._cfg.tile_co_num
        else:
            loops = self._sch.get_loops(eodma)
            c = loops[1]
            co_group_tile = self._output_C // self._groups
        if self._groups > 1:
            _, co_group = self._sch.split(c, factors=[self._groups, None])
            if not self._interleaved_data:
                c1, _ = self._sch.split(co_group, factors=[None, 16])
        else:
            if not self._interleaved_data:
                c1, _ = self._sch.split(c, factors=[None, 16])

        # eodma write dm buffer should align c0 dimension with 16
        odma = get_producer_block(self._sch, eodma, 0)
        self._sch.storage_align(
            odma, 0, len(self._sch.get_sref(eodma).stmt.writes[0].buffer.shape) - 4, 16, 0
        )
        if co_group_tile % 16 != 0:
            self._sch.loop_partition([c1])

        if self._cfg.tile_co:
            self._sch.pragma(self._sch.get_loops(eodma)[2], "nnp_dma_scope", "eodma")
        else:
            self._sch.pragma(self._sch.get_loops(eodma)[1], "nnp_dma_scope", "eodma")

    def __conv3d_tensorize(
        self,
        Xdm,
        Wdm,
        Bdm,
        Xbuf,
        Wbuf,
        Bbuf,
        Ydm,
        Yddr,
        Compute,
    ):
        """Conv3d tensorize helper"""
        kernel_size = self._kernel_size
        strides = self._strides
        dilation = self._dilation
        if Xdm is not None:
            self.tensorize_eidma(Xdm)
        self.tensorize_idma(Xbuf, kernel_size, strides, dilation)
        self.tensorize_ewdma_wdma(Wdm, Wbuf, kernel_size)
        self.tensorize_ebdma_bdma(Bdm, Bbuf)
        self.tensorize_cube(Compute, kernel_size)
        self.tensorize_odma(Ydm, strides)
        if Yddr is not None:
            self.tensorize_eodma(Yddr)

    # todo(someone): process all conditions
    def __create_bdma_block(
        self,
        block,
        bias_param_buf,
        multiply_param_buf,
        shift_param_buf,
        relu_multiply_param_buf,
        relu_shift_param_buf,
    ):
        s = self._sch

        # sanity checks
        n_channel = self._weight_O
        assert relu_multiply_param_buf is None and relu_shift_param_buf is None
        has_bias = bias_param_buf is not None
        if has_bias:
            assert bias_param_buf.dtype == "int8"

        has_norm = multiply_param_buf is not None and shift_param_buf is not None
        if has_norm:
            assert multiply_param_buf.dtype == "int8"
            assert shift_param_buf.dtype == "int8"
        else:
            assert multiply_param_buf is None and shift_param_buf is None

        if has_bias:
            if has_norm:
                self._bias_mode = 0
                self._norm_coeff_mode = 0
                self._psum_out_en = 0
                relay_rewrite_per_channel_bias_and_norm(
                    s,
                    block,
                    bias_param_buf,
                    multiply_param_buf,
                    shift_param_buf,
                    n_channel,
                    self._relay_rewrite_mgr,
                )
            else:
                self._bias_mode = 0
                self._psum_out_en = 1
                relay_rewrite_per_channel_bias_only(
                    s, block, bias_param_buf, n_channel, self._relay_rewrite_mgr
                )
        else:
            raise NotImplementedError("not implemented")

        self._has_bias = has_bias
        self._has_norm = has_norm
        return s.cache_read(block, 1, "bbuf")

    def __merge_post_conv_ops(self, head: tir.schedule.BlockRV, channel_index):
        # match and rewrite bias/quantize parameters
        s = self._sch
        matcher = PostConvOpMatcher(s, channel_index=channel_index)

        conv_consumers = s.get_consumers(head)
        if len(conv_consumers) != 1:
            return head, None
        if not matcher.match(conv_consumers[0]):
            return head, None

        bias_param_buf = None
        multiply_param_buf = None
        shift_param_buf = None
        if matcher.bias_add_block is not None:
            (bias_param_buf,) = rewrite_param_to_dtype(
                s,
                head,
                matcher.bias_add_block,
                dtype="int8",
                is_reinterpret=True,
                relay_rewrite_mgr=self._relay_rewrite_mgr,
            )
        if matcher.quantize_multiply_block is not None:
            assert matcher.quantize_shift_block is not None
            (multiply_param_buf,) = rewrite_param_to_dtype(
                s,
                matcher.pre_quantize_block,
                matcher.quantize_multiply_block,
                dtype="int8",
                is_reinterpret=True,
                pre_cast_dtype="uint16",
                relay_rewrite_mgr=self._relay_rewrite_mgr,
            )
            (shift_param_buf,) = rewrite_param_to_dtype(
                s,
                matcher.quantize_multiply_block,
                matcher.quantize_shift_block,
                dtype="int8",
                is_reinterpret=False,
                relay_rewrite_mgr=self._relay_rewrite_mgr,
            )
        if matcher.relu_block is not None:
            self._has_relu = True
            self._relu_mode = 0

        # inline post cube ops
        last_sref = s.get_sref(matcher.last_block)
        cur = head
        while True:
            cur_sref = s.get_sref(cur)
            if cur_sref.same_as(last_sref):
                break
            next_block = s.get_consumers(cur)[0]
            s.compute_inline(cur)
            cur = next_block

        # stack post cube params
        Bbuf = self.__create_bdma_block(
            cur, bias_param_buf, multiply_param_buf, shift_param_buf, None, None
        )
        return cur, Bbuf

    def schedule(self):
        """Conv3d edgex schedule helper"""
        self._ci_para_lines = get_line_num(self._input_dtype)

        # inline padding into idma, assume there is always a padding block before conv
        Conv = self._conv_block
        Xpad = get_producer_block(self._sch, Conv, 1)  # read order: [out, in, weight]
        Xbuf = self._sch.cache_read(Conv, 1, "iobuf")
        self._sch.compute_inline(Xpad)

        # if original input is on DDR, create a cache read into DM
        Xdm = None
        if self._cfg.is_ddr_input:
            Xdm = self._sch.cache_read(Xbuf, 0, "dm")

        # merge post-cube operations, will modify the psum_out_en
        Y_iobuf = self._sch.cache_write(Conv, 0, "cube")
        Ydm, Bbuf = self.__merge_post_conv_ops(Y_iobuf, channel_index=1)
        Bdm = None if Bbuf is None else self._sch.cache_read(Bbuf, 0, "dm")

        # if output will write to ddr, create a cache write from DM
        last_Y_block = Ydm
        Yddr = None
        if self._cfg.is_ddr_output:
            Yddr = self._sch.cache_write(Ydm, 0, "dm")
            last_Y_block = Yddr
        else:
            buf = self._sch.get_sref(Ydm).stmt.writes[0].buffer
            tmp_buf = tvm.tir.decl_buffer(buf.shape, buf.dtype, buf.name, scope="dm")
            self._sch.replace_buffer(Ydm, buf, tmp_buf)

        # TODO(all): do not update config during processing, infer the tiles ahead before
        # actual schedule operations starts.
        self._odma_out_elem_bytes = get_conv_odma_output_bytes(
            self._psum_out_en, self._output_dtype, self._int_type
        )
        swift_tile_cfg(
            self._cfg,
            self._global_hw_cfg,
            self._output_shape,
            self._odma_out_elem_bytes,
            self._layout,
        )

        # read weight from ddr -> dm -> wbuf, refactor weight layouts
        self.__rewrite_conv3d_weight_layout_oidhw(Conv)
        Wdm = self._sch.cache_read(Conv, 2, "dm")
        Wbuf = self._sch.cache_read(Conv, 2, "wbuf")

        # schedule tiling
        if self._cfg.tile_co:
            # tile C dim for r"NCDHW(\d*c)?" layout
            loops = self._sch.get_loops(last_Y_block)
            no, co = loops[:2]
            co_o, co_i = self._sch.split(co, factors=[self._cfg.tile_co_num, None])
            self._sch.reorder(co_o, no, co_i)
            if Ydm != last_Y_block:  # output ddr exists
                self._sch.compute_at(Ydm, co_o, preserve_unit_loops=True)
            self._sch.compute_at(Conv, co_o, preserve_unit_loops=True)
            self._sch.compute_at(Xbuf, co_o, preserve_unit_loops=True)
            if Xdm is not None:  # input ddr exists
                self._sch.compute_at(Xdm, co_o, preserve_unit_loops=True)
            self._sch.compute_at(Wbuf, co_o, preserve_unit_loops=True)
            self._sch.compute_at(Wdm, co_o, preserve_unit_loops=True)
            if Bbuf:
                self._sch.compute_at(Bbuf, co_o, preserve_unit_loops=True)
                self._sch.compute_at(Bdm, co_o, preserve_unit_loops=True)
            self._sch.pragma(
                self._sch.get_loops(Ydm)[1],
                "nnp_num_co",
                self._output_C // self._cfg.tile_co_num,
            )
        else:
            self._sch.pragma(self._sch.get_loops(Ydm)[0], "nnp_num_co", self._output_C)

        # block tensorizations
        self.__conv3d_tensorize(Xdm, Wdm, Bdm, Xbuf, Wbuf, Bbuf, Ydm, Yddr, Conv)
        print(self._sch.mod["main"])


def schedule_edgex_conv_block(
    sched,
    conv,
    kernel_size,
    strides,
    padding,
    dilation,
    groups,
    layout="NCDHW",
    kernel_layout="OIDHW",
    relay_rewrite_mgr=None,
    cfg=None,
):
    """schedule edgex convolution on current schedule state"""
    if not (layout == "NCDHW" and kernel_layout == "OIDHW"):
        el.e(r"Only support 'NCDHW'-'OIHW' layout.")
    block_stmt = sched.get_sref(conv).stmt
    output_buffer = block_stmt.reads[0].buffer
    input_buffer = block_stmt.reads[1].buffer
    weight_buffer = block_stmt.reads[2].buffer
    output_shape = list(get_const_tuple(output_buffer.shape))
    input_shape = list(get_const_tuple(input_buffer.shape))
    weight_shape = list(get_const_tuple(weight_buffer.shape))

    # schedule use input shape before padding
    input_shape[2] -= padding[0] + padding[3]
    input_shape[3] -= padding[1] + padding[4]
    input_shape[4] -= padding[2] + padding[5]

    input_dtype = input_buffer.dtype
    weight_dtype = weight_buffer.dtype
    output_dtype = output_buffer.dtype
    scheduler = ScheduleConv3d(
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
        layout=layout,
        kernel_layout=kernel_layout,
        relay_rewrite_mgr=relay_rewrite_mgr,
        cfg=cfg,
    )
    scheduler.schedule()


def conv3d_tir_schedule(attrs, prim_func, tgt):
    """Conv3d edgex tir schedule"""
    kernel_size = attrs.get_int_tuple("kernel_size")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    padding = attrs.get_int_tuple("padding")
    padding = get_pad_tuple3d(padding, kernel_size)
    groups = attrs.groups
    layout = attrs.data_layout
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
        layout=layout,
        kernel_layout=kernel_layout,
        relay_rewrite_mgr=relay_rewrite_mgr,
    )
    return relay_rewrite_mgr.create_annotated_func()
