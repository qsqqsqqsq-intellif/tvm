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
# pylint: disable=cell-var-from-loop, bare-except
"""The naive edgex vu schedule strategy"""

from __future__ import annotations
from functools import reduce
import tvm
from tvm import tir
import tvm.testing
from tvm.contrib.edgex.tir.schedule import EdgexSchedule
from tvm.contrib.edgex.relay.transform import PostScheduleArgumentRewriteManager
from tvm.tir.expr import IntImm
from tvm.arith import Analyzer
from .utils import PostConvOpMatcher, rewrite_param_to_dtype


class NaiveVUScheduleError(Exception):
    def __init__(self, msg):
        super().__init__(self, "Naive vu schedule: %s" % msg)


def contains_var(expr, var):
    """whether stmt/expr contain var"""
    result = False

    def fvisit(obj):
        nonlocal result
        if obj.same_as(var):
            result = True

    tvm.tir.stmt_functor.post_order_visit(expr, fvisit)
    return result


def get_block_realize(s, block):
    """get block realize stmt helper"""
    sref = s.get_sref(block)
    parent = sref.parent.stmt
    if isinstance(parent, (tir.Block, tir.For)):
        return parent.body
    if isinstance(parent, tir.SeqStmt):
        return parent[sref.seq_index]
    return None


def is_reduce_block(s, block):
    """determine a block has reduction loop"""
    block_stmt = s.get_sref(block).stmt
    return any([v.iter_type == tir.IterVar.CommReduce for v in block_stmt.iter_vars])


def is_param_buffer(buffer):
    """naively determine small buffer"""
    return len([x for x in buffer.shape if x > 1]) <= 1


def inline_all_blocks(s: EdgexSchedule):
    """try inline all except anchor compute blocks"""

    def recursive_inline(block):
        consumers = s.get_consumers(block)
        if len(consumers) != 1:
            return
        if is_reduce_block(s, block):
            return
        for producer in s.get_producers(block):
            recursive_inline(producer)
        try:
            s.compute_inline(block)
        except:
            return

    def recursive_reverse_inline(block):
        producers = s.get_producers(block)
        if len(producers) != 1:
            return
        if is_reduce_block(s, block):
            return
        for consumer in s.get_consumers(block):
            recursive_reverse_inline(consumer)
        try:
            s.reverse_compute_inline(block)
        except:
            return

    for block in s.get_child_blocks(s.get_block("root")):
        recursive_inline(block)
    for block in reversed(s.get_child_blocks(s.get_block("root"))):
        recursive_reverse_inline(block)

    blocks = s.get_child_blocks(s.get_block("root"))
    return blocks


def cache_buffer_packing(
    s: EdgexSchedule,
    pack_vf,
    buffer_pack_dims,
    is_tiling,
    eidma_blocks,
    vidma_blocks,
    vodma_blocks,
    eodma_blocks,
):
    """packing all of the dm/vm buffers"""

    def pack_write_buffer(block, pack_dim, reorder_loop, is_write_buffer):
        if is_write_buffer:
            buffer_axes = list(s.get_write_buffer_axes(block, 0))
        else:
            buffer_axes = list(s.get_read_buffer_axes(block, 0))
        c_o, c_i = s.split_buffer(buffer_axes[pack_dim], factor=pack_vf)
        buffer_axes[pack_dim] = c_o
        buffer_axes.append(c_i)
        s.reorder_buffer(*buffer_axes)
        if reorder_loop:
            loop_axes = list(s.get_loops(block))
            loop_extent = s.get_sref(loop_axes[pack_dim]).stmt.extent
            c_o, c_i = s.split(loop_axes[pack_dim], factors=[None, pack_vf])
            if loop_extent % pack_vf != 0:
                # generally non-dividable packing axis create conditions
                s.loop_partition([c_o, c_i])
            loop_axes[pack_dim] = c_o
            loop_axes.append(c_i)
            s.reorder(*loop_axes)

    reorder_loop = not is_tiling
    for buffer in buffer_pack_dims:
        if is_param_buffer(buffer):
            continue
        pack_dim = buffer_pack_dims[buffer]
        if buffer in eidma_blocks:
            pack_write_buffer(eidma_blocks[buffer], pack_dim, reorder_loop, is_write_buffer=True)
            pack_write_buffer(vidma_blocks[buffer], pack_dim, reorder_loop, is_write_buffer=True)
        if buffer in vodma_blocks:
            pack_write_buffer(vodma_blocks[buffer], pack_dim, reorder_loop, is_write_buffer=False)
            pack_write_buffer(eodma_blocks[buffer], pack_dim, reorder_loop, is_write_buffer=False)


def tensorize_dma_intrinsics(
    s: EdgexSchedule,
    num_outer_loops,
    is_cpu,
    eidma_blocks,
    vidma_blocks,
    vodma_blocks,
    eodma_blocks,
):
    """tensorize all of the dma blocks"""
    if is_cpu:
        return

    def annotate_dma(buffer, block, dma_name):
        if is_param_buffer(buffer):
            s.pragma(s.get_loops(block)[0], "nnp_dma_scope", dma_name)
        else:
            s.pragma(s.get_loops(block)[num_outer_loops], "nnp_dma_scope", dma_name)

    for buffer in eidma_blocks:
        annotate_dma(buffer, eidma_blocks[buffer], "eidma")
    for buffer in vidma_blocks:
        annotate_dma(buffer, vidma_blocks[buffer], "vidma")
    for buffer in eodma_blocks:
        annotate_dma(buffer, eodma_blocks[buffer], "eodma")
    for buffer in vodma_blocks:
        annotate_dma(buffer, vodma_blocks[buffer], "vodma")


def rewrite_quantize_params_to_u8(s, relay_rewrite_mgr):
    """utility to find all quantize params in vu block and rewrite them to uint8"""
    visited = set()

    def __dfs(block):
        sref = s.get_sref(block)
        if sref in visited:
            return
        visited.add(sref)
        consumers = s.get_consumers(block)
        matcher = PostConvOpMatcher(s, channel_index=-1)

        # assume quantize subgraph startswith i64 cast
        if matcher.is_elemwise_cast(s.get_sref(block).stmt, "int64") and matcher.match(block):
            multiply_block = matcher.quantize_multiply_block
            shift_block = matcher.quantize_shift_block
            if shift_block is not None and multiply_block is not None:
                rewrite_param_to_dtype(
                    s,
                    matcher.pre_quantize_block,
                    matcher.quantize_multiply_block,
                    dtype="uint8",
                    is_reinterpret=False,
                    relay_rewrite_mgr=relay_rewrite_mgr,
                )
                rewrite_param_to_dtype(
                    s,
                    matcher.quantize_multiply_block,
                    matcher.quantize_shift_block,
                    dtype="uint8",
                    is_reinterpret=False,
                    relay_rewrite_mgr=relay_rewrite_mgr,
                )
                __dfs(matcher.last_block)
                return
        for consumer_block in consumers:
            __dfs(consumer_block)

    for block in s.get_child_blocks(s.get_block("root")):
        __dfs(block)


class NaiveVuScheduleConfig:
    """vu schedule configuration"""

    # whether input data need load from DDR
    is_ddr_input: bool = True

    # whether output data need store to DDR
    is_ddr_output: bool = True

    # is there a block can be compute_at or reverse_compute_at to the main_block
    has_reverse_compute_at: bool = False

    # cache read/write should be executed before tiling when tiling is not divisible.
    # This is a work around for cache write bug when tiling is not divisible.
    # TODO(@qingsiqi): remove this when cache write bug fixed.
    is_tiling_divisible: bool = True


class NaiveVuSchedule:
    """vu schedule class for edgex"""

    def __init__(
        self,
        prim_func: tir.PrimFunc,
        sch: tir.schedule.Schedule = None,
        is_cpu: bool = False,
        allow_multi_block: bool = False,
        relay_rewrite_mgr: PostScheduleArgumentRewriteManager = None,
        cfg: NaiveVuScheduleConfig = None,
    ) -> None:
        self._prim_func = prim_func
        self._cfg = NaiveVuScheduleConfig() if cfg is None else cfg
        self._sch = EdgexSchedule(self._prim_func, debug_mode=False) if sch is None else sch
        self._relay_rewrite_mgr = relay_rewrite_mgr
        self._is_cpu = is_cpu
        self._allow_multi_block = allow_multi_block
        self.analyze()

    def analyze_compute_at(self):
        """analyze if there is a block can be compute_at or reverse_compute_at to the main_block"""
        s = EdgexSchedule(self._prim_func, debug_mode=False)

        # Phase1: try inline all blocks
        main_blocks = inline_all_blocks(s)

        block_info = {}
        for block in main_blocks:
            block_info[s.get_sref(block)] = "normal"

        for main_block in main_blocks:
            main_block_sref = s.get_sref(main_block)
            block_stmt = main_block_sref.stmt
            write_bufs = {
                write_region.buffer: i for i, write_region in enumerate(block_stmt.writes)
            }
            read_bufs = {read_region.buffer: i for i, read_region in enumerate(block_stmt.reads)}

            if block_info[main_block_sref] != "compute_ated":
                # Phase2: determine packing dimensions and spatial dimensions
                axes = s.get_loops(main_block)
                axes_info, pack_axis, pack_vf, _, _ = self.analyse_compute_axes(
                    s, main_block, block_stmt, axes
                )

                # Phase3: tiling
                pack_vf, inner_axes, _ = self.naive_loop_tiling_and_packing(
                    s,
                    axes,
                    pack_axis,
                    pack_vf,
                    axes_info,
                    write_bufs,
                    read_bufs,
                )

                # Phase4: try reverse compute_at
                has_reverse_compute_at = False
                if len(s.get_consumers(main_block)) == 1:
                    if inner_axes:
                        compute_at_axes = inner_axes[-1]
                    else:
                        compute_at_axes = s.get_loops(main_block)[-1]

                    block = s.get_consumers(main_block)[0]
                    if not is_reduce_block(s, block):
                        try:
                            s.reverse_compute_at(block, compute_at_axes)
                            block_info[s.get_sref(block)] = "compute_ated"
                            has_reverse_compute_at = True
                        except:
                            pass

        return has_reverse_compute_at

    def analyze(self):
        """analyze vu prim_func to get tiling and packing information."""
        self._cfg.has_reverse_compute_at = self.analyze_compute_at()
        s = self._sch

        # Phase0: i64 quantize param rewrite
        rewrite_quantize_params_to_u8(s, self._relay_rewrite_mgr)

        # Phase1: try inline all blocks
        self._main_blocks = inline_all_blocks(s)
        if len(self._main_blocks) != 1 and not self._allow_multi_block:
            raise NaiveVUScheduleError("only support single block computation after inline")

        # Phase2: determine packing dimensions and spatial dimensions. Only do
        # analyse_compute_axes for the first main_block.
        main_block = self._main_blocks[0]
        block_stmt = s.get_sref(main_block).stmt
        self._axes = s.get_loops(main_block)
        (
            self._axes_info,
            self._pack_axis,
            self._pack_vf,
            self._buffer_pack_dims,
            self._cfg.is_tiling_divisible,
        ) = self.analyse_compute_axes(s, main_block, block_stmt, self._axes)
        self._need_pack = self._axes[-1] != self._pack_axis

    def cache_read_and_cache_write(
        self,
        s: EdgexSchedule,
        main_block,
        block_info,
        write_bufs,
    ):
        """set the local buffer'scope to "vm" and load/save to dm/vm"""

        eidma_blocks = {}
        vidma_blocks = {}
        eodma_blocks = {}
        vodma_blocks = {}

        block_stmt = s.get_sref(main_block).stmt
        # use repalce_buffer to set the local buffer'scope to "vm"
        consumer_blocks = s.get_consumers(main_block)
        if (
            len(consumer_blocks) == 1
            and s.get_sref(consumer_blocks[0]) in block_info
            and block_info[s.get_sref(consumer_blocks[0])] == "compute_ated"
        ):
            for i, write_region in enumerate(block_stmt.writes):
                buf = write_region.buffer
                consumer_block = consumer_blocks[0]
                if buf in [
                    read_region.buffer for read_region in s.get_sref(consumer_block).stmt.reads
                ]:
                    tmp_buf = tvm.tir.decl_buffer(buf.shape, buf.dtype, buf.name, scope="vm")
                    s.replace_buffer(main_block, buf, tmp_buf)

        for i in range(len(block_stmt.reads)):
            buf = s.get_buffer_of(s.get_read_buffer_axes(main_block, i)[0])
            if buf in write_bufs or buf.scope() == "vm":
                continue
            vidma = s.cache_read(main_block, i, "vm")
            vidma_blocks[buf] = vidma
            eidma = s.cache_read(vidma, 0, "dm")
            eidma_blocks[buf] = eidma
        for i in range(len(block_stmt.writes)):
            buf = s.get_buffer_of(s.get_write_buffer_axes(main_block, i)[0])
            if buf.scope() == "vm":
                continue
            vodma = s.cache_write(main_block, i, "vm")
            eodma = s.cache_write(vodma, 0, "dm")
            vodma_blocks[buf] = vodma
            eodma_blocks[buf] = eodma

        return eidma_blocks, vidma_blocks, eodma_blocks, vodma_blocks

    def analyse_compute_axes(
        self,
        s: EdgexSchedule,
        main_block: tir.schedule.BlockRV,
        block_stmt: tir.Block,
        axes: list[tir.schedule.LoopRV],
    ):
        """detect pack axis, spatial axes and etc"""
        axes_info = {}  # axis -> (type, extent)

        # collect Var -> LoopRV
        loop_rv_dict = {}
        loop_stmts = []
        analyzer = Analyzer()
        repl_dict = {}
        for loop_rv in axes:
            loop_stmt = s.get_sref(loop_rv).stmt
            loop_rv_dict[loop_stmt.loop_var] = loop_rv
            loop_stmts.append(loop_stmt)
            extent = loop_stmt.extent
            if not isinstance(extent, IntImm) or loop_stmt.min != 0:
                raise NaiveVUScheduleError("do not support dynamic loop")
            if extent == 1:
                repl_dict[loop_stmt.loop_var] = 0
                analyzer.bind(loop_stmt.loop_var, 0)
            axes_info[loop_rv] = ("normal", int(extent))

        # build loop var -> block var mapping
        block_realize = get_block_realize(s, main_block)
        n_vars = len(block_realize.iter_values)
        for i in range(n_vars):
            block_iter_var = block_stmt.iter_vars[i]
            binding = block_realize.iter_values[i]
            repl_dict[block_iter_var.var] = binding
            analyzer.bind(block_iter_var.var, binding)
            if block_iter_var.iter_type == tir.IterVar.CommReduce and binding in loop_rv_dict:
                loop_rv = loop_rv_dict[binding]
                axes_info[loop_rv] = ("reduce", axes_info[loop_rv][1])

        # pack dim detection
        buffer_pack_dim_candidates = []
        for i, loop_rv in enumerate(axes):
            valid = True
            buffer_pack_dim = {}
            loop_var = loop_stmts[i].loop_var

            def fvisit(obj):
                nonlocal valid
                if isinstance(obj, (tir.BufferLoad, tir.BufferStore)):
                    buffer = obj.buffer
                    for k, expr in enumerate(obj.indices):
                        expr = tvm.tir.stmt_functor.substitute(expr, repl_dict)
                        expr = analyzer.simplify(expr)
                        if expr.same_as(loop_var):
                            if buffer in buffer_pack_dim and buffer_pack_dim[buffer] != k:
                                valid = False  # A[k][0][k]
                            elif buffer.shape[k] < 8:
                                valid = False
                            else:
                                buffer_pack_dim[buffer] = k
                        elif isinstance(expr, tir.FloorMod) and expr.a.same_as(loop_var):
                            buffer_pack_dim[buffer] = k
                        elif isinstance(expr, tir.FloorDiv) and expr.a.same_as(loop_var):
                            continue
                        elif contains_var(expr, loop_var):
                            valid = False  # A[k+1]
                            p = axes_info[loop_rv]
                            if p[0] != "reduce":
                                axes_info[loop_rv] = ("spatial", p[1])

            tvm.tir.stmt_functor.post_order_visit(loop_stmts[i], fvisit)
            if valid and len(buffer_pack_dim) > 0:
                buffer_pack_dim_candidates.append((loop_rv, buffer_pack_dim))

        # select pack axis with innermost buffer dim, thus pure elemwise
        # op's pack dim is just the innermost one
        if len(buffer_pack_dim_candidates) == 0:
            raise NaiveVUScheduleError("no vectorizable axis detected")
        pack_info = min(
            buffer_pack_dim_candidates,
            key=lambda p: min([len(buffer.shape) - p[1][buffer] for buffer in p[1]]),
        )
        pack_vf = 64
        pack_axis, buffer_pack_dim = pack_info
        for buffer in buffer_pack_dim:
            nbytes = tvm.DataType(buffer.dtype).bits // 8
            pack_vf = min(pack_vf, 64 // nbytes)
            while pack_vf > buffer.shape[buffer_pack_dim[buffer]]:
                pack_vf = pack_vf // 2

        need_pack = axes[-1] != pack_axis
        is_tiling_divisible = True
        if need_pack:
            for loop_rv in axes:
                extent = axes_info[loop_rv][1]
                if loop_rv == pack_axis:
                    if extent % pack_vf != 0:
                        is_tiling_divisible = False
                    else:
                        is_tiling_divisible = True

        return axes_info, pack_axis, pack_vf, buffer_pack_dim, is_tiling_divisible

    def naive_loop_tiling_and_packing(
        self,
        s: EdgexSchedule,
        axes: list[tir.schedule.LoopRV],
        pack_axis: tir.schedule.LoopRV,
        pack_vf,
        axes_info,
        write_bufs,
        read_bufs,
    ):
        """do naive loop tiling, loop reorder and compute at dmas to inner loop"""
        # esitimate required vm bytes (since dm spatial > vm spatial), this is a naive estimation
        vm_read_bytes = tvm.tir.IntImm("int32", 0)
        vm_write_bytes = tvm.tir.IntImm("int32", 0)
        max_vm_bytes = 16 * 1024
        for buffer in read_bufs:
            if buffer in write_bufs:
                continue
            vm_read_bytes += reduce(
                lambda x, y: x * y, buffer.shape, tvm.DataType(buffer.dtype).bits // 8
            )
        for buffer in write_bufs:
            vm_write_bytes += reduce(
                lambda x, y: x * y, buffer.shape, tvm.DataType(buffer.dtype).bits // 8
            )
        if not isinstance(vm_read_bytes, tir.IntImm) or not isinstance(vm_write_bytes, tir.IntImm):
            raise NaiveVUScheduleError("do not support non-constant shaped buffers")
        required_bytes = max(vm_read_bytes.value, vm_write_bytes.value)

        outer_axes = []
        inner_axes = []
        # no need to do tiling case
        if required_bytes < max_vm_bytes:
            need_pack = axes[-1] != pack_axis
            if need_pack:
                reorder_axes = []
                c_i = None
                for loop_rv in axes:
                    typ, extent = axes_info[loop_rv]
                    if loop_rv == pack_axis:
                        c_o, c_i = s.split(loop_rv, factors=[None, pack_vf])
                        if extent % pack_vf != 0:
                            # generally non-dividable packing axis create conditions
                            s.loop_partition([c_o, c_i])
                        reorder_axes.append(c_o)
                        axes_info[c_o] = ("normal", (extent + pack_vf - 1) // pack_vf)
                        axes_info[c_i] = ("vectorize", pack_vf)
                        axes_info.pop(loop_rv)
                    else:
                        reorder_axes.append(loop_rv)
                reorder_axes.append(c_i)
                s.reorder(*reorder_axes)
            else:
                axes_info[pack_axis] = ("vectorize", axes_info[pack_axis][1])
            return pack_vf, inner_axes, outer_axes

        # esitimate how large outer iteration extents we need, this is a naive estimation
        cur_outer_factor = 1
        while pack_vf >= 8:
            spatial_factor = 8 if pack_vf == 16 else 4
            outer_factor_estimation = (required_bytes // max_vm_bytes + 1) // 2 * 2
            cur_outer_factor = 1
            for loop_rv in axes:
                typ, extent = axes_info[loop_rv]
                if loop_rv == pack_axis:
                    if axes[-1] == pack_axis:
                        pass
                    else:
                        cur_outer_factor *= extent // pack_vf
                elif extent <= spatial_factor:
                    pass
                elif typ == "spatial":
                    cur_outer_factor *= extent // spatial_factor
                elif typ == "reduce":
                    pass
                else:
                    cur_outer_factor *= extent
            if cur_outer_factor >= outer_factor_estimation:
                break
            pack_vf = pack_vf // 2

        if cur_outer_factor < outer_factor_estimation:
            raise NaiveVUScheduleError("can not find valid tiling")

        c_i = None
        cur_outer_factor = 1
        for loop_rv in axes:
            typ, extent = axes_info[loop_rv]
            if loop_rv == pack_axis:
                if axes[-1] == pack_axis:
                    axes_info[pack_axis] = ("vectorize", extent)
                    c_i = pack_axis
                elif cur_outer_factor >= outer_factor_estimation:
                    c_i = pack_axis
                else:
                    c_o, c_i = s.split(loop_rv, factors=[None, pack_vf])
                    outer_axes.append(c_o)
                    axes_info.pop(loop_rv)
                    axes_info[c_o] = ("normal", (extent + pack_vf - 1) // pack_vf)
                    axes_info[c_i] = ("vectorize", pack_vf)
                    cur_outer_factor *= extent // pack_vf
                    if extent % pack_vf != 0:
                        # generally non-dividable packing axis create conditions
                        s.loop_partition([c_o, c_i])
            elif extent <= spatial_factor or cur_outer_factor >= outer_factor_estimation:
                inner_axes.append(loop_rv)
            elif typ == "spatial":
                h_o, h_i = s.split(loop_rv, factors=[None, spatial_factor])
                outer_axes.append(h_o)
                inner_axes.append(h_i)
                axes_info.pop(loop_rv)
                axes_info[h_o] = ("spatial", (extent + spatial_factor - 1) // spatial_factor)
                axes_info[h_i] = ("spatial", spatial_factor)
                cur_outer_factor *= extent // spatial_factor
            elif typ == "reduce":
                inner_axes.append(loop_rv)
            else:
                outer_axes.append(loop_rv)
                cur_outer_factor *= extent
        inner_axes.append(c_i)
        s.reorder(*(outer_axes + inner_axes))

        return pack_vf, inner_axes, outer_axes

    def schedule(self):
        """A naive strategy to schedule vu computation with single anchor op"""

        s = self._sch
        is_tiling_divisible = self._cfg.is_tiling_divisible

        block_info = {}
        for block in self._main_blocks:
            block_info[s.get_sref(block)] = "normal"

        for main_block in self._main_blocks:
            main_block_sref = s.get_sref(main_block)
            block_stmt = main_block_sref.stmt
            write_bufs = {
                write_region.buffer: i for i, write_region in enumerate(block_stmt.writes)
            }
            read_bufs = {read_region.buffer: i for i, read_region in enumerate(block_stmt.reads)}

            # block already compute_at to other block don't do tiling and vectorize
            if block_info[main_block_sref] != "compute_ated":
                if len(block_stmt.writes) != 1:
                    raise NaiveVUScheduleError("only support single output")

                # Phase1: determine packing dimensions and spatial dimensions.
                # main_blocks[0] have already done analyse_compute_axes in self.analyse()
                if main_block != self._main_blocks[0]:
                    self._axes = s.get_loops(main_block)
                    (
                        self._axes_info,
                        self._pack_axis,
                        self._pack_vf,
                        self._buffer_pack_dims,
                        is_tiling_divisible,
                    ) = self.analyse_compute_axes(s, main_block, block_stmt, self._axes)
                    self._need_pack = self._axes[-1] != self._pack_axis

                # Phase2: load/save to dm/vm before loop_tiling when no compute_at
                # and tiling not divisible
                if not self._cfg.has_reverse_compute_at and not is_tiling_divisible:
                    (
                        eidma_blocks,
                        vidma_blocks,
                        eodma_blocks,
                        vodma_blocks,
                    ) = self.cache_read_and_cache_write(s, main_block, block_info, write_bufs)

                # Phase3: tiling
                self._pack_vf, inner_axes, outer_axes = self.naive_loop_tiling_and_packing(
                    s,
                    self._axes,
                    self._pack_axis,
                    self._pack_vf,
                    self._axes_info,
                    write_bufs,
                    read_bufs,
                )
                self._axes = None

                # Phase4: vector computation scheduling
                for loop_rv, (typ, extent) in self._axes_info.items():
                    if typ == "vectorize" and extent <= 64:
                        s.vectorize(loop_rv)
                    elif typ == "vectorize" and extent > 64:
                        v_outer, v_inner = s.split(loop_rv, factors=[None, 64])
                        if extent % 64 != 0:
                            s.loop_partition([v_inner, v_outer])
                        s.vectorize(v_inner)
                    elif typ == "reduce":
                        if extent < 5:
                            s.unroll(loop_rv)  # unroll small reduce loop
                    elif typ == "spatial":
                        # generally spatial loop contains conditions
                        if not self._is_cpu:  # partitioned loops compile slow under cpu llvm
                            s.loop_partition(loop_rv)

            # Phase5: try reverse_compute_at the consumer block to curent block at inner_axis[-1]
            # only support one consumer block
            if self._cfg.has_reverse_compute_at and len(s.get_consumers(main_block)) == 1:
                if inner_axes:
                    compute_at_axes = inner_axes[-1]
                else:
                    compute_at_axes = s.get_loops(main_block)[-1]

                block = s.get_consumers(main_block)[0]
                if not is_reduce_block(s, block):
                    try:
                        s.reverse_compute_at(block, compute_at_axes)
                        block_info[s.get_sref(block)] = "compute_ated"
                    except:
                        pass

            # Phase6: load/save to dm/vm
            if self._cfg.has_reverse_compute_at or is_tiling_divisible:
                (
                    eidma_blocks,
                    vidma_blocks,
                    eodma_blocks,
                    vodma_blocks,
                ) = self.cache_read_and_cache_write(s, main_block, block_info, write_bufs)

            # Phase7 load/save cache in inner computation
            if outer_axes:
                for buffer in vidma_blocks:
                    if is_param_buffer(buffer):
                        continue
                    s.compute_at(vidma_blocks[buffer], outer_axes[-1])
                for buffer in eidma_blocks:
                    if is_param_buffer(buffer):
                        continue
                    s.compute_at(eidma_blocks[buffer], outer_axes[-1])
                for buffer in vodma_blocks:
                    if is_param_buffer(buffer):
                        continue
                    s.reverse_compute_at(vodma_blocks[buffer], outer_axes[-1])
                for buffer in eodma_blocks:
                    if is_param_buffer(buffer):
                        continue
                    s.reverse_compute_at(eodma_blocks[buffer], outer_axes[-1])

            # Phase8: rewrite cache buffer to packed format
            if block_info[main_block_sref] != "compute_ated" and self._need_pack:
                cache_buffer_packing(
                    s,
                    self._pack_vf,
                    self._buffer_pack_dims,
                    is_tiling=len(outer_axes) > 0,
                    eidma_blocks=eidma_blocks,
                    vidma_blocks=vidma_blocks,
                    vodma_blocks=vodma_blocks,
                    eodma_blocks=eodma_blocks,
                )

            # Phase9: tensorize dma intrinsics
            tensorize_dma_intrinsics(
                s,
                len(outer_axes),
                is_cpu=self._is_cpu,
                eidma_blocks=eidma_blocks,
                vidma_blocks=vidma_blocks,
                vodma_blocks=vodma_blocks,
                eodma_blocks=eodma_blocks,
            )

        if self._relay_rewrite_mgr:
            return self._relay_rewrite_mgr.create_annotated_func()
        return s.mod["main"]


def naive_vu_schedule(prim_func, is_cpu=False, allow_multi_block=False, enable_relay_rewrite=False):
    sch = EdgexSchedule(prim_func)
    relay_rewrite_mgr = (
        PostScheduleArgumentRewriteManager(sch) if enable_relay_rewrite is not None else None
    )
    cfg = NaiveVuScheduleConfig()
    scheduled_func = NaiveVuSchedule(
        prim_func, sch, is_cpu, allow_multi_block, relay_rewrite_mgr, cfg
    ).schedule()
    return scheduled_func
