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
"""The memcpy style edgex vu schedule"""

from functools import reduce
import numpy as np
import tvm
from tvm.contrib.edgex.tir.schedule import EdgexSchedule


def schedule_memcpy_style_edgex_impl(func, target):
    """memcpy style op edgex schedule implementation"""
    is_cpu = target.kind.name != "edgex"
    s = EdgexSchedule(func, debug_mode=False)
    main_blocks = s.get_child_blocks(s.get_block("root"))
    outer = None
    num_outer_loops = 0
    read_buf_extent = 1

    for main_block in main_blocks:
        main_block_sref = s.get_sref(main_block)
        block_stmt = main_block_sref.stmt
        read_bufs = [read_region.buffer for read_region in block_stmt.reads]

        dm_read_bytes = tvm.tir.IntImm("int32", 0)
        max_dm_bytes = 3 * 1024 * 1024
        for buffer in read_bufs:
            dm_read_bytes += reduce(
                lambda x, y: x * y, buffer.shape, tvm.DataType(buffer.dtype).bits // 8
            )

        for val in read_bufs[0].shape:
            if val > 1:
                read_buf_extent = val
                break

        # loop tiling if necessary
        if dm_read_bytes > max_dm_bytes:
            min_tiling_size = tvm.tir.indexdiv(dm_read_bytes + max_dm_bytes - 1, max_dm_bytes)
            loop_axes = list(s.get_loops(main_block))
            for idx, loop_rv in enumerate(loop_axes):
                loop_extent = s.get_sref(loop_rv).stmt.extent
                if loop_extent >= min_tiling_size:
                    # tiling_size is greatest common divisor of read_buf_extent and loop_extent.
                    # This is a workround for compute_at problem when schedule reshape op.
                    tiling_size = tvm.tir.IntImm(
                        "int32", np.gcd(int(read_buf_extent), int(loop_extent))
                    )
                    if tiling_size < min_tiling_size:
                        raise Exception("memcpy_style_schedule tiling size can not be found.")
                    outer, inner = s.split(loop_rv, factors=[tiling_size, None])
                    loop_axes.pop(idx)
                    loop_axes.insert(idx, inner)
                    loop_axes.insert(0, outer)
                    break
            s.reorder(*loop_axes)

    # loop_partition all the block's loop
    for blk in main_blocks:
        loops = s.get_loops(blk)
        for loop in loops:
            s.loop_partition(loop)

    # cache_read only the first block's read buf
    eidma_blocks = []
    stmt = s.get_sref(main_blocks[0]).stmt
    for i in range(len(stmt.reads)):
        eidma = s.cache_read(main_blocks[0], i, "dm")
        eidma_blocks.append(eidma)

    # compute_at cache_read block in inner axes.
    if outer:
        for block in eidma_blocks:
            s.compute_at(block, outer)
        num_outer_loops = 1

    # tensorize dma intrin
    if not is_cpu:
        for blk in eidma_blocks:
            s.pragma(s.get_loops(blk)[num_outer_loops], "nnp_dma_scope", "eidma")
        # tensorize memcpy block
        for blk in main_blocks:
            s.pragma(s.get_loops(blk)[num_outer_loops], "nnp_dma_scope", "eodma")

    return s.mod["main"]
