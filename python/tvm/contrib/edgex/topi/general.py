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

from tvm.contrib.edgex.tir.schedule import EdgexSchedule


def schedule_memcpy_style_edgex_impl(func, target):
    """memcpy style op edgex schedule implementation"""
    is_cpu = target.kind.name != "edgex"
    s = EdgexSchedule(func, debug_mode=False)

    main_blocks = s.get_child_blocks(s.get_block("root"))
    eidma_blocks = []

    # loop_partition all the block's loop, may not be the best choise
    for blk in main_blocks:
        loops = s.get_loops(blk)
        for loop in loops:
            s.loop_partition(loop)

    # cache_read only the first block's read buf
    stmt = s.get_sref(main_blocks[0]).stmt
    for i in range(len(stmt.reads)):
        eidma = s.cache_read(main_blocks[0], i, "dm")
        eidma_blocks.append(eidma)

    # tensorize dma intrin
    if not is_cpu:
        for blk in eidma_blocks:
            s.pragma(s.get_loops(blk)[0], "nnp_dma_scope", "eidma")
        # tensorize memcpy block
        for blk in main_blocks:
            s.pragma(s.get_loops(blk)[0], "nnp_dma_scope", "eodma")

    return s.mod["main"]
