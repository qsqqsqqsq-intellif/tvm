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
"""The concatenate edgex vu schedule"""

from tvm.contrib.edgex.tir.schedule import EdgexSchedule


def schedule_concatenate_edgex_impl(func, is_cpu):
    """concatenate edgex schedule implementation"""
    s = EdgexSchedule(func, debug_mode=False)

    concat = s.get_child_blocks(s.get_block("root"))[0]

    _, c, _, _ = s.get_loops(concat)
    s.loop_partition(c)

    eidma_blocks = []

    concat_stmt = s.get_sref(concat).stmt

    for i in range(len(concat_stmt.reads)):
        eidma = s.cache_read(concat, i, "dm")
        eidma_blocks.append(eidma)

    # tensorize dma intrin
    if not is_cpu:
        for blk in eidma_blocks:
            s.pragma(s.get_loops(blk)[0], "nnp_dma_scope", "eidma")
        # tensorize concat block
        s.pragma(s.get_loops(concat)[0], "nnp_dma_scope", "eodma")

    return s.mod["main"]
