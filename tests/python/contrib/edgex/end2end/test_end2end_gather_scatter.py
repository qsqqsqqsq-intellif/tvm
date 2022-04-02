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
import numpy as np
import tvm.testing
import tvm.script.tir as T
from tvm.contrib.edgex.tir.schedule import EdgexSchedule
from tvm.contrib.edgex.testing import check_edgex_tir_build
from tvm.contrib.edgex.tir.schedule import EdgexSchedule


def test_simple_scatter_1d():
    @T.prim_func
    def simple_scatter_1d(
        data: T.Buffer[(128,), "int32"],
        indices: T.Buffer[(16,), "int32"],
        updates: T.Buffer[(16,), "int32"],
        out: T.Buffer[(128,), "int32"],
    ) -> None:
        for i in range(indices.shape[0]):
            with T.block("compute"):
                vi = T.axis.remap("S", [i])
                if i == 0:
                    for j in range(128):
                        out[j] = data[j]
                out[indices[vi]] = updates[vi]

    def simple_schedule_scatter_1d(attrs, primfunc, target):
        s = EdgexSchedule(primfunc)
        blk = s.get_block("compute")
        for i in range(len(s.get_sref(blk).stmt.reads)):
            vidma = s.cache_read(blk, i, "vm")
            eidma = s.cache_read(vidma, 0, "dm")
            s.pragma(s.get_loops(vidma)[0], "nnp_dma_scope", "vidma")
            s.pragma(s.get_loops(eidma)[0], "nnp_dma_scope", "eidma")
        vodma = s.cache_write(blk, 0, "vm")
        eodma = s.cache_write(vodma, 0, "dm")
        s.pragma(s.get_loops(vodma)[0], "nnp_dma_scope", "vodma")
        s.pragma(s.get_loops(eodma)[0], "nnp_dma_scope", "eodma")
        vi = s.get_loops(blk)[0]
        s.vectorize(vi)
        s.pragma(vi, "nnp_scatter_store_scope", 1)
        return s.mod["main"]

    # execute test
    data = np.random.randint(-256, 256, [128]).astype("int32")
    indices = np.random.choice(np.arange(128), [16], replace=False).astype("int32")
    check_edgex_tir_build(
        "simple_scatter_1d",
        simple_scatter_1d,
        edgex_fschedule=simple_schedule_scatter_1d,
        output_idx=3,
        input_data=[data, indices, None, None],
    )


def test_simple_gather_1d():
    @T.prim_func
    def simple_gather_1d(
        data: T.Buffer[(128,), "int32"],
        indices: T.Buffer[(16,), "int32"],
        out: T.Buffer[(128,), "int32"],
    ) -> None:
        for i in range(indices.shape[0]):
            with T.block("compute"):
                vi = T.axis.remap("S", [i])
                out[vi] = data[indices[vi]]

    def simple_schedule_gather_1d(attrs, primfunc, tgt):
        s = EdgexSchedule(primfunc)
        blk = s.get_block("compute")
        for i in range(len(s.get_sref(blk).stmt.reads)):
            vidma = s.cache_read(blk, i, "vm")
            eidma = s.cache_read(vidma, 0, "dm")
            s.pragma(s.get_loops(vidma)[0], "nnp_dma_scope", "vidma")
            s.pragma(s.get_loops(eidma)[0], "nnp_dma_scope", "eidma")
        vodma = s.cache_write(blk, 0, "vm")
        eodma = s.cache_write(vodma, 0, "dm")
        s.pragma(s.get_loops(vodma)[0], "nnp_dma_scope", "vodma")
        s.pragma(s.get_loops(eodma)[0], "nnp_dma_scope", "eodma")
        vi = s.get_loops(blk)[0]
        s.vectorize(vi)
        s.pragma(vi, "nnp_gather_load_scope", 1)
        return s.mod["main"]

    # execute test
    indices = np.random.choice(np.arange(128), [16], replace=False).astype("int32")
    check_edgex_tir_build(
        "simple_gather_1d",
        simple_gather_1d,
        edgex_fschedule=simple_schedule_gather_1d,
        input_data=[None, indices, None],
    )


if __name__ == "__main__":
    test_simple_scatter_1d()
    test_simple_gather_1d()
