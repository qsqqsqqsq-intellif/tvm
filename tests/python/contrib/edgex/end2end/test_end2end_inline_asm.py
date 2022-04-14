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
import pytest
import sys
import tvm.script.tir as T
from tvm.contrib.edgex.tir.schedule import EdgexSchedule
from tvm.contrib.edgex.testing import check_edgex_tir_build


def do_simple_schedule(s: EdgexSchedule):
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


def test_vu_scalar_add_with_asm():
    @T.prim_func
    def add(
        X: T.Buffer[(16,), "int32"], Y: T.Buffer[(16,), "int32"], Z: T.Buffer[(16,), "int32"]
    ) -> None:
        for i in range(16):
            with T.block("compute"):
                vi = T.axis.spatial(16, i)
                Z[vi] = T.nnp_inline_asm_vcu(
                    "={vv},{vv},{vv}",
                    "nop.10\nvadd.s32 $0 $1 $2\nnop.10\n",
                    0,  # no vectorize factor
                    0,  # no state regs
                    2,  # two inputs
                    X[i],
                    Y[i],
                    0,  # no extra placeholders
                    dtype="int32",
                )

    s = EdgexSchedule(add)
    do_simple_schedule(s)
    check_edgex_tir_build(
        "vu_scalar_add_with_asm", s.mod["main"], numpy_func=lambda x, y: x + y, check_cpu=False
    )


def test_vu_add_with_asm():
    @T.prim_func
    def add(
        X: T.Buffer[(128,), "int32"], Y: T.Buffer[(128,), "int32"], Z: T.Buffer[(128,), "int32"]
    ) -> None:
        for i in range(2):
            with T.block("compute"):
                T.reads([X[i * 64 : i * 64 + 64], Y[i * 64 : i * 64 + 64]])
                T.writes([Z[i * 64 : i * 64 + 64]])
                vi = T.axis.spatial(2, i)
                Z[T.ramp(vi * 64, 1, 64)] = T.nnp_inline_asm_vcu(
                    "={vv},{vv},{vv}",
                    "nop.10\nvadd.s32 $0 $1 $2\nnop.10\n",
                    8,  # vectorize factor = 8
                    0,  # no state regs
                    2,  # two inputs
                    X[T.ramp(vi * 64, 1, 64)],
                    Y[T.ramp(vi * 64, 1, 64)],
                    1,  # one useless placeholder
                    Z.data,  # useless placeholder
                    dtype="int32x64",
                )

    s = EdgexSchedule(add)
    do_simple_schedule(s)
    check_edgex_tir_build(
        "vu_add_with_asm", s.mod["main"], numpy_func=lambda x, y: x + y, check_cpu=False
    )


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
