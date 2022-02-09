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
from tvm import te
from tvm import tir
from tvm.script import tir as T
from tvm.contrib.edgex.uniform_schedule.example_scheduler import ExampleScheduler
from tvm.contrib.edgex.uniform_schedule.graph import *
from tvm.contrib.edgex.uniform_schedule.cascade_scheduler_300t_compat import (
    CascadeScheduler300TCompat,
)
from tvm.contrib.edgex.uniform_schedule.scheduler_base import ScheduleContext


@T.prim_func
def scheduled_simple_multibranch_func(
    x0: T.Buffer[(16,), "int32"],
    T_add: T.Buffer[(16,), "int32"],
    T_add_1: T.Buffer[(16,), "int32"],
    T_add_2: T.Buffer[(16,), "int32"],
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    T_add_3 = T.alloc_buffer([16], dtype="int32")
    T_add_4 = T.alloc_buffer([16], dtype="int32")
    T_add_5 = T.alloc_buffer([16], dtype="int32")
    T_add_6 = T.alloc_buffer([16], dtype="int32")
    for i0_0, i0_1 in T.grid(2, 8):
        with T.block("T_add"):
            ax0 = T.axis.spatial(16, i0_0 * 8 + i0_1)
            T_add_3[ax0] = x0[ax0] + 1
    for i0_0, i0_1 in T.grid(2, 8):
        with T.block("T_add_1"):
            ax0 = T.axis.spatial(16, i0_0 * 8 + i0_1)
            T_add_4[ax0] = T_add_3[ax0] + 2
    for i0_0, i0_1 in T.grid(2, 8):
        with T.block("T_add_2"):
            ax0 = T.axis.spatial(16, i0_0 * 8 + i0_1)
            T_add[ax0] = T_add_4[ax0] + 3
    for i0_0 in T.serial(2):
        for ax0 in T.serial(8):
            with T.block("T_add_3"):
                ax0_1 = T.axis.spatial(16, i0_0 * 8 + ax0)
                T_add_5[ax0_1] = T_add_4[ax0_1] + 4
        for i0_1 in T.serial(8):
            with T.block("T_add_4"):
                ax0 = T.axis.spatial(16, i0_0 * 8 + i0_1)
                T_add_1[ax0] = T_add_5[ax0] + 5
    for i0_0 in T.serial(2):
        for ax0 in T.serial(8):
            with T.block("T_add_5"):
                ax0_2 = T.axis.spatial(16, i0_0 * 8 + ax0)
                T_add_6[ax0_2] = T_add_3[ax0_2] + 6
        for i0_1 in T.serial(8):
            with T.block("T_add_6"):
                ax0_3 = T.axis.spatial(16, i0_0 * 8 + i0_1)
                T_add_2[ax0_3] = T_add_6[ax0_3] + 7


def test_simple():
    x0 = te.placeholder([16], "int32", "x0")
    x1 = x0 + 1
    x2 = x1 + 2
    x3 = x2 + 3
    x4 = x2 + 4
    x5 = x4 + 5
    x6 = x1 + 6
    x7 = x6 + 7
    simple_multibranch_func = te.create_prim_func([x0, x3, x5, x7])
    s = tir.schedule.Schedule(simple_multibranch_func)
    ctx = ScheduleContext(s, None)
    scheduler = CascadeScheduler300TCompat(ctx)
    graph = scheduler.graph
    for subgraph in graph.data:
        subgraph.set_scheduler(ExampleScheduler(ctx))
    plans = scheduler.plan(graph)
    scheduler.execute(graph, plans[0])
    tvm.ir.structural_equal(s.mod["main"], scheduled_simple_multibranch_func)


if __name__ == "__main__":
    test_simple()
