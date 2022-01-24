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
# pylint: disable=wildcard-import, arguments-differ
"""Scheduler base classes"""
from typing import Iterable, List, Tuple
from tvm.ir.expr import PrimExpr
from .graph import Edge, BlockGraph


class ScheduleContext:
    """Maintain schedule, relay rewriter and other global objects"""

    def __init__(self, sched, relay_rewrite_mgr):
        self.sched = sched
        self.relay_rewrite_mgr = relay_rewrite_mgr


class EdgeSpec:
    """Schedule information on subgraph's input/output edges"""

    # Expected full data shape to transform to
    target_shape: List[PrimExpr]

    # The storage scope the data should stay in
    storage_scope: str

    def __init__(self, edge: Edge):
        self.shape = edge.buffer().shape
        self.storage_scope = "global"


class SubgraphRequirement:
    """Requirements on input/output edges for specific subgraph plan"""

    # Input edges spec, one for each input edge
    input_specs: List[EdgeSpec]

    # Output edges spec, one for each output edge
    output_specs: List[EdgeSpec]

    def __init__(self, graph: BlockGraph):
        self.input_specs = [EdgeSpec(_) for _ in graph.inputs]
        self.output_specs = [EdgeSpec(_) for _ in graph.outputs]


class SubgraphPlan:
    """Base class of subgraph's schedule plan"""

    # Basic subgraph requirement
    requirement: SubgraphRequirement

    def __init__(self, subgraph: BlockGraph, requirement: SubgraphRequirement = None):
        if requirement is None:
            requirement = SubgraphRequirement(subgraph)
        self.requirement = requirement

    def get_input_spec(self, idx) -> EdgeSpec:
        """Get input edge requirement"""
        return self.requirement.input_specs[idx]

    def get_output_spec(self, idx) -> EdgeSpec:
        """Get output edge requirement"""
        return self.requirement.output_specs[idx]


class SubgraphScheduler:
    """Base class of schedule logic on specific subgraph types"""

    def __init__(self, ctx: ScheduleContext):
        self.ctx = ctx

    def plan(
        self, graph: BlockGraph, requirement: SubgraphRequirement = None
    ) -> Iterable[SubgraphPlan]:
        """Given subgraph and optional requirement, return candidate schedule plans"""
        raise NotImplementedError()

    def execute(self, graph: BlockGraph, plan: SubgraphPlan) -> None:
        """Given subgraph and schedule plan, execute the schedule"""
        raise NotImplementedError()


class UniformPlan(SubgraphPlan):

    """Subgraph schedule tasks that should be scheduled consecutively"""

    tasks: List[Tuple[BlockGraph, SubgraphPlan]]

    def __init__(self, graph: BlockGraph, tasks: List[Tuple[BlockGraph, SubgraphPlan]]):
        SubgraphPlan.__init__(self, graph)
        self.tasks = tasks
        for subgraph, subplan in self.tasks:
            for idx, edge in enumerate(subgraph.inputs):
                if not edge.parent:
                    continue
                edge_spec = subplan.get_input_spec(idx)
                global_idx = edge.parent.dst_idx
                self.get_input_spec(global_idx).shape = edge_spec.shape
                self.get_input_spec(global_idx).storage_scope = edge_spec.storage_scope
            for idx, edge in enumerate(subgraph.outputs):
                if not edge.parent:
                    continue
                edge_spec = subplan.get_output_spec(idx)
                global_idx = edge.parent.src_idx
                self.get_output_spec(global_idx).shape = edge_spec.shape
                self.get_output_spec(global_idx).storage_scope = edge_spec.storage_scope


class BaseUniformScheduler(SubgraphScheduler):
    """A helper stateful base class for uniform scheduler"""

    def __init__(self, ctx: ScheduleContext):
        self.graph = BlockGraph.create(ctx.sched)
        SubgraphScheduler.__init__(self, ctx)

    def plan_subgraphs(self) -> List[Tuple[BlockGraph, SubgraphPlan]]:
        """Plan interface for child classes, return sub-plans on optimal graph partition"""
        raise NotImplementedError()

    def plan(
        self, graph: BlockGraph, requirement: SubgraphRequirement = None
    ) -> Iterable[UniformPlan]:
        """Overrided plan"""
        assert graph == self.graph
        assert requirement is None
        tasks = self.plan_subgraphs()
        return [UniformPlan(graph, tasks)]

    def execute(self, plan: UniformPlan) -> None:
        """Overrided execution"""
        for subgraph, subplan in plan.tasks:
            scheduler = subgraph.scheduler
            assert scheduler is not None
            scheduler.execute(subplan)
        sched = self.ctx.sched
        relay_rewrite_mgr = self.ctx.relay_rewrite_mgr
        if relay_rewrite_mgr is not None:
            return relay_rewrite_mgr.create_annotated_func()
        return sched.mod["main"]
