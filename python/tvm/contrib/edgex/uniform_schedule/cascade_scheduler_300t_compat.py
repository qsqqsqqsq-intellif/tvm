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
# pylint: disable=unused-wildcard-import,wildcard-import,arguments-differ
"""A scheduler follow NNP300T strategy"""
from typing import Dict, Iterable, List, Set, Tuple
from .scheduler_base import *
from .graph import BlockGraph, Edge


def update_plan_request(
    subgraph: BlockGraph, edge_dict: Dict[Edge, EdgeSpec], request: SubgraphRequirement
):
    """Helper to create plan requests for subgraph"""
    for i, edge in enumerate(subgraph.inputs):
        if edge in edge_dict:
            request.input_specs[i] = edge_dict[edge]
        elif edge.src is None:
            request.input_specs[i].target_shape = edge.buffer().shape
            request.input_specs[i].storage_scope = "global"
    for i, edge in enumerate(subgraph.outputs):
        if edge in edge_dict:
            request.output_specs[i] = edge_dict[edge]
        elif edge.dst is None:
            request.output_specs[i].target_shape = edge.buffer().shape
            request.output_specs[i].storage_scope = "global"


def dfs_plan_subgraphs(
    cur_idx: int,
    graph_seq: List[BlockGraph],
    cur_plans: List[SubgraphPlan],
    edge_dict: Dict[Edge, EdgeSpec],
    result_handler,
    enable_cascade_tiling=False,
):
    """Helper to plan all subgraphs, assuming all edge on the same buffer share the same spec"""

    def update_spec(edge: Edge, spec: EdgeSpec):
        if edge in edge_dict:
            cur_spec = edge_dict[edge]
            assert cur_spec.same_as(
                spec
            ), f"Incompatible requirement on edge #{edge.id} {edge.buffer()}"
            return False
        edge_dict[edge] = spec
        if edge.parent:
            edge_dict[edge.parent] = spec
        return True

    if cur_idx == len(graph_seq):
        # stop recursion
        if result_handler:
            result_handler(graph_seq, cur_plans)
        return

    subgraph = graph_seq[cur_idx]
    scheduler: SubgraphScheduler = subgraph.scheduler
    assert scheduler, "Scheduler is not set for subgraph"
    request = SubgraphRequirement(subgraph)
    update_plan_request(subgraph, edge_dict, request)
    if enable_cascade_tiling:
        if cur_idx == 0:
            request.root_tile_info = RootTilingInfo(None, None, None, None)
        else:
            request.root_tile_info = cur_plans[-1].requirement.root_tile_info

    subplans = scheduler.plan(subgraph, request)
    for plan in subplans:
        updated = []
        input_specs = plan.requirement.input_specs
        output_specs = plan.requirement.output_specs
        for i, spec in enumerate(input_specs):
            edge = subgraph.inputs[i]
            assert spec.is_fully_constrainted(), f"Missing response spec for #{edge.id}"
            if update_spec(edge, spec):
                updated.append(edge)
        for i, spec in enumerate(output_specs):
            edge = subgraph.outputs[i]
            assert spec.is_fully_constrainted(), f"Missing response spec for #{edge.id}"
            if update_spec(edge, spec):
                updated.append(edge)
        cur_plans.append(plan)
        dfs_plan_subgraphs(
            cur_idx + 1,
            graph_seq,
            cur_plans,
            edge_dict,
            result_handler,
            enable_cascade_tiling=enable_cascade_tiling,
        )
        cur_plans.pop()
        for edge in updated:
            edge_dict.pop(edge)


class SubChainScheduler(SubgraphScheduler):
    """Scheduler for single chain's tiling"""

    class SubChainSchedulerPlan(SubgraphPlan):
        """Plan type for the chain scheduler, it is just sequence of plan of chained subgraphs"""

        def __init__(self, chain: BlockGraph, subplans):
            SubgraphPlan.__init__(self, chain)
            assert not chain.is_single_block()
            self.subplans = subplans

    def plan(
        self, chain: BlockGraph, request: SubgraphRequirement = None
    ) -> Iterable[SubgraphPlan]:
        """Plan overrides"""
        assert (
            not chain.is_single_block()
        ), "SubChainScheduler should be used with non-single block graph"
        if request is None:
            request = SubgraphRequirement(chain)
        edge_specs = {}
        for i, spec in enumerate(request.input_specs):
            if spec.is_fully_constrainted():
                edge_specs[chain.inputs[i]] = spec
                edge_specs[chain.inputs[i].dst_child] = spec
        for i, spec in enumerate(request.output_specs):
            if spec.is_fully_constrainted():
                edge_specs[chain.outputs[i]] = spec
                edge_specs[chain.outputs[i].src_child] = spec

        enable_cascade_tiling = False
        if len(chain.data[-1].outputs_by_buffer) == 1:
            enable_cascade_tiling = True

        all_plans = []

        def result_handler(_, reversed_plan_seq):
            plan_seq = list(reversed(reversed_plan_seq))
            chain_plan = SubChainScheduler.SubChainSchedulerPlan(chain, plan_seq)
            update_plan_request(chain, edge_specs, chain_plan.requirement)
            all_plans.append(chain_plan)

        reverse_seq = list(reversed(chain.data))
        dfs_plan_subgraphs(
            0,
            reverse_seq,
            [],
            edge_specs,
            result_handler,
            enable_cascade_tiling=enable_cascade_tiling,
        )
        return all_plans

    def execute(self, chain: BlockGraph, plan: SubgraphPlan) -> None:
        """Execute overrides"""
        assert (
            not chain.is_single_block()
        ), "SubChainScheduler should be used with non-single block graph"
        assert isinstance(plan, SubChainScheduler.SubChainSchedulerPlan)
        assert len(plan.subplans) == len(chain.data)
        for i in reversed(range(len(chain.data))):
            subplan = plan.subplans[i]
            subgraph = chain.data[i]
            assert subgraph.scheduler, "Scheduler is not set for subgraph"
            subgraph.scheduler.execute(subgraph, subplan)


class CascadeScheduler300TCompat(BaseUniformScheduler):
    """DM chain decomposition scheduler via 300T strategy"""

    def merge_atomic_patterns(self):
        # TODO(bxq): implement atomic patterns match fusion stitch
        pass

    def decompose_cascade_chains(self):
        """Chain decomposition logic to simulate part of strategies of 300T schedule,
        We stop the chain when
        (1) no subsequent consumers
        (2) multiple outputs or multiple consumers for current output
        (3) consumers graph can not fuse with producer via patterns
        (4) consumer is already chained with other graph
        """

        def __dfs(cur: List[BlockGraph], results: List[List[BlockGraph]], visited: Set[BlockGraph]):
            subgraph = cur[-1]

            # multi-out case
            if len(subgraph.outputs) != 1:
                results.append(list(cur))
                for out_edge in subgraph.outputs:
                    dst = out_edge.dst
                    if not dst:
                        continue
                    if dst in visited:
                        continue
                    visited.add(dst)
                    __dfs([dst], results, visited)
                return

            out_edge = subgraph.outputs[0]
            dst = out_edge.dst
            if dst is None or dst in visited or not self.is_chainable(out_edge):
                # no subsequent consumers or already chained or incompatible
                results.append(list(cur))
                return
            visited.add(dst)
            cur.append(dst)
            __dfs(cur, results, visited)

        partitioned_chains = []
        visited = set()
        for subgraph in self.graph.get_start_subgraphs():
            if subgraph in visited:
                continue
            visited.add(subgraph)
            __dfs([subgraph], partitioned_chains, visited)

        # merge chains
        for chain in partitioned_chains:
            merged = BlockGraph.merge(*chain)
            merged.set_scheduler(SubChainScheduler(self.ctx))

    def is_chainable(self, edge: Edge) -> bool:
        """Determine whether two consecutive subgraph can be chained"""
        # pylint: disable=unidiomatic-typecheck
        return type(edge.src.scheduler) == type(edge.dst.scheduler)

    def estimate_cost(self, _: List[SubgraphPlan]):
        """Get cost estimation for combination of subgraph plans"""
        # TODO(bxq): add cost model
        return 0

    def plan_subgraphs(self) -> List[Tuple[BlockGraph, SubgraphPlan]]:
        """Plan entrance"""
        self.merge_atomic_patterns()
        self.decompose_cascade_chains()

        best_cost = None
        best_subplans = None

        def result_handler(_, plan_seq):
            nonlocal best_cost, best_subplans
            cost = self.estimate_cost(plan_seq)
            if cost is None:
                return
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_subplans = list(plan_seq)

        dfs_plan_subgraphs(0, self.graph.data, [], {}, result_handler)
        return list(zip(self.graph.data, best_subplans))
