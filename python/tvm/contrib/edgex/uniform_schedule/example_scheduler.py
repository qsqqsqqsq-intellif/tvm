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
"""Example scheduler with simple tiling logic"""
from .graph import *
from .scheduler_base import *


class ExampleScheduler(SubgraphScheduler, SubgraphCascadeSupport):
    """This is an example scheduler with simple cascade tiling support"""

    class ExampleSchedulePlan(SubgraphPlan):
        """This is a simple example plan"""

        def __init__(self, subgraph: BlockGraph):
            super().__init__(subgraph)
            self.do_cascade_tiling = False

    def plan(
        self, subgraph: BlockGraph, request: SubgraphRequirement = None
    ) -> Iterable[SubgraphPlan]:
        """Overrided plan interface"""
        root_tile_info = request.root_tile_info
        if root_tile_info is None:
            plan = ExampleScheduler.ExampleSchedulePlan(subgraph)
            response = plan.requirement
            self.set_shape_and_scope(subgraph, request, response)
            return [plan]

        is_root = root_tile_info.root_spec is None
        if is_root:
            return self.plan_for_tiling_root(subgraph, request)
        return self.plan_for_tiling_child(subgraph, request)

    def set_shape_and_scope(
        self, subgraph: BlockGraph, request: SubgraphRequirement, response: SubgraphRequirement
    ):
        """Fill in required storage scope and shape of inputs/outputs"""
        for i, spec in enumerate(response.input_specs):
            buffer = subgraph.inputs[i].buffer()
            req = request.input_specs[i]
            spec.storage_scope = req.storage_scope if req.storage_scope else "global"
            spec.target_shape = req.target_shape if req.target_shape else buffer.shape
        for i, spec in enumerate(response.output_specs):
            buffer = subgraph.outputs[i].buffer()
            req = request.output_specs[i]
            spec.storage_scope = req.storage_scope if req.storage_scope else "global"
            spec.target_shape = req.target_shape if req.target_shape else buffer.shape

    def plan_for_tiling_root(
        self, subgraph: BlockGraph, request: SubgraphRequirement
    ) -> List[ExampleSchedulePlan]:
        """Plan for root subgraph of cascade tiling chain, should decide tiling sizes"""

        # now assume buffer axes and loops take the same order
        root_buffer = subgraph.outputs[0].buffer()
        buffer_axis_to_loop_idx = [idx for idx, _ in enumerate(root_buffer.shape)]

        # now assume loop order is outer0, outer1, ..., inner0, inner1, ...
        loop_positions = []
        outer_pos = []
        outer_loops = 0
        for size in root_buffer.shape:
            if size <= 1:
                outer_pos.append(-1)
            else:
                outer_pos.append(outer_loops)
                outer_loops += 1
        for idx, _ in enumerate(root_buffer.shape):
            loop_positions.append((outer_pos[idx], outer_loops + idx))

        # sample different candidate tile sizes
        plan_candidates = []
        for n in [8, 16, 32]:
            plan = ExampleScheduler.ExampleSchedulePlan(subgraph)
            response = plan.requirement
            output_tilespecs = []
            tile_shape = [n for _ in range(len(root_buffer.shape))]
            root_spec = CascadeTileSpec.from_buffer(
                root_buffer, is_symbolic=False, tile_shape=tile_shape
            )
            root_tile_info = RootTilingInfo(
                subgraph.block_info.get_block_rv(subgraph.outputs[0].writer()),
                root_spec,
                loop_positions=loop_positions,
                buffer_axis_to_loop_idx=buffer_axis_to_loop_idx,
            )
            for edge in subgraph.outputs:
                buffer = edge.buffer()
                if buffer.same_as(root_buffer):
                    output_tilespecs.append(root_spec)
                else:
                    output_tilespecs.append(
                        CascadeTileSpec.from_buffer(buffer, tile_shape=buffer.shape)
                    )
            input_tilespecs = self.infer_tile_specs(subgraph, output_tilespecs, root_spec)
            if input_tilespecs:
                plan.do_cascade_tiling = True
                response.root_tile_info = root_tile_info
                for i, spec in enumerate(response.input_specs):
                    spec.tile_spec = input_tilespecs[i]
                for i, spec in enumerate(response.output_specs):
                    spec.tile_spec = output_tilespecs[i]
            self.set_shape_and_scope(subgraph, request, response)
            plan_candidates.append(plan)
        return plan_candidates

    def plan_for_tiling_child(
        self, subgraph: BlockGraph, request: SubgraphRequirement
    ) -> List[ExampleSchedulePlan]:
        """Plan for child subgraph of cascade tiling chain,
        the tiling configuration is determined by root"""
        plan = ExampleScheduler.ExampleSchedulePlan(subgraph)
        response = plan.requirement
        root_tilespec = request.root_tile_info.root_spec
        output_tilespecs = []
        for i, spec in enumerate(request.output_specs):
            if spec.tile_spec is not None:
                output_tilespecs.append(spec.tile_spec)
            else:
                buffer = subgraph.outputs[i].buffer()
                output_tilespecs.append(
                    CascadeTileSpec.from_buffer(buffer, tile_shape=buffer.shape)
                )
        input_tilespecs = self.infer_tile_specs(subgraph, output_tilespecs, root_tilespec)
        if input_tilespecs:
            plan.do_cascade_tiling = True
            response.root_tile_info = request.root_tile_info
            for i, spec in enumerate(response.input_specs):
                spec.tile_spec = input_tilespecs[i]
            for i, spec in enumerate(response.output_specs):
                spec.tile_spec = output_tilespecs[i]
        self.set_shape_and_scope(subgraph, request, response)
        return [plan]

    def execute(self, subgraph: BlockGraph, plan: SubgraphPlan) -> None:
        """Overrided execute interface"""
        assert isinstance(plan, ExampleScheduler.ExampleSchedulePlan)
        s = self.ctx.sched

        # do nothing for non-tiling case
        if not plan.do_cascade_tiling:
            return
        root_tile_info = plan.get_root_tileinfo()
        root_sref = s.get_sref(root_tile_info.root_block_rv)

        for sref in subgraph.get_blocks_in_reverse_topo_order():
            block = subgraph.block_info.get_block_rv(sref)
            if sref.same_as(root_sref):
                self.execute_for_tiling_root(s, block, root_tile_info)
            else:
                self.execute_for_tiling_child(s, block, root_tile_info)

    def execute_for_tiling_root(self, s, block: BlockRV, root_tile_info: RootTilingInfo):
        """Execute for root subgraph of cascade tiling chain"""
        loops = s.get_loops(block)
        out_buffer = s.get_sref(block).stmt.writes[0].buffer
        loop_positions = root_tile_info.loop_positions
        tile_shape = root_tile_info.root_spec.tile_shape
        ordered_loops = [None for _ in range(2 * len(out_buffer.shape))]
        for i in range(len(out_buffer.shape)):
            loop_idx = root_tile_info.buffer_axis_to_loop_idx[i]
            if loop_idx < 0:
                continue
            outer_pos, inner_pos = loop_positions[i]
            if outer_pos >= 0:
                outer, inner = s.split(loops[loop_idx], factors=[None, tile_shape[i]])
                ordered_loops[outer_pos] = outer
                ordered_loops[inner_pos] = inner
            else:
                ordered_loops[inner_pos] = loops[loop_idx]
        ordered_loops = [_ for _ in ordered_loops if _ is not None]
        s.reorder(*ordered_loops)

    def execute_for_tiling_child(self, s, block: BlockRV, root_tile_info: RootTilingInfo):
        """Execute for child subgraph of cascade tiling chain"""
        loop_positions = root_tile_info.loop_positions
        innermost_outer_loop_pos = max([_[0] for _ in loop_positions])
        assert innermost_outer_loop_pos >= 0
        loops = s.get_loops(root_tile_info.root_block_rv)
        s.compute_at(block, loops[innermost_outer_loop_pos])
