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
# pylint: disable=wildcard-import, arguments-differ, unused-wildcard-import
"""Scheduler base classes"""
import functools
from typing import Iterable, List, Tuple
import tvm
from tvm import tir
from tvm.tir.schedule import BlockRV
from tvm.ir.expr import PrimExpr
from tvm.tir.schedule.schedule import Schedule
from .graph import Edge, BlockGraph
from .analysis import *


class ScheduleContext:
    """Maintain schedule, relay rewriter and other global objects"""

    def __init__(self, sched: Schedule, relay_rewrite_mgr):
        self.sched = sched
        self.relay_rewrite_mgr = relay_rewrite_mgr


class CascadeTileSpec:
    """Represent cascade tiling info for a block-wise data dependency"""

    # the coresponding buffer
    buffer: tir.Buffer

    # tile relations
    relations: List[TileRelation]

    # tile sizes
    tile_shape: List[PrimExpr]

    # number of total tiles
    tile_count: PrimExpr

    # total bytes accessed for the tiles, calculate lazily
    total_access_bytes: PrimExpr = None

    def __init__(self, buffer, relations, tile_shape, tile_count):
        self.buffer = buffer
        self.relations = relations
        self.tile_shape = tile_shape
        self.tile_count = tile_count

    def __str__(self):
        res = (
            "CascadeTileSpec(\n"
            + f"    buffer={self.buffer.name}{self.buffer.shape},\n"
            + f"    tile_shape={self.tile_shape},\n"
            + f"    tile_cnt={self.tile_count},\n"
            + f"    relations=[\n"
        )
        for rel in self.relations:
            res += f"      {rel.lower_bound_coeffs}, {rel.upper_bound_coeffs};\n"
        res += "    ]\n)"
        return res

    def equals(self, other: "CascadeTileSpec") -> bool:
        """Structural equality"""
        if self == other:
            return True
        return (
            self.buffer.same_as(other.buffer)
            and all(
                [
                    x == y or tvm.ir.structural_equal(x, y)
                    for x, y in zip(self.relations, other.relations)
                ]
            )
            and all(
                [
                    x == y or tvm.ir.structural_equal(x, y)
                    for x, y in zip(self.tile_shape, other.tile_shape)
                ]
            )
        )

    def merge(self, other: "CascadeTileSpec") -> "CascadeTileSpec":
        """Create a new merged spec to cover different tile specs"""
        if other is None:
            return self
        if self.equals(other):
            return self
        return self

    def estimate_access_bytes(self):
        if self.total_access_bytes is not None:
            return self.total_access_bytes
        elem_bytes = tvm.DataType(self.buffer.dtype).bits // 8
        tile_volume = functools.reduce(lambda x, y: x * y, self.tile_shape)
        self.total_access_bytes = tile_volume * self.tile_count * elem_bytes
        return self.total_access_bytes

    def estimate_tile_bytes(self):
        elem_bytes = tvm.DataType(self.buffer.dtype).bits // 8
        tile_volume = functools.reduce(lambda x, y: x * y, self.tile_shape)
        return tile_volume * elem_bytes

    @staticmethod
    def from_buffer(buffer: tir.Buffer, is_symbolic=True, tile_shape=None):
        """Create root spec from buffer

        Parameters
        ----------
        buffer: tir.Buffer
            buffer object

        is_symbolic: bool
            whether use var as tile size placeholders

        tile_shape: List[PrimExpr]
            tile size per axis
        """
        analyzer = tvm.arith.Analyzer()
        ndim = len(buffer.shape)
        tile_relations = get_root_tile_relations(buffer)
        if is_symbolic:
            assert tile_shape is None
            tile_shape = [tir.Var(f"x{k}", "int32") for k in range(ndim)]
        elif not tile_shape:
            tile_shape = buffer.shape
        tile_count = functools.reduce(
            lambda x, y: x * y,
            [(buffer.shape[i] - 1 + tile_shape[i]) // tile_shape[i] for i in range(ndim)],
        )
        return CascadeTileSpec(buffer, tile_relations, tile_shape, analyzer.simplify(tile_count))

    @staticmethod
    def estimate_tile_count(tile_shape: List[PrimExpr], root_spec: "CascadeTileSpec") -> PrimExpr:
        """Get total count of tiles estimation"""
        root_ndim = len(root_spec.tile_shape)
        minimum_depth = root_ndim

        def __fvisit(obj):
            nonlocal minimum_depth
            for i, root_size in enumerate(root_spec.tile_shape):
                if obj.same_as(root_size):
                    minimum_depth = min(minimum_depth, i)

        for expr in tile_shape:
            tvm.tir.stmt_functor.post_order_visit(expr, __fvisit)
        cnt = 1
        for k in range(minimum_depth, root_ndim):
            cnt *= (
                root_spec.buffer.shape[k] + root_spec.tile_shape[k] - 1
            ) // root_spec.tile_shape[k]
        return cnt


class RootTilingInfo:
    """Tiling information for root block to compute at"""

    # root block to compute at
    root_block_rv: BlockRV

    # root tiling spec object
    root_spec: CascadeTileSpec

    # loop_positions[i] = loop position of ith axis of output buffer:
    # (outer_pos, inner_pos), outer_pos=-1 denotes no tiling
    loop_positions: List[Tuple[int, int]]

    # original loop index for ith axis of output buffer
    buffer_axis_to_loop_idx: List[int]

    def __init__(
        self,
        root_block_rv: tir.Block,
        root_spec: CascadeTileSpec,
        loop_positions,
        buffer_axis_to_loop_idx,
    ):
        self.root_block_rv = root_block_rv
        self.root_spec = root_spec
        self.loop_positions = loop_positions
        self.buffer_axis_to_loop_idx = buffer_axis_to_loop_idx


class EdgeSpec:
    """Schedule information on subgraph's input/output edges"""

    # Expected full data shape to transform to, None denotes there is no constraint
    target_shape: List[PrimExpr]

    # The storage scope the data should stay in, None denotes there is no constraint
    storage_scope: str

    # Cascade tiling specification
    tile_spec: CascadeTileSpec

    def __init__(self, edge: Edge = None):
        if edge is None:
            self.target_shape = None
            self.storage_scope = None
        else:
            self.target_shape = edge.buffer().shape
            self.storage_scope = edge.buffer().scope()
        self.tile_spec = None

    def is_fully_constrainted(self) -> bool:
        """Whether the specification is fully constrainted"""
        return self.target_shape is not None and self.storage_scope is not None

    def same_as(self, other: "EdgeSpec") -> bool:
        """Compare two specs"""
        return self.storage_scope == other.storage_scope and (
            self.target_shape == other.target_shape
            or (
                len(self.target_shape) == len(other.target_shape)
                and all(
                    [
                        tvm.ir.structural_equal(x, y)
                        for x, y in zip(self.target_shape, other.target_shape)
                    ]
                )
            )
        )


class SubgraphRequirement:
    """Requirements on input/output edges for specific subgraph plan"""

    # Input edges spec, one for each input edge
    input_specs: List[EdgeSpec]

    # Output edges spec, one for each output edge
    output_specs: List[EdgeSpec]

    # Tiling information for the cascade root, default to None
    root_tile_info: RootTilingInfo

    def __init__(self, graph: BlockGraph):
        self.input_specs = [EdgeSpec() for _ in graph.inputs]
        self.output_specs = [EdgeSpec() for _ in graph.outputs]
        self.root_tile_info = None


class SubgraphPlan:
    """Base class of subgraph's schedule plan"""

    # Basic subgraph requirement
    requirement: SubgraphRequirement

    def __init__(self, subgraph: BlockGraph):
        self.requirement = SubgraphRequirement(subgraph)

    def get_input_spec(self, idx) -> EdgeSpec:
        """Get input edge requirement"""
        return self.requirement.input_specs[idx]

    def get_output_spec(self, idx) -> EdgeSpec:
        """Get output edge requirement"""
        return self.requirement.output_specs[idx]

    def get_root_tileinfo(self) -> RootTilingInfo:
        """Get tiling info for the cascade root"""
        return self.requirement.root_tile_info


class SubgraphScheduler:
    """Base class of schedule logic on specific subgraph types"""

    def __init__(self, ctx: ScheduleContext):
        self.ctx = ctx

    def plan(
        self, graph: BlockGraph, request: SubgraphRequirement = None
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
                self.get_input_spec(global_idx).target_shape = edge_spec.target_shape
                self.get_input_spec(global_idx).storage_scope = edge_spec.storage_scope
            for idx, edge in enumerate(subgraph.outputs):
                if not edge.parent:
                    continue
                edge_spec = subplan.get_output_spec(idx)
                global_idx = edge.parent.src_idx
                self.get_output_spec(global_idx).target_shape = edge_spec.target_shape
                self.get_output_spec(global_idx).storage_scope = edge_spec.storage_scope


class BaseUniformScheduler(SubgraphScheduler):
    """A helper stateful base class for uniform scheduler"""

    def __init__(self, ctx: ScheduleContext):
        self.graph = BlockGraph.create(ctx.sched)
        SubgraphScheduler.__init__(self, ctx)

    def plan_subgraphs(self) -> List[Tuple[BlockGraph, SubgraphPlan]]:
        """Plan interface for child classes, return sub-plans on optimal graph partition"""
        raise NotImplementedError()

    def plan(self, graph: BlockGraph, request: SubgraphRequirement = None) -> Iterable[UniformPlan]:
        """Overrided plan"""
        assert graph == self.graph, "BaseUniformScheduler should use it's owned graph"
        assert request is None
        tasks = self.plan_subgraphs()
        return [UniformPlan(graph, tasks)]

    def execute(self, graph: BlockGraph, plan: UniformPlan) -> None:
        """Overrided execution"""
        assert graph == self.graph
        for subgraph, subplan in plan.tasks:
            scheduler = subgraph.scheduler
            assert scheduler is not None
            scheduler.execute(subgraph, subplan)


class SubgraphCascadeSupport:
    """Trait for subgraph scheduler to support cascade tiling schedule"""

    def infer_tile_specs(
        self, subgraph: BlockGraph, output_specs: List[CascadeTileSpec], root_spec: CascadeTileSpec
    ) -> List[CascadeTileSpec]:
        """Infer input tile specs from output tile specs and root spec"""
        return SubgraphCascadeSupport.infer_tile_specs_default(subgraph, output_specs, root_spec)

    @staticmethod
    def infer_tile_specs_default(
        subgraph: BlockGraph, output_specs: List[CascadeTileSpec], root_spec: CascadeTileSpec
    ) -> List[CascadeTileSpec]:
        """Fallback support for `infer_tile_specs`"""
        per_buffer_specs: Dict[tir.Buffer, CascadeTileSpec] = {}  # buffer -> tile spec
        for i, spec in enumerate(output_specs):
            buffer = subgraph.outputs[i].buffer()
            if buffer in per_buffer_specs:
                per_buffer_specs[buffer] = per_buffer_specs[buffer].merge(spec)
            else:
                per_buffer_specs[buffer] = spec

        read_cnt = {}  # buffer -> reads times count
        for block in subgraph.get_blocks_in_reverse_topo_order():
            block_stmt = block.stmt
            if len(block_stmt.writes) != 1:
                return None

            tile_relations = get_tile_relations(subgraph.block_info.sched, block)
            if len(tile_relations) == 0:
                return None

            write_buffer = block_stmt.writes[0].buffer
            write_spec = per_buffer_specs[write_buffer]
            for read_buffer in tile_relations:
                if read_buffer.same_as(write_buffer):
                    continue
                cur_relation = tile_relations[read_buffer]
                read_relations = compose_tile_relations(cur_relation, write_spec.relations)
                read_tile_shape = estimate_tile_shape(
                    read_buffer, read_relations, root_spec.tile_shape
                )
                read_tile_count = CascadeTileSpec.estimate_tile_count(read_tile_shape, root_spec)
                input_spec = CascadeTileSpec(
                    read_buffer, read_relations, read_tile_shape, read_tile_count
                )
                if read_buffer in per_buffer_specs:
                    input_spec = per_buffer_specs[read_buffer].merge(input_spec)
                per_buffer_specs[read_buffer] = input_spec
                read_cnt[read_buffer] = read_cnt.get(read_buffer, 0) + 1

        input_specs = []
        for edge in subgraph.inputs:
            input_specs.append(per_buffer_specs[edge.buffer()])
        return input_specs
