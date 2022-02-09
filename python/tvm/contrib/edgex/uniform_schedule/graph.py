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
"""Graph data structure of uniform schedule"""
import itertools
import functools
from typing import Dict, Iterable, List, Union, Set
from tvm.tir.buffer import Buffer
from tvm.tir.schedule.block_scope import StmtSRef
from tvm.tir.schedule.schedule import BlockRV, Schedule


class BlockEdge:
    """Static edge information related to block read/write behaviors"""

    # Writer block for the edge
    writer: StmtSRef

    # Write buffer index in the writer block
    write_indices: List[int]

    # Reader block for the edge
    reader: StmtSRef

    # Read buffer index in the reader block
    read_indices: List[int]

    # Buffer for the edge
    buffer: Buffer

    def __init__(self, writer, write_indices, reader, read_indices):
        assert writer is not None or reader is not None
        self.writer = writer
        self.write_indices = write_indices
        self.reader = reader
        self.read_indices = read_indices
        if writer:
            self.buffer = writer.stmt.writes[write_indices[0]].buffer
        else:
            self.buffer = reader.stmt.reads[read_indices[0]].buffer


class Edge:
    """Edge class connecting subgraphs"""

    # Unique identifier
    id: int  # pylint: disable=invalid-name

    # Subgraph of the source
    src: "BlockGraph" = None

    # Source subgraph input idx
    src_idx: int = None

    # Subgraph of the dest
    dst: "BlockGraph" = None

    # Dest subgraph output idx
    dst_idx: int = None

    # Parent edge in parent graph
    parent: "Edge" = None

    # Child edge in src subgraph
    src_child: "Edge" = None

    # Child edge in dst subgraph
    dst_child: "Edge" = None

    # Block level static information
    block_edge: BlockEdge

    def __init__(self, edge_id, block_edge):
        """Constructor"""
        self.id = edge_id  # pylint: disable=invalid-name
        self.block_edge = block_edge

    def parent_graph(self) -> "BlockGraph":
        """Get parent graph"""
        return self.src.parent if self.src else self.dst.parent

    def writer(self) -> StmtSRef:
        """Get write block for the edge"""
        return self.block_edge.writer

    def reader(self) -> StmtSRef:
        """Get read block for the edge"""
        return self.block_edge.reader

    def buffer(self) -> Buffer:
        """Get buffer for the edge"""
        return self.block_edge.buffer


class BlockGraphInfo:
    """Store the static information for blocks in the graph"""

    # Counter for new edges
    edge_count: int = 0

    # Block sref to the block index
    block_idx: Dict[StmtSRef, int]

    # Block index to a block RV object
    block_idx_to_rv: List[BlockRV]

    # Block index to a block sref object
    block_idx_to_sref: List[StmtSRef]

    def __init__(self, s: Schedule):
        """Constructor"""
        self.sched = s
        self.block_idx_to_rv = s.get_child_blocks(s.get_block("root"))
        self.block_idx_to_sref = []
        self.block_idx = dict()
        for idx, block_rv in enumerate(self.block_idx_to_rv):
            sref = s.get_sref(block_rv)
            self.block_idx_to_sref.append(sref)
            self.block_idx[sref] = idx

    def get_block_idx(self, block: Union[StmtSRef, BlockRV]) -> int:
        """Query the index of the block"""
        if isinstance(block, BlockRV):
            block = self.sched.get_sref(block)
        if block not in self.block_idx:
            raise ValueError(f"The block is not in the current schedule: \n{block.stmt}")
        return self.block_idx[block]

    def block_num(self) -> int:
        """Query num of blocks"""
        return len(self.block_idx_to_sref)

    def get_block_sref(self, idx: Union[int, BlockRV, StmtSRef]) -> StmtSRef:
        """Fetch the sref for kth block"""
        if isinstance(idx, StmtSRef):
            return idx
        if isinstance(idx, BlockRV):
            return self.sched.get_sref(idx)
        return self.block_idx_to_sref[idx]

    def get_block_rv(self, idx: Union[int, BlockRV, StmtSRef]) -> BlockRV:
        """Fetch the rv for kth block"""
        if isinstance(idx, BlockRV):
            return idx
        if isinstance(idx, StmtSRef):
            idx = self.get_block_idx(idx)
        return self.block_idx_to_rv[idx]

    def get_producers(self, block: Union[BlockRV, StmtSRef]) -> List[StmtSRef]:
        """Wrapper for srefs on s.get_producers()"""
        block_rv = self.get_block_rv(block)
        return [self.get_block_sref(_) for _ in self.sched.get_producers(block_rv)]

    def get_consumers(self, block: Union[BlockRV, StmtSRef]) -> List[StmtSRef]:
        """Wrapper for srefs on s.get_consumers()"""
        block_rv = self.get_block_rv(block)
        return [self.get_block_sref(_) for _ in self.sched.get_consumers(block_rv)]


class BlockGraph:
    """Nested graph data structure"""

    # Input edges
    inputs: List[Edge]

    # Output edges
    outputs: List[Edge]

    # Input buffers, map to all edges on this buffer
    inputs_by_buffer: Dict[Buffer, List[Edge]]

    # Output buffers, map to all edges on this buffer
    outputs_by_buffer: Dict[Buffer, List[Edge]]

    # Subgraphs or single block
    data: Union[List["BlockGraph"], StmtSRef]

    # Parent subgraph
    parent: "BlockGraph" = None

    # Global static information for blocks
    block_info: BlockGraphInfo

    # Unique bitset identifier used to hash the subgraph via the blocks set
    graph_hash: int

    # Optional scheduler bind to the graph
    scheduler = None

    def __init__(self):
        """Constructor"""
        self.inputs = []
        self.inputs_by_buffer = {}
        self.outputs = []
        self.outputs_by_buffer = {}
        self.scheduler = None
        self.num_blocks = 0
        self.graph_hash = 0

    def set_scheduler(self, scheduler):
        """Bind a scheduler to the graph"""
        self.scheduler = scheduler

    def clean_scheduler(self):
        """Remove the scheduler of the graph"""
        self.scheduler = None

    def get_schedule(self) -> Schedule:
        """Get the global schedule"""
        return self.block_info.sched

    def is_single_block(self) -> bool:
        """Whether a single block subgraph"""
        return isinstance(self.data, StmtSRef)

    def get_blocks_in_reverse_topo_order(  # pylint: disable=invalid-name
        self,
    ) -> Iterable[StmtSRef]:
        """Iterate helper for blocks within graph"""
        if self.is_single_block():
            return [self.data]
        return itertools.chain(*[_.get_blocks_in_reverse_topo_order() for _ in reversed(self.data)])

    def get_blocks_in_topo_order(self) -> Iterable[StmtSRef]:
        """Iterate helper for blocks within graph"""
        if self.is_single_block():
            return [self.data]
        return itertools.chain(*[_.get_blocks_in_topo_order() for _ in self.data])

    def get_start_subgraphs(self) -> Set["BlockGraph"]:
        """Get subgraphs that has only external input dependencies"""
        res = set()
        for edge in self.inputs:
            if edge.dst_child is None:
                continue
            dst = edge.dst_child.dst
            if dst is None:
                continue
            if all([_.src is None or _.src.parent != self for _ in dst.inputs]):
                res.add(dst)
        return res

    def get_end_subgraphs(self) -> Set["BlockGraph"]:
        """Get subgraphs that has only external output dependencies"""
        res = set()
        for edge in self.outputs:
            if edge.src_child is None:
                continue
            src = edge.src_child.src
            if src is None:
                continue
            if all([_.dst is None or _.dst.parent != self for _ in src.outputs]):
                res.add(src)
        return res

    def __getitem__(self, key):
        """Override []"""
        return self.data.__getitem__(key)

    def print_partitions(self, indent=0, do_print=True):
        """Print utility for current subgraph partitions"""
        res = ""
        for _ in range(indent):
            res += "    "
        res += "- "
        subres = []
        if isinstance(self.data, StmtSRef):
            blocks = [self.data]
        else:
            blocks = []
            for subgraph in self.data:
                t, subblocks = subgraph.print_partitions(indent + 1, do_print=False)
                subres.append(t)
                blocks += subblocks

        if len(blocks) > 0:
            res += "["
        res += ", ".join([str(_.stmt.name_hint) for _ in blocks])
        if len(blocks) > 0:
            res += "]"
        res += " ("
        res += ", ".join([f"#{e.id}{f'^#{e.parent.id}' if e.parent else ''}" for e in self.inputs])
        res += ") -> ("
        res += ", ".join([f"#{e.id}{f'^#{e.parent.id}' if e.parent else ''}" for e in self.outputs])
        res += ")"
        if len(subres) > 0:
            res += "\n"
            res += "\n".join(subres)
        if do_print:
            print(res)
        return res, blocks

    def __add_input(self, edge: Edge):
        """Add input edge helper"""
        self.inputs.append(edge)
        if edge.buffer() not in self.inputs_by_buffer:
            self.inputs_by_buffer[edge.buffer()] = []
        self.inputs_by_buffer[edge.buffer()].append(edge)
        edge.dst = self
        edge.dst_idx = len(self.inputs) - 1

    def __add_output(self, edge: Edge):
        """Add output edge helper"""
        self.outputs.append(edge)
        if edge.buffer() not in self.outputs_by_buffer:
            self.outputs_by_buffer[edge.buffer()] = []
        self.outputs_by_buffer[edge.buffer()].append(edge)
        edge.src = self
        edge.src_idx = len(self.outputs) - 1

    def __reset_input(self, idx, edge):
        """Reset input edge helper"""
        origin = self.inputs[idx]
        assert origin.buffer().same_as(edge.buffer())
        self.inputs[idx] = edge
        buffer2edges = self.inputs_by_buffer[edge.buffer()]
        buffer2edges[buffer2edges.index(origin)] = edge

    def __reset_output(self, idx, edge):
        """Reset output edge helper"""
        origin = self.outputs[idx]
        assert origin.buffer().same_as(edge.buffer())
        self.outputs[idx] = edge
        buffer2edges = self.outputs_by_buffer[edge.buffer()]
        buffer2edges[buffer2edges.index(origin)] = edge

    def __create_edge(self, info: BlockEdge) -> Edge:
        """Create edge with unique identifier"""
        edge = Edge(self.block_info.edge_count, info)
        self.block_info.edge_count += 1
        return edge

    def validate(self) -> None:
        """Validate the graph nested structure"""
        has_parent = self.parent is not None

        # (1) external inputs
        in_edge_set = set(self.inputs)
        for idx, in_edge in enumerate(self.inputs):
            assert in_edge.dst == self, f"#{in_edge.id} dst invalid"
            assert in_edge.dst_idx == idx
            parent_edge = in_edge.parent
            if has_parent:
                if in_edge.src is None or in_edge.src.parent != self.parent:
                    assert parent_edge is not None
                    assert (
                        parent_edge.dst_child == in_edge
                    ), f"#{in_edge.id} is not dst child of #{parent_edge.id}"
                else:
                    assert parent_edge is None, f"#{in_edge.id} should not has parent edge"
            else:
                assert parent_edge is None, f"#{in_edge.id} should not has parent edge"
            src_child = in_edge.src_child
            if src_child:
                assert src_child.parent == in_edge
                assert src_child.dst is None or src_child.dst.parent == self
            dst_child = in_edge.dst_child
            if dst_child:
                assert dst_child.parent == in_edge
                assert dst_child.src is None or dst_child.src.parent == self
        for buffer in self.inputs_by_buffer:
            for in_edge in self.inputs_by_buffer[buffer]:
                assert in_edge in in_edge_set
                in_edge_set.remove(in_edge)
        assert len(in_edge_set) == 0

        # (2) external outputs
        out_edge_set = set(self.outputs)
        for idx, out_edge in enumerate(self.outputs):
            assert out_edge.src == self
            assert out_edge.src_idx == idx
            parent_edge = out_edge.parent
            if has_parent:
                if out_edge.dst is None or out_edge.dst.parent != self.parent:
                    assert parent_edge is not None, f"#{out_edge.id} should has parent edge"
                    assert parent_edge.src_child == out_edge
                else:
                    assert parent_edge is None
            else:
                assert parent_edge is None
            src_child = out_edge.src_child
            if src_child:
                assert (
                    src_child.parent == out_edge
                ), f"#{src_child.id}'s parent should be #{out_edge.id}"
                assert src_child.dst is None or src_child.dst.parent == self
            dst_child = out_edge.dst_child
            if dst_child:
                assert (
                    dst_child.parent == out_edge
                ), f"#{dst_child.id} parent should be #{out_edge.id}, get #{dst_child.parent.id}"
                assert dst_child.src is None or dst_child.src.parent == self
        for buffer in self.outputs_by_buffer:
            for out_edge in self.outputs_by_buffer[buffer]:
                assert (
                    out_edge in out_edge_set
                ), f"#{out_edge.id} not compatible {buffer} {[_.id for _ in self.outputs]}"
                out_edge_set.remove(out_edge)
        assert len(out_edge_set) == 0

        # (3) subgraphs
        if not self.is_single_block():
            for subgraph in self.data:
                subgraph.validate()

    def flatten(self) -> None:
        """Flatten internal nested structure"""
        new_data: List[BlockGraph] = []
        for subgraph in self.data:
            if subgraph.is_single_block():
                new_data.append(subgraph)
            else:
                new_data.extend(subgraph.data)
        for subgraph in new_data:
            for idx, in_edge in enumerate(subgraph.inputs):
                new_edge = in_edge.parent
                if not new_edge:
                    continue
                subgraph.__reset_input(idx, new_edge)
                new_edge.dst = subgraph
                new_edge.dst_idx = idx
                if in_edge.src_child:
                    in_edge.src_child.parent = new_edge
                    new_edge.src_child = in_edge.src_child
                if in_edge.dst_child:
                    in_edge.dst_child.parent = new_edge
                    new_edge.dst_child = in_edge.dst_child

            for idx, out_edge in enumerate(subgraph.outputs):
                new_edge = out_edge.parent
                if not new_edge:
                    continue
                subgraph.__reset_output(idx, new_edge)
                new_edge.src = subgraph
                new_edge.src_idx = idx
                if out_edge.src_child:
                    out_edge.src_child.parent = new_edge
                    new_edge.src_child = out_edge.src_child
                if out_edge.dst_child:
                    out_edge.dst_child.parent = new_edge
                    new_edge.dst_child = out_edge.dst_child

        self.data = new_data

    @staticmethod
    def merge(*subgraphs: "BlockGraph") -> "BlockGraph":
        """Merge subgraphs of the same parent graph"""
        parent = subgraphs[0].parent
        if parent is None:
            raise ValueError("Can not merge root graphs")
        for graph in subgraphs:
            if graph.parent != parent:
                raise ValueError("Can not merge subgraphs of different parent subgraph")
        merged = BlockGraph()
        merged.parent = parent
        merged.block_info = subgraphs[0].block_info
        merged.data = subgraphs
        merged.graph_hash = functools.reduce(lambda a, b: a | b.graph_hash, subgraphs, 0)
        merged.num_blocks = functools.reduce(lambda a, b: a + b.num_blocks, subgraphs, 0)

        subgraph_set = set(subgraphs)
        for graph in subgraphs:
            num_internal_edges = 0
            for edge in graph.inputs:
                if edge.parent is not None and edge.src is not None:
                    raise ValueError("Can not merge subgraphs of different parent subgraph")
                src = edge.src
                if src in subgraph_set:
                    num_internal_edges += 1
                    continue
                parent_edge = BlockGraph.__create_parent_edge(edge, edge_in_src_graph=False)
                merged.__add_input(parent_edge)
                if src:
                    src.__reset_output(parent_edge.src_idx, parent_edge)

            for edge in graph.outputs:
                if edge.parent is not None and edge.dst is not None:
                    raise ValueError("Can not merge subgraphs of different parent subgraph")
                dst = edge.dst
                if dst in subgraph_set:
                    num_internal_edges += 1
                    continue
                parent_edge = BlockGraph.__create_parent_edge(edge, edge_in_src_graph=True)
                merged.__add_output(parent_edge)
                if dst:
                    dst.__reset_input(parent_edge.dst_idx, parent_edge)

            if num_internal_edges == 0 and len(subgraphs) > 1:
                raise ValueError("Can not merge disjoint subgraphs")
        new_data = []
        insert = False
        for g in parent.data:
            if g in subgraphs:
                if not insert:
                    insert = True
                    new_data.append(merged)
            else:
                new_data.append(g)
        parent.data = new_data
        return merged

    @staticmethod
    def create(s: Schedule):
        """Create the root graph for current schedule"""
        block_info = BlockGraphInfo(s)
        graph = BlockGraph()
        graph.data = []
        graph.block_info = block_info
        graph.graph_hash = 0

        read_buffer_dict = []
        write_buffer_dict = []
        for block_idx in range(block_info.block_num()):
            sref = block_info.get_block_sref(block_idx)
            buffer_to_read_idx = {}
            for k, read in enumerate(sref.stmt.reads):
                if read.buffer not in buffer_to_read_idx:
                    buffer_to_read_idx[read.buffer] = []
                buffer_to_read_idx[read.buffer].append(k)
            buffer_to_write_idx = {}
            for k, write in enumerate(sref.stmt.writes):
                if write.buffer not in buffer_to_write_idx:
                    buffer_to_write_idx[write.buffer] = []
                buffer_to_write_idx[write.buffer].append(k)
            read_buffer_dict.append(buffer_to_read_idx)
            write_buffer_dict.append(buffer_to_write_idx)

            # for each block, we create a single node subgraph
            subgraph = BlockGraph()
            subgraph.data = sref
            subgraph.parent = graph
            subgraph.block_info = block_info
            subgraph.graph_hash = 1 << block_idx
            subgraph.num_blocks = 1
            graph.graph_hash |= subgraph.graph_hash
            graph.num_blocks += 1
            graph.data.append(subgraph)

        visited = set()

        def __fvisit(block_idx):
            if block_idx in visited:
                return
            visited.add(block_idx)
            block_sref = block_info.get_block_sref(block_idx)
            src_node = graph.data[block_idx]

            producer_srefs = block_info.get_producers(block_sref)
            for buffer in read_buffer_dict[block_idx]:
                if buffer in write_buffer_dict[block_idx]:
                    continue
                is_external = True
                for producer_sref in producer_srefs:
                    producer_idx = block_info.get_block_idx(producer_sref)
                    if buffer in write_buffer_dict[producer_idx]:
                        is_external = False
                        break
                if is_external:
                    block_read_indices = read_buffer_dict[block_idx][buffer]
                    edge = graph.__create_edge(
                        BlockEdge(None, None, block_sref, block_read_indices)
                    )
                    src_node.__add_input(edge)
                    graph.__add_input(
                        BlockGraph.__create_parent_edge(edge, edge_in_src_graph=False)
                    )

            consumer_srefs = block_info.get_consumers(block_sref)
            nexts = []
            for buffer in write_buffer_dict[block_idx]:
                is_external = True
                for consumer_sref in consumer_srefs:
                    consumer_idx = block_info.get_block_idx(consumer_sref)
                    nexts.append(consumer_idx)
                    if not buffer in read_buffer_dict[consumer_idx]:
                        continue
                    dst_node = graph.data[consumer_idx]
                    read_block_indices = read_buffer_dict[consumer_idx][buffer]
                    write_block_indices = write_buffer_dict[block_idx][buffer]
                    edge = graph.__create_edge(
                        BlockEdge(
                            block_sref, write_block_indices, consumer_sref, read_block_indices
                        )
                    )
                    src_node.__add_output(edge)
                    dst_node.__add_input(edge)
                    is_external = False
                if is_external:
                    block_write_indices = write_buffer_dict[block_idx][buffer]
                    edge = graph.__create_edge(
                        BlockEdge(block_sref, block_write_indices, None, None)
                    )
                    src_node.__add_output(edge)
                    graph.__add_output(
                        BlockGraph.__create_parent_edge(edge, edge_in_src_graph=True)
                    )

            for next_block_idx in nexts:
                __fvisit(next_block_idx)

        for block_idx in range(block_info.block_num()):
            __fvisit(block_idx)
        return graph

    @staticmethod
    def __create_parent_edge(edge: Edge, edge_in_src_graph: bool) -> "Edge":
        """Helper function to create parent edge"""
        parent_graph = edge.parent_graph()
        if parent_graph is None:
            raise ValueError("Can not create edge in non-existing parent graph")
        parent_edge = parent_graph.__create_edge(edge.block_edge)
        origin_parent = edge.parent
        if edge.src:
            parent_edge.src = edge.src
            parent_edge.src_idx = edge.src_idx
        if edge.src_child and not edge_in_src_graph:
            parent_edge.src_child = edge.src_child
            edge.src_child.parent = parent_edge
        if edge.dst:
            parent_edge.dst = edge.dst
            parent_edge.dst_idx = edge.dst_idx
        if edge.dst_child and edge_in_src_graph:
            parent_edge.dst_child = edge.dst_child
            edge.dst_child.parent = parent_edge
        parent_edge.parent = origin_parent
        if origin_parent:
            if edge_in_src_graph:
                origin_parent.src_child = parent_edge
            else:
                origin_parent.dst_child = parent_edge
        edge.parent = parent_edge
        if edge_in_src_graph:
            assert edge.src, f"#{edge.id} has no source"
            edge.dst = None
            edge.dst_idx = None
            edge.dst_child = None
            parent_edge.src_child = edge
        else:
            assert edge.dst, f"#{edge.id} has no dest"
            edge.src = None
            edge.src_idx = None
            edge.src_child = None
            parent_edge.dst_child = edge
        return parent_edge
