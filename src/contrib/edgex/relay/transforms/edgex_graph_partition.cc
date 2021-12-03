/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/tvm/relay/pass/fuse_ops.cc
 *
 * \brief This is a backend-aware optimization pass.
 *   Fuse necessary ops into a single one.
 */

#include "edgex_graph.h"

namespace tvm {
namespace relay {

/*!
 * \brief A partition of the graph marked by union find data structure.
 */
class EdgexSubFunctionGraphPartitioner {
 public:
  SubgraphIdSet CreateSubgraphIdSetByIdSet(const EdgexDependencyGraph& graph, const IdSet& id_set) {
    auto starts = GetStartsOfSubgraph(graph, id_set);
    auto ends = GetEndsOfSubgraph(graph, id_set);
    return {starts, ends, id_set};
  }

  SubgraphIdSet CreateSubgraphIdsetByNNP(const EdgexDependencyGraph& graph) {
    // step1: set device type for every node
    auto device_types = GetDeviceTypes(graph);

    // step2: find subgraph set of NNP subgraph
    IdSet all_ids;
    for (size_t nid = 0; nid < graph.size(); nid++) {
      EdgexDependencyGraph::Node* graph_node = graph[nid];
      if (device_types[nid] == S_DEVICE_EDGEX && is_main_node(graph_node->tvm_node)) {
        IdSetPushBack(&all_ids, nid);
      }
    }

    return CreateSubgraphIdSetByIdSet(graph, all_ids);
  }

  std::vector<SubgraphIdSet> CreateSubGraphIdSetsByDevice(const EdgexDependencyGraph& graph) {
    // get node_id set of subgraph
    auto device_types = GetDeviceTypes(graph);
    std::vector<IdSet> subgraph_sets;
    std::vector<size_t> subgraph_ids(graph.size(), SIZE_MAX);
    std::unordered_map<size_t, IdSet> subgraph_dependecies;

    auto push_to_subgraph = [&](size_t graph_node_id, size_t subgraph_id) {
      IdSetPushBack(&subgraph_sets[subgraph_id], graph_node_id);
      subgraph_ids[graph_node_id] = subgraph_id;
    };

    auto depend_on_subgraph = [&](size_t subgraph_id_from, size_t subgraph_id_to) {
      if (subgraph_dependecies.find(subgraph_id_from) == subgraph_dependecies.end()) {
        subgraph_dependecies[subgraph_id_from] = IdSet();
      }
      IdSetPushBack(&subgraph_dependecies[subgraph_id_from], subgraph_id_to);
    };

    auto merge_subgraph = [&](size_t subgraph_id_from, size_t subgraph_id_to) {
      if (subgraph_id_from == subgraph_id_to) return;

      // merge sets
      auto& subgraph_to = subgraph_sets[subgraph_id_to];
      auto& subgraph_from = subgraph_sets[subgraph_id_from];
#ifdef FUSIONS_DEBUG_1
      LOG(INFO) << "subgraph_id_from = " << subgraph_id_from;
      LOG(INFO) << "subgraph_id_to = " << subgraph_id_to;
      LOG(INFO) << "[1] subgraph_from = " << TextOfIdSet(subgraph_sets[subgraph_id_from]);
      LOG(INFO) << "[1] subgraph_to = " << TextOfIdSet(subgraph_sets[subgraph_id_to]);
#endif
      for (size_t id : subgraph_from) {
        IdSetPushBack(&subgraph_to, id);
        subgraph_ids[id] = subgraph_id_to;
      }
      subgraph_from.clear();
#ifdef FUSIONS_DEBUG_1
      LOG(INFO) << "[2] subgraph_from=" << TextOfIdSet(subgraph_sets[subgraph_id_from]);
      LOG(INFO) << "[2] subgraph_to=" << TextOfIdSet(subgraph_sets[subgraph_id_to]);
#endif
      // update dependencies
      auto dependency_to_iter = subgraph_dependecies.find(subgraph_id_to);
      auto dependency_from_iter = subgraph_dependecies.find(subgraph_id_from);
      if (dependency_from_iter != subgraph_dependecies.end()) {
        if (dependency_to_iter != subgraph_dependecies.end()) {
          auto& set_to = dependency_to_iter->second;
          auto& set_from = dependency_from_iter->second;
          for (size_t ii : set_from) {
            IdSetPushBack(&set_to, ii);
          }
        } else {
          subgraph_dependecies[subgraph_id_to] = dependency_from_iter->second;
        }
        subgraph_dependecies.erase(dependency_from_iter);
      }
      for (auto& kv : subgraph_dependecies) {
        if (IdSetFoundId(kv.second, subgraph_id_from)) {
          IdSetErase(&kv.second, subgraph_id_from);
          IdSetPushBack(&kv.second, subgraph_id_to);
        }
      }
    };

    for (size_t nid = 0; nid < graph.size(); nid++) {
      if (!IsValidGraphNode(nid, graph)) continue;
#ifdef FUSIONS_DEBUG_0
      LOG(INFO) << "-- nid = " << nid;
#endif
      auto* graph_node = graph[nid];
      for (auto n = graph_node->children.head; n != nullptr; n = n->next) {
        auto* child = n->value;
        size_t child_id = child->index;
        if (IsValidGraphNode(child_id, graph)) {
          size_t child_subgraph_id = subgraph_ids[child_id];
          CHECK_NE(child_subgraph_id, SIZE_MAX);  // TODO(cww): remove
          if (device_types[nid] == device_types[child_id]) {
            // check subgraph dependency
            bool check_dependency_ok = true;
            for (auto nn = graph_node->children.head; nn != nullptr; nn = nn->next) {
              auto* child2 = nn->value;
              size_t child2_id = child2->index;
              if (subgraph_ids[child2_id] != subgraph_ids[nid] &&
                  subgraph_ids[child2_id] != subgraph_ids[child_id]) {
                if (IsDepndOn(child2_id, child_id, subgraph_dependecies)) {
                  check_dependency_ok = false;
                }
              }
            }
            if (check_dependency_ok) {
              // merge nid and child_id
              if (subgraph_ids[nid] == SIZE_MAX) {
                push_to_subgraph(nid, subgraph_ids[child_id]);
              } else {
                merge_subgraph(subgraph_ids[child_id], subgraph_ids[nid]);
              }
            }
          }
        }
      }
#ifdef FUSIONS_DEBUG_0
      LOG(INFO) << "nid = " << nid << " subgraph_id = " << subgraph_ids[nid];
#endif

      // create new subgraph
      if (subgraph_ids[nid] == SIZE_MAX) {
        size_t new_id = subgraph_sets.size();
        subgraph_ids[nid] = new_id;
        IdSet new_set{nid};
        subgraph_sets.push_back(new_set);
      }
#ifdef FUSIONS_DEBUG_0
      LOG(INFO) << "nid = " << nid << " subgraph_id = " << subgraph_ids[nid];
#endif

      // add subgraph dependency
      for (auto n = graph_node->children.head; n != nullptr; n = n->next) {
        auto* child = n->value;
        size_t child_id = child->index;
        if (IsValidGraphNode(child_id, graph)) {
          if (subgraph_ids[nid] != subgraph_ids[child_id]) {
            depend_on_subgraph(nid, child_id);
          }
        }
      }
    }

#ifdef FUSIONS_DEBUG_1
    // check if any nid missing
    for (size_t nid = 0; nid < graph.size(); nid++) {
      if (IsValidGraphNode(nid, graph)) {
        bool found = false;
        auto* graph_node = graph[nid];
        for (auto s : subgraph_sets) {
          if (IdSetFoundId(s, nid)) {
            if (found == true)
              LOG(FATAL) << "Duplicate nid " << nid << " in subgraph_sets";
            else
              found = true;
          }
        }
        if (!found) LOG(FATAL) << "Not found nid " << nid << " in subgraph_sets";
      }
    }
#endif

    // get start set and end set from subgraph set
    std::vector<SubgraphIdSet> subgraph_idsets;
    for (auto s : subgraph_sets) {
#ifdef FUSIONS_DEBUG_1
      LOG(INFO) << "--" << TextOfIdSet(s);
#endif
      if (s.size() > 0) {
        auto subgraph_idset = CreateSubgraphIdSetByIdSet(graph, s);
        subgraph_idsets.push_back(subgraph_idset);
      }
    }
    return subgraph_idsets;
  }

  IdSet GetAllIdsOfSubgraphByFrontier_old(const EdgexDependencyGraph& graph, const IdSet& starts,
                                          const IdSet& ends) {
    // forward
    IdSet forward_set;
    for (size_t nid = 0; nid < graph.size(); ++nid) {
      auto* graph_node = graph[nid];
      if (IdSetFoundId(starts, nid)) {
        IdSetPushBack(&forward_set, nid);
      } else {
        for (auto* n = graph_node->children.head; n != nullptr; n = n->next) {
          if (IdSetFoundId(forward_set, n->value->index) && !IdSetFoundId(ends, n->value->index)) {
            IdSetPushBack(&forward_set, nid);
          }
        }
      }
    }

    // backward
    IdSet backward_set;
    for (size_t nid = graph.size() - 1; nid >= 0 && nid < graph.size(); nid--) {
      auto* graph_node = graph[nid];
      if (IdSetFoundId(ends, nid)) {
        IdSetPushBack(&backward_set, nid);
      } else {
        for (auto* n = graph_node->parents.head; n != nullptr; n = n->next) {
          if (IdSetFoundId(backward_set, n->value->index) &&
              !IdSetFoundId(starts, n->value->index)) {
            IdSetPushBack(&backward_set, nid);
          }
        }
      }
    }

    IdSet subgraph_set = IdSetIntersection(forward_set, backward_set);
    return subgraph_set;
  }

 private:
  bool IsValidGraphNode(size_t nid, const EdgexDependencyGraph& graph) {
    return is_main_node(graph[nid]->tvm_node);
  }

  std::vector<int> GetDeviceTypes(const EdgexDependencyGraph& graph) {
    std::vector<int> device_types(graph.size(), S_DEVICE_NULL);
    for (size_t nid = 0; nid < graph.size(); nid++) {
      EdgexDependencyGraph::Node* graph_node = graph[nid];

      if (auto* callnode = GetRef<ObjectRef>(graph_node->tvm_node).as<CallNode>()) {
        if (callnode->op.same_as(Op::Get("on_device"))) {
          device_types[nid] = S_DEVICE_DEDSP;
        }
      }

      for (auto* n = graph_node->parents.head; n != nullptr; n = n->next) {
        auto* next_graph_node = n->value;
        if (auto* callnode = GetRef<ObjectRef>(next_graph_node->tvm_node).as<CallNode>()) {
          if (callnode->op.same_as(Op::Get("on_device"))) {
            device_types[nid] = S_DEVICE_DEDSP;
            break;
          }
        }
      }

      if (device_types[nid] == S_DEVICE_NULL) {
        device_types[nid] = S_DEVICE_EDGEX;
      }
#ifdef FUSIONS_DEBUG_0
      LOG(INFO) << "nid=" << nid << " device_type=" << device_types[nid];
#endif
    }
    return device_types;
  }

  IdSet GetStartsOfSubgraph(const EdgexDependencyGraph& graph, const IdSet& subgraph_set) {
    IdSet starts;
    for (auto nid : subgraph_set) {
      auto* graph_node = graph[nid];
      bool is_start = false;
      if (graph_node->children.head != nullptr) {
        for (auto* n = graph_node->children.head; n != nullptr; n = n->next) {
          // TODO(cww): SIZE_MAX is not valid node, such as Constant
          if (n->value->index != SIZE_MAX && !IdSetFoundId(subgraph_set, n->value->index)) {
            is_start = true;
            break;
          }
        }
      } else {  // the first node
        is_start = true;
      }
      if (is_start) IdSetPushBack(&starts, nid);
    }

    return starts;
  }

  IdSet GetEndsOfSubgraph(const EdgexDependencyGraph& graph, const IdSet& subgraph_set) {
    IdSet ends;
    for (auto nid : subgraph_set) {
      auto* graph_node = graph[nid];
      bool is_end = false;
      if (graph_node->parents.head != nullptr) {
        for (auto* n = graph_node->parents.head; n != nullptr; n = n->next) {
          if (!IdSetFoundId(subgraph_set, n->value->index)) {
            is_end = true;
            break;
          }
        }
      } else {  // the last node
        is_end = true;
      }

      if (is_end) IdSetPushBack(&ends, nid);
    }
    return ends;
  }

  // if a_id is depend on b_id
  bool IsDepndOn(size_t a_id, size_t b_id, const std::unordered_map<size_t, IdSet>& dependencies) {
    if (dependencies.find(a_id) != dependencies.end()) {
      auto sets = dependencies.at(a_id);
      for (auto id : sets) {
        if (id == b_id || IsDepndOn(id, b_id, dependencies)) {
          return true;
        }
      }
    }
    return false;
  }
};

// Used by FusionStitching
SubgraphIdSet CreateSubgraphIdSet(const EdgexDependencyGraph& dgraph, const std::set<size_t>& ids) {
  auto graph_partitioner = EdgexSubFunctionGraphPartitioner();
  IdSet id_set(ids.begin(), ids.end());
  return graph_partitioner.CreateSubgraphIdSetByIdSet(dgraph, id_set);
}

Expr EdgexGraphPartitionByIdSet(const Expr& expr, const IdSet& id_set) {
  support::Arena arena;

  auto func = InferType(expr);
  auto dgraph = EdgexDependencyGraph::Create(&arena, func);

  auto graph_partitioner = EdgexSubFunctionGraphPartitioner();
  SubgraphIdSet subgraph_idset = graph_partitioner.CreateSubgraphIdSetByIdSet(dgraph, id_set);

  auto extractor = EdgexSubFunctionExtractor();
  return extractor.Transform(func, dgraph, subgraph_idset);
}

// Used by FusionStitching
Expr EdgexGraphPartitionByIdSet(const Expr& expr, const std::set<size_t>& ids) {
  support::Arena arena;

  IdSet id_set;
  for (size_t id : ids) {
    IdSetPushBack(&id_set, id);
  }

  return EdgexGraphPartitionByIdSet(expr, id_set);
}

Expr EdgexGraphPartitionByIdset(const Expr& expr, const Array<Integer> idset) {
  support::Arena arena;

  IdSet id_set;
  for (auto i : idset) {
    IdSetPushBack(&id_set, i->value);
  }

  return EdgexGraphPartitionByIdSet(expr, id_set);
}

Expr EdgexGraphPartitionByFrontier(const Expr& expr, const Array<Integer> starts,
                                   const Array<Integer> ends) {
  support::Arena arena;

  IdSet old_start_set, old_end_set;
  for (auto i : starts) {
    IdSetPushBack(&old_start_set, i->value);
  }
  for (auto i : ends) {
    IdSetPushBack(&old_end_set, i->value);
  }

  auto func = InferType(expr);
  auto dgraph = EdgexDependencyGraph::Create(&arena, func);

  auto graph_partitioner = EdgexSubFunctionGraphPartitioner();
  auto all_ids =
      graph_partitioner.GetAllIdsOfSubgraphByFrontier_old(dgraph, old_start_set, old_end_set);
  SubgraphIdSet subgraph_idset = graph_partitioner.CreateSubgraphIdSetByIdSet(dgraph, all_ids);

  auto extractor = EdgexSubFunctionExtractor();
  return extractor.Transform(func, dgraph, subgraph_idset);
}

Expr EdgexGraphPartitionByNNP(const Expr& expr) {
  support::Arena arena;
  auto func = InferType(expr);
  auto graph = EdgexDependencyGraph::Create(&arena, func);

  auto graph_partitioner = EdgexSubFunctionGraphPartitioner();
  auto subgraph_idset = graph_partitioner.CreateSubgraphIdsetByNNP(graph);

  auto extractor = EdgexSubFunctionExtractor();
  return extractor.Transform(func, graph, subgraph_idset);
}

Array<Expr> EdgexGraphPartitionByDevice(const Expr& expr) {
  support::Arena arena;
  auto func = InferType(expr);
  auto graph = EdgexDependencyGraph::Create(&arena, func);

  // get frontier list of subgraphs
  auto graph_partitionier = EdgexSubFunctionGraphPartitioner();
  auto subgraph_idsets = graph_partitionier.CreateSubGraphIdSetsByDevice(graph);

  // get subfunc from function by frontier
  Array<Expr> ret;
  for (auto subgraph_idset : subgraph_idsets) {
    // EdgexSubFunctionExtractor is not clear and needs to init every time.
    auto extractor = EdgexSubFunctionExtractor();
    auto subfunc = extractor.Transform(func, graph, subgraph_idset);
    ret.push_back(subfunc);
  }
  return ret;
}

TVM_REGISTER_GLOBAL("relay._ir_pass.graph_partition_by_idset")
    .set_body_typed(EdgexGraphPartitionByIdset);

TVM_REGISTER_GLOBAL("relay._ir_pass.graph_partition_by_frontier")
    .set_body_typed(EdgexGraphPartitionByFrontier);

TVM_REGISTER_GLOBAL("relay._ir_pass.graph_partition_by_nnp")
    .set_body_typed(EdgexGraphPartitionByNNP);

TVM_REGISTER_GLOBAL("relay._ir_pass.graph_partition_by_device")
    .set_body_typed(EdgexGraphPartitionByDevice);

}  // namespace relay
}  // namespace tvm
