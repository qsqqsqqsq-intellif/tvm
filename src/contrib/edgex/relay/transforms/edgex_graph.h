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
 *
 * \file tvm/contrib/edgex/relay/transforms/edgex_graph.h
 * \brief Utilities for writing passes
 */
#ifndef TVM_CONTRIB_EDGEX_RELAY_TRANSFORMS_EDGEX_GRAPH_H_
#define TVM_CONTRIB_EDGEX_RELAY_TRANSFORMS_EDGEX_GRAPH_H_

#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/attrs/device_copy.h>

#include <algorithm>
#include <climits>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../../../relay/backend/utils.h"
#include "../../../../relay/transforms/pass_utils.h"
#include "../../../../relay/transforms/pattern_utils.h"

namespace tvm {
namespace relay {
using TargetsMap = Map<tvm::Integer, tvm::Target>;

#define S_DEVICE_DEDSP (tvm::Target("dedsp")->kind->device_type)
#define S_DEVICE_EDGEX 16  // (tvm::Target("edgex")->kind->device_type)
#define S_DEVICE_NULL -1

using support::LinkedList;
using support::LinkNode;

template <typename R, typename... Args>
R EdgexCallPackedFunc(const std::string& name, Args... args) {
  auto pf = backend::GetPackedFunc(name);
  return (*pf)(std::forward<Args>(args)...);
}

template <typename... Args>
Function EdgexCallPackedFunc(const std::string& name, Args... args) {
  auto pf = backend::GetPackedFunc(name);
  if (pf)
    return (*pf)(std::forward<Args>(args)...);
  else
    return Function();
}

inline Function InferType(const Expr& expr) {
  auto mod = IRModule::FromExpr(expr);
  mod = transform::InferType()(mod);
  return Downcast<Function>(mod->Lookup("main"));
}

inline std::string GetNodeDetails(const tvm::Object* tvm_node) {
  std::ostringstream os;
  if (tvm_node) {
    os << " tvm::Object[" << tvm_node->_type_key;
    if (auto* callnode = GetRef<ObjectRef>(tvm_node).as<CallNode>()) {
      if (callnode->op.as<OpNode>()) {
        os << ", " << callnode->op;
      } else {
        os << " is not op";
      }
    }
    os << ", " << tvm_node;
  } else {
    os << " tvm:: Node[nullptr";
  }
  os << "]";
  return os.str();
}
#if 0
/*!
 * \brief GraphCodegen module wrapper
 *
 */
struct EdgexGraphCodegen {
 public:
  EdgexGraphCodegen() {
    auto pf = backend::GetPackedFunc("relay.build_module._GraphRuntimeCodegen");
    mod = (*pf)();
  }
  ~EdgexGraphCodegen() {}

  void Init(runtime::Module* m, TargetsMap targets) { CallFunc("init", m, targets); }

  void Codegen(const Function& func) { CallFunc("codegen", func); }

  void Reset() { CallFunc("reset"); }

  std::string GetJSON() { return CallFunc<std::string>("get_graph_json", nullptr); }

  Map<std::string, Array<LoweredFunc>> GetLoweredFunc() {
    return CallFunc<Map<std::string, Array<LoweredFunc>>>("get_lowered_funcs", nullptr);
  }

  std::unordered_map<std::string, tvm::runtime::NDArray> GetParams() {
    std::unordered_map<std::string, tvm::runtime::NDArray> ret;
    auto names = CallFunc<Array<HalideIR::Expr>>("list_params_name", nullptr);
    for (auto expr : names) {
      auto key = expr.as<ir::StringImm>()->value;
      ret[key] = CallFunc<runtime::NDArray>("get_param_by_name", key);
    }
    return ret;
  }

 protected:
  tvm::runtime::Module mod;
  template <typename R, typename... Args>
  R CallFunc(const std::string& name, Args... args) {
    auto pf = mod.GetFunction(name, false);
    return pf(std::forward<Args>(args)...);
  }
  template <typename... Args>
  void CallFunc(const std::string& name, Args... args) {
    auto pf = mod.GetFunction(name, false);
    pf(std::forward<Args>(args)...);
    return;
  }
};
#endif
/*!
 * \brief Group as a union find data structure.
 */
struct Group {
  /*! \brief The parent in the union find data structure. */
  Group* parent{nullptr};
  /*! \brief The pattern of the group */
  OpPatternKind pattern;
  /*! \brief reference to the root node. */
  const tvm::Object* root_ref{nullptr};
  /*!
   * \brief Reference to the master node,
   * this field is not nullptr only if pattern is kOutEWiseFusable.
   */
  const tvm::Object* master_ref{nullptr};

  int device_type{S_DEVICE_NULL};

  bool is_stride_gt_1{false};

  bool is_data_oversize{false};

  bool after_split_node{false};

  /*!
   * \brief Find the group root, perform path compression
   * \return The root type node.
   */
  Group* FindRoot() {
    // fast path
    if (this->parent == nullptr) return this;
    // slow path with path compression.
    Group* root = this;
    while (root->parent != nullptr) {
      root = root->parent;
    }
    for (Group* p = this; p != root;) {
      Group* parent = p->parent;
      p->parent = root;
      p = parent;
    }
    return root;
  }

  std::string root_details() {
    std::ostringstream os;
    os << " GroupNode: ";
    os << GetNodeDetails(root_ref);
    os << " pattern[" << pattern << "]";
    os << " device_type[" << device_type << "]";
    return os.str();
  }

  std::string master_details() { return GetNodeDetails(master_ref); }
};
/*!
 * \brief Indexed data flow graph in forward direction.
 *  This is a temporary data structure used for operator fusion analysis.
 *
 *  This data structure only captures the dataflow fragement and
 *  could ignore blocks like let by simply ordering each dataflow block
 *  and mark the output node as extern_ref;
 */
class EdgexIndexedForwardGraph {
 public:
  struct Node;
  /*!
   * The forward edge in the dataflow graph.
   */
  struct Edge {
    /*! \brief The corresponding node */
    Node* node{nullptr};
    /*! \brief The respective pattern of this op */
    OpPatternKind pattern{kOpaque};
  };
  /*! \brief A node in the graph. */
  struct Node {
    /*! \brief weak reference to the corresponding edge. */
    const tvm::Object* ref{nullptr};
    /*! \brief The index of the node in topological order. */
    size_t index{0};
    /*! \brief Whether this node is referenced by external source */
    bool extern_ref{false};
    /*! \brief The general pattern in the node */
    OpPatternKind pattern{kOpaque};
    /*! \brief The inputs of the node. */
    LinkedList<Edge> inputs;
    /*! \brief The outputs of the node. */
    LinkedList<Edge> outputs;

    int InputOpNum() const {
      int input_op_num = 0;
      for (auto* link = inputs.head; link != nullptr; link = link->next) {
        auto input_node = GetRef<ObjectRef>(link->value.node->ref);
        if (input_node.as<CallNode>() || input_node.as<TupleNode>() ||
            input_node.as<TupleGetItemNode>()) {
          input_op_num++;
        }
      }
      return input_op_num;
    }

    int OutputNum() const {
      int output_num = 0;
      for (auto* link = outputs.head; link != nullptr; link = link->next) {
        output_num++;
      }
      return output_num;
    }

    std::string details() const { return GetNodeDetails(ref); }

    std::string outputs_details() const {
      std::ostringstream os;
      os << "graph_node[" << index << "], "
         << " outputs=[";
      for (auto* link = outputs.head; link != nullptr; link = link->next) {
        os << link->value.node->index << ", ";
      }
      os << "]";
      return os.str();
    }
  };

  /*! \brief The node map that maps node to graph */
  std::unordered_map<const tvm::Object*, Node*> node_map;
  /*! \brief All the nodes in post DFS order */
  std::vector<Node*> post_dfs_order;

  /*! \brief Dump the graph into string. */
#if 0
  void DebugDump() {
    std::ostringstream os;
    for (size_t i = 0; i < post_dfs_order.size(); ++i) {
      Node* node = post_dfs_order[i];
      os << "node[" << i << "], "
         << GetRef<ObjectRef>(node->ref)
         << " outputs=[";
      for (auto* link = node->outputs.head; link != nullptr; link = link->next) {
        os << link->value.node->index << ", ";
      }
      os << "]\n";
    }
    LOG(INFO) << os.str();
  }
#else
  void DebugDump() {
    std::ostringstream os;
    for (size_t i = 0; i < post_dfs_order.size(); ++i) {
      Node* node = post_dfs_order[i];
      // auto *constantnode = GetRef<ObjectRef>(node->ref).as<ConstantNode>();
      // if (constantnode != NULL) continue;
      auto* callnode = GetRef<ObjectRef>(node->ref).as<CallNode>();
      if (callnode == NULL) continue;
      os << "node[" << i << "], "
         << callnode->op
         // << GetRef<ObjectRef>(node->ref)
         << " outputs=[";
      for (auto* link = node->outputs.head; link != nullptr; link = link->next) {
        os << link->value.node->index << ", ";
      }
      os << "]\n";
    }
    LOG(INFO) << os.str();
  }
#endif

  /*!
   * \brief create a indexed forward graph.
   * \param arena The arena used for data allocation.
   * \param body The body of the expression to create a graph.
   */
  static EdgexIndexedForwardGraph Create(support::Arena* arena, const Expr& body);

 private:
  class Creator;
};

/*!
 * \brief Dominator tree that represent domination or
 *  post domination relation of the node.
 */
class EdgexDominatorTree {
 public:
  /*!
   * \brief A node in the dominator tree.
   */
  struct Node {
    /*! \brief The node in the tree */
    EdgexIndexedForwardGraph::Node* gnode{nullptr};
    /*! \brief parent of the tree */
    Node* parent{nullptr};
    /*! \brief current depth*/
    int depth{0};
    /*! \brief aggregated pattern to parent */
    OpPatternKind pattern{kOpaque};
  };
  // index -> node.
  std::vector<Node*> nodes;
  /*!
   * \brief compute a post dominator relation for a given dataflow graph.
   * \param arena The arena used for node allocation.
   * \param graph The graph to be analyze.
   * \return The dominator tree of the graph.
   * \note This algorithm makes use of the fact that graph is DAG,
   *       and runs a single pass algorithm via LCA.
   */
  static EdgexDominatorTree PostDom(support::Arena* arena, const EdgexIndexedForwardGraph& graph);

 private:
  // Combine pattern together.
  static OpPatternKind CombinePattern(OpPatternKind lhs, OpPatternKind rhs) {
    if (lhs > rhs) return lhs;
    return rhs;
  }
  /*!
   * \brief Find the least common acenstor of the two nodes.
   * \param lhs The left node.
   * \param rhs The right node.
   * \param edge_pattern
   *        The combined edge pattern across all the parents.
   * \return The least common ancestor of thw two.
   */
  static Node* LeastCommonAncestor(Node* lhs, Node* rhs, OpPatternKind* edge_pattern) {
    while (lhs != rhs) {
      if (lhs == nullptr) return nullptr;
      if (rhs == nullptr) return nullptr;
      if (lhs->depth < rhs->depth) {
        edge_pattern[0] = CombinePattern(edge_pattern[0], rhs->pattern);
        rhs = rhs->parent;
      } else if (rhs->depth < lhs->depth) {
        edge_pattern[0] = CombinePattern(edge_pattern[0], lhs->pattern);
        lhs = lhs->parent;
      } else {
        edge_pattern[0] = CombinePattern(edge_pattern[0], lhs->pattern);
        edge_pattern[0] = CombinePattern(edge_pattern[0], rhs->pattern);
        lhs = lhs->parent;
        rhs = rhs->parent;
      }
    }
    return lhs;
  }
};

/* EdgexDependencyGraph track input and output of an Expr.
 * Additionally, dummy scope is created to model scope.
 * It allow us to traverse the graph in reverse order.
 */
class EdgexDependencyGraph {
 public:
  /*! \brief A node in the graph. */
  struct Node {
    // index of post dfs order
    size_t index{SIZE_MAX};
    // tvm Node
    tvm::Object* tvm_node;
    // incoming edges
    LinkedList<Node*> children;
    // outgoing edges
    LinkedList<Node*> parents;

    int children_size() {
      int i = 0;
      for (auto h = children.head; h != nullptr; h = h->next) {
        i++;
      }
      return i;
    }

    int parents_size() {
      int i = 0;
      for (auto h = parents.head; h != nullptr; h = h->next) {
        i++;
      }
      return i;
    }

    std::string detail() {
      std::ostringstream os;
      os << "Index[" << index << "]";
      if (auto callnode = GetRef<ObjectRef>(tvm_node).as<CallNode>()) {
        os << ", OP[" << callnode->op << "]";
      } else if (GetRef<ObjectRef>(tvm_node).as<TupleNode>()) {
        os << ", OP[tuple]";
      } else if (GetRef<ObjectRef>(tvm_node).as<TupleGetItemNode>()) {
        os << ", OP[tuple_get]";
      }
      os << ", children_size[" << children_size() << "], parents_size[" << parents_size() << "]";
      return os.str();
    }
  };

  Node* operator[](size_t index) const { return post_dfs_order_[index]; }

  size_t size() const { return post_dfs_order_.size(); }

  size_t index_of(const tvm::Object* tvm_node) const {
    CHECK(tvm_node_2_node_.count(tvm_node) > 0)
        << "Cannot find graph_node of " << GetNodeDetails(tvm_node);
    return tvm_node_2_node_.at(tvm_node)->index;
  }

  Node* node_of(const tvm::Object* tvm_node) const {
    CHECK(tvm_node_2_node_.count(tvm_node) > 0)
        << "Cannot find graph_node of " << GetNodeDetails(tvm_node);
    return tvm_node_2_node_.at(tvm_node);
  }

  bool found_node(const tvm::Object* tvm_node) const {
    if (tvm_node_2_node_.count(tvm_node) > 0) {
      return true;
    } else {
      return false;
    }
  }

  void set_map(Node* node) { tvm_node_2_node_[node->tvm_node] = node; }

  void push_back(Node* node) {
    node->index = post_dfs_order_.size();
    post_dfs_order_.push_back(node);
  }

  void pop_back() { post_dfs_order_.pop_back(); }

  /*!
   * \brief Create a dependency graph.
   * \param arena The arena used for data allocation.
   * \param body The body of the expression to create a graph.
   */
  static EdgexDependencyGraph Create(support::Arena* arena, const Expr& body);

 private:
  /*! \brief The dependency graph in post DFS order. */
  std::vector<Node*> post_dfs_order_;

  /*! \brief Maps a tvm::ObjectRef to its node in the dependency graph. */
  // std::unordered_map<ObjectRef, Node*, NodeHash, NodeEqual> tvm_node_ref_2_node_;
  std::unordered_map<const tvm::Object*, Node*> tvm_node_2_node_;

  class Creator;
};

using IdSet = std::vector<size_t>;
// using Frontier = std::tuple<IdSet, IdSet>;

struct SubgraphIdSet {
  IdSet starts;
  IdSet ends;
  IdSet all;
};

// inline SubgraphIdSet GetSubgraphIdSet(const Frontier& frontier, const IdSet& all_ids) {
//   return {std::get<0>(frontier), std::get<1>(frontier), all_ids};
// }

inline bool IdSetFoundId(const IdSet& id_set, size_t id) {
  if (std::find(id_set.begin(), id_set.end(), id) != id_set.end()) {
    return true;
  } else {
    return false;
  }
}

inline void IdSetPushBack(IdSet* p_id_set, size_t id) { p_id_set->push_back(id); }

inline void IdSetErase(IdSet* p_id_set, size_t id) {
  auto iter = std::find(p_id_set->begin(), p_id_set->end(), id);
  if (iter != p_id_set->end()) {
    p_id_set->erase(iter);
  }
}

inline std::string TextOfIdSet(const IdSet& int_set) {
  std::ostringstream os;
  for (size_t i : int_set) {
    os << " " << i;
  }
  return os.str();
}

inline std::string TextOfStdSet(const std::set<size_t>& id_set) {
  std::ostringstream os;
  for (size_t i : id_set) {
    os << " " << i;
  }
  return os.str();
}

inline std::string TextOfSubgraphIdset(const SubgraphIdSet& subgraph_idset) {
  std::ostringstream os;
  os << std::endl << "[subgraph_idset] starts = " << TextOfIdSet(subgraph_idset.starts);
  os << std::endl << "[subgraph_idset] ends = " << TextOfIdSet(subgraph_idset.ends);
  os << std::endl << "[subgraph_idset] all = " << TextOfIdSet(subgraph_idset.all);
  return os.str();
}

inline IdSet IdSetIntersection(IdSet v1, IdSet v2) {
  IdSet v;
  std::sort(v1.begin(), v1.end());
  std::sort(v2.begin(), v2.end());
  std::set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(v));
  return v;
}

inline bool is_main_node(const Object* node) {
  if (node->IsInstance<CallNode>() || node->IsInstance<TupleNode>() ||
      node->IsInstance<TupleGetItemNode>()) {
    return true;
  } else {
    return false;
  }
}

inline bool is_var_node(const Object* node) {
  if (node->IsInstance<VarNode>()) {
    return true;
  } else {
    return false;
  }
}

inline bool is_const_node(const Object* node) {
  if (node->IsInstance<ConstantNode>()) {
    return true;
  } else {
    return false;
  }
}

SubgraphIdSet CreateSubgraphIdSet(const EdgexDependencyGraph& dgraph, const std::set<size_t>& ids);

Expr EdgexGraphPartitionByIdSet(const Expr& expr, const std::set<size_t>& ids);

class EdgexSubFunctionExtractor : private ExprMutator {
 public:
  // Run the transform
  Expr Transform(const Expr& body, const EdgexDependencyGraph& graph,
                 const SubgraphIdSet& subgraph_idset) {
    Reset();

    auto starts = subgraph_idset.starts;
    auto ends = subgraph_idset.ends;
    auto all_ids = subgraph_idset.all;

    auto f_check = [=](size_t i) -> bool {
      if (i >= 0 && i < graph.size()) {
        if (is_main_node(graph[i]->tvm_node)) return true;
      }
      return false;
    };

    for (size_t start : starts) {
#ifdef FUSIONS_DEBUG_0
      LOG(INFO) << "start = " << start;
#endif
      if (!f_check(start))
        LOG(FATAL)
            << "start index " << start
            << " is not valid, graph_partition only support call/tuple/tuple_get, please check.";
    }

    for (size_t end : ends) {
#ifdef FUSIONS_DEBUG_0
      LOG(INFO) << "end = " << end;
#endif
      if (!f_check(end))
        LOG(FATAL)
            << "end index " << end
            << " is not valid, graph_partition only support call/tuple/tuple_get, please check.";
    }

    for (size_t id : all_ids) {
#ifdef FUSIONS_DEBUG_0
      LOG(INFO) << "id = " << id;
#endif
      if (!f_check(id))
        LOG(FATAL)
            << "all_ids index " << id
            << " is not valid, graph_partition only support call/tuple/tuple_get, please check.";
    }

    graph_ = graph;
    starts_ = starts;
    ends_ = ends;
    all_ids_ = all_ids;
    ends_expr_.resize(ends_.size());
    return this->Mutate(body);
  }

 private:
  // The ends exprs found by visit
  std::vector<Expr> ends_expr_;

  // The new parameters of the new function.
  int param_index_ = 0;
  Array<Var> params_;
  // The parameters need to remove from function params.
  std::set<Var> params_to_remove_;
  // The parameters of orignal function.
  tvm::Array<Var> original_params_;

  // The arguments to call the functions.
  Array<Expr> arguments_;

  support::Arena arena_;
  IdSet starts_, ends_, all_ids_;
  EdgexDependencyGraph graph_;

  void Reset() {
    param_index_ = 0;
    while (!params_.empty()) params_.pop_back();
    while (!arguments_.empty()) arguments_.pop_back();
    while (!ends_expr_.empty()) ends_expr_.pop_back();
  }

  // Get a new parameter or allocate an old one
  Var GetOrAllocParam(const Expr& expr, const Type& type) {
    // run linear scan as most fused groups contain only a few inputs.
    for (size_t i = 0; i < arguments_.size(); ++i) {
      if (expr.same_as(arguments_[i])) return params_[i];
    }

    // create a new parameter.
    std::ostringstream os;
    if (auto* var_node = expr.as<VarNode>()) {
      os << var_node->name_hint();  // remain original name of variable
    } else {
      if (param_index_ == 0) {
        os << "data";
      } else {
        os << "data" << param_index_;
      }
      param_index_++;
    }

    auto var = Var(os.str(), type);
    params_.push_back(var);
    arguments_.push_back(expr);
    return var;
  }

  bool IsVarInOriginalFunction(const Expr& expr) {
    for (auto param : original_params_) {
      if (param.same_as(expr)) return true;
    }
    return false;
  }

  Expr VisitExpr(const Expr& expr) {
    auto it = this->memo_.find(expr);
    if (it != this->memo_.end()) {
      return it->second;
    } else {
      auto new_expr = ExprMutator::VisitExpr(expr);

      if (graph_.found_node(expr.get())) {
        size_t index = graph_.index_of(expr.get());
        auto id_iter = std::find(ends_.begin(), ends_.end(), index);
        if (id_iter != ends_.end()) {
          int id_index = std::distance(ends_.begin(), id_iter);
          ends_expr_[id_index] = new_expr;
        }
      }

      return new_expr;
    }
  }

  // Skip primitive function.
  Expr VisitExpr_(const FunctionNode* fn_node) {
    original_params_ = fn_node->params;
    auto body = this->Mutate(fn_node->body);

    // remove var from final tuple
    Array<Expr> ends_expr_final;
    for (auto e : ends_expr_) {
      if (is_main_node(e.get())) {
        if (auto* tuple = e.as<TupleNode>()) {
          for (auto field : tuple->fields) {
            // ignore var in ends_expr_final and remove it from function params
            if (auto* var_node = field.as<VarNode>()) {
              params_to_remove_.insert(GetRef<Var>(var_node));
            } else {
              ends_expr_final.push_back(field);
            }
          }
        } else {
          ends_expr_final.push_back(e);
        }
      }
      if (auto* var_node = e.as<VarNode>()) {
        params_to_remove_.insert(GetRef<Var>(var_node));
      }
    }

    if (ends_expr_final.size() > 1) {
      body = Tuple(ends_expr_final);
      bool need_on_device = false;
      // need to add on_device(dsp) on final tuple if some input is on dsp.
      for (auto end_expr : ends_expr_final) {
        if (auto* call_node = end_expr.as<CallNode>()) {
          if (call_node->op == Op::Get("on_device")) {
            need_on_device = true;
            break;
          }
        }
      }
      if (need_on_device) {
        auto attrs = make_object<OnDeviceAttrs>();
        attrs->se_scope = SEScope::ForDeviceType(kDLEdgeX);
        static const Op& op = Op::Get("on_device");
        body = Call(op, {body}, Attrs(attrs), {});
      }
    } else if (ends_expr_final.size() == 1) {
      body = ends_expr_final[0];
    } else {
      LOG(FATAL) << "Miss end of subgraph.";
    }

    Array<Var> params_final;
    for (auto param : params_) {
      if (params_to_remove_.count(param) == 0) {
        params_final.push_back(param);
      }
    }

    return Function(params_final, body, Type(), {});
  }

  // Transform calls.
  Expr VisitExpr_(const CallNode* call) {
    if (call->op.as<OpNode>()) {
      size_t index = graph_.index_of(call);
      Array<Expr> new_args = GetNewArguments(call->args, index);
      return Call(call->op, new_args, call->attrs, call->type_args);
    } else {
      return ExprMutator::VisitExpr_(call);
    }
  }

  Expr VisitExpr_(const TupleNode* tuple) {
    // This tuple is an intermediate node in the group
    size_t index = graph_.index_of(tuple);
    Array<Expr> new_fields = GetNewArguments(tuple->fields, index);
    return Tuple(new_fields);
  }

  Expr VisitExpr_(const TupleGetItemNode* tuple_get) {
    size_t index = graph_.index_of(tuple_get);
    auto new_tuple = GetNewArguments({tuple_get->tuple}, index)[0];
    return TupleGetItem(new_tuple, tuple_get->index);
  }

  Array<Expr> GetNewArguments(const tvm::Array<Expr>& args, int current_index) {
    Array<Expr> new_args;
    for (auto arg : args) {
      auto type = arg->checked_type();
      Expr new_arg = this->Mutate(arg);
      if (is_main_node(arg.get())) {
        size_t arg_index = graph_.index_of(arg.get());
        if (!IdSetFoundId(all_ids_, arg_index) &&
            IdSetFoundId(all_ids_, current_index)) {  // this arg is outside of the subgraph
          Var param = GetOrAllocParam(new_arg, type);
          new_args.push_back(param);
        } else {
          new_args.push_back(new_arg);
        }
      } else if (is_var_node(arg.get())) {
        if (IdSetFoundId(all_ids_, current_index) && IsVarInOriginalFunction(arg)) {
          Var param = GetOrAllocParam(new_arg, type);
          new_args.push_back(param);
        } else {
          new_args.push_back(new_arg);
        }
      } else {
        new_args.push_back(new_arg);
      }
    }
    return new_args;
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_CONTRIB_EDGEX_RELAY_TRANSFORMS_EDGEX_GRAPH_H_
