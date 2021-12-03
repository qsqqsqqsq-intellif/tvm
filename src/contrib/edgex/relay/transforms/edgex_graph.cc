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
 * \file edgex_graph.cc
 *
 * \brief ...
 */

#include "edgex_graph.h"

#include <set>
#include <unordered_set>

namespace tvm {
namespace relay {
std::set<std::string> injective_ops = {"tanh",
                                       "sigmoid",
                                       "exp",
                                       "nn.softmax",
                                       "nn.log_softmax",
                                       "vision.multibox_prior",
                                       "vision.multibox_transform_loc",
                                       "vision.get_valid_counts",
                                       "vision.non_max_suppression",
                                       "vision.proposal",
                                       "vision.roi_pool",
                                       "nn.lrn",
                                       "argmax",
                                       "argmin",
                                       "mean"};

std::set<std::string> opaque_ops = {"reshape", "concatenate", "squeeze",
                                    "contrib.adaptive_avg_pool2d", "contrib.adaptive_max_pool2d"};

// Creator of post dominator tree of the dataflow
class EdgexIndexedForwardGraph::Creator : private ExprVisitor {
 public:
  explicit Creator(support::Arena* arena) : arena_(arena) {}

  EdgexIndexedForwardGraph Prepare(const Expr& body) {
    this->Update(body, nullptr, kOpaque);
    this->VisitExpr(body);
    return std::move(graph_);
  }

 private:
  /*! \brief allocator of all the internal node object */
  support::Arena* arena_;
  // The output.
  EdgexIndexedForwardGraph graph_;
  // attribute equal comparator
  StructuralEqual attr_equal_;
  // Update the message stored at the node.
  void Update(const Expr& node, EdgexIndexedForwardGraph::Node* parent, OpPatternKind pattern) {
    const tvm::Object* key = node.get();
    EdgexIndexedForwardGraph::Node* current;
    auto it = graph_.node_map.find(key);
    if (it != graph_.node_map.end()) {
      current = it->second;
    } else {
      current = arena_->make<EdgexIndexedForwardGraph::Node>();
      graph_.node_map[key] = current;
    }
    if (parent != nullptr) {
      auto* link_out = arena_->make<LinkNode<EdgexIndexedForwardGraph::Edge>>();
      link_out->value.node = parent;
      link_out->value.pattern = pattern;
      current->outputs.Push(link_out);

      auto* link_in = arena_->make<LinkNode<EdgexIndexedForwardGraph::Edge>>();
      link_in->value.node = current;
      link_in->value.pattern = kOpaque;
      parent->inputs.Push(link_in);
    } else {
      current->extern_ref = true;
    }
  }

  void AddNode(const tvm::Object* key) {
    auto it = graph_.node_map.find(key);
    CHECK(it != graph_.node_map.end()) << "Cannot find node " << GetRef<ObjectRef>(key);
    EdgexIndexedForwardGraph::Node* node = it->second;
    CHECK(node->ref == nullptr);
    node->ref = key;
    node->index = graph_.post_dfs_order.size();
    graph_.post_dfs_order.push_back(node);
  }

  // Post order tree
  void VisitExpr_(const FunctionNode* op) final {
    for (auto param : op->params) {
      this->Update(param, nullptr, kOpaque);
    }
    this->Update(op->body, nullptr, kOpaque);
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const ConstantNode* op) final {
    this->AddNode(op);
    Node* node = graph_.node_map.at(op);
    DataType dtype = DataType(op->data->dtype);
    // This rule must be consistent with code generator.
    bool is_simple_const =
        (dtype == DataType::Int(32) || dtype == DataType::Int(64) || dtype == DataType::Float(32) ||
         dtype == DataType::Float(64) || dtype == DataType::Bool());
    if (op->is_scalar() && is_simple_const) {
      node->pattern = kElemWise;
    } else {
      // for now, mark non-scalar constant
      // as opaque, we will not choose to fuse it.
      node->pattern = kElemWise;  // can merge to its parent for NNP200
    }
  }

  void VisitExpr_(const CallNode* call) final {
    CHECK(graph_.node_map.count(call));
    Node* node = graph_.node_map.at(call);
    static auto fpattern = Op::GetAttrMap<TOpPattern>("TOpPattern");
    // Now we set the pattern of this call.
    //
    // If we see a call mentioning an operator we should mark it with its
    // annotated pattern.
    //
    // If the pattern is not annotated we will default to opaque.
    //
    // Finally if the operator position is not a call node we will
    // need to call Update, as it may be an arbitrary expression.
    OpPatternKind op_pattern = kOpaque;
    if (const OpNode* opnode = call->op.as<OpNode>()) {
      op_pattern = static_cast<OpPatternKind>(fpattern[GetRef<Op>(opnode)]);
      auto op_name = opnode->name;
      if (injective_ops.count(op_name)) {
        op_pattern = kInjective;
      }
      if (opaque_ops.count(op_name)) {
        op_pattern = kOpaque;
      }
    } else {
      this->Update(call->op, node, kOpaque);
    }

    node->pattern = op_pattern;
    this->Update(call->op, nullptr, kOpaque);

    const auto* rtype = call->checked_type().as<TensorTypeNode>();
    // int input_op_num = 0;
    // pass the message back to all the children it references.
    for (size_t i = 0; i < call->args.size(); ++i) {
      const auto* arg_type = call->args[i]->checked_type().as<TensorTypeNode>();
      // specifically check if result type
      OpPatternKind edge_pattern = op_pattern;
      if (edge_pattern == kBroadcast && arg_type != nullptr && rtype != nullptr &&
          attr_equal_(rtype->shape, arg_type->shape)) {
        edge_pattern = kElemWise;
      }
      // if (call->args[i].as<CallNode>() || call->args[i].as<TupleNode>() ||
      //     call->args[i].as<TupleGetItemNode>()) {
      //   input_op_num++;
      // }
      this->Update(call->args[i], node, edge_pattern);
    }
    // node->input_op_num = input_op_num;
    ExprVisitor::VisitExpr_(call);
    this->AddNode(call);
  }

  void VisitExpr_(const TupleNode* op) final {
    CHECK(graph_.node_map.count(op));
    Node* tuple_node = graph_.node_map.at(op);
    tuple_node->pattern = kTuple;
    // int input_op_num = 0;
    for (const Expr& field : op->fields) {
      // if (field.as<CallNode>() || field.as<TupleNode>() || field.as<TupleGetItemNode>()) {
      //   input_op_num++;
      // }

      if (field->checked_type().as<TensorTypeNode>()) {
        this->Update(field, tuple_node, kInjective);
      } else {
        this->Update(field, nullptr, kOpaque);
      }
    }
    // tuple_node->input_op_num = input_op_num;
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }

  void VisitExpr_(const TupleGetItemNode* op) final {
    auto tuple_type = op->tuple->checked_type().as<TupleTypeNode>();
    CHECK(tuple_type);
    // when TVM lowers a fused function, it expects all arguments to be a Tensor or
    // a tuple containing only Tensors. But this tuple may contain a reference or
    // another tuple. To avoid modifying codegen logic, we do not allow fusing through this node
    // if the tuple contains such non Tensor fields. However, all fields will be recursively
    // visited via call to ExprVisitor::VisitExpr_(op) below and corresponding visitor methods.
    bool has_non_tensor = false;
    for (auto ty : tuple_type->fields) {
      if (!ty.as<TensorTypeNode>()) {
        has_non_tensor = true;
        break;
      }
    }
    if (has_non_tensor) {
      this->Update(op->tuple, nullptr, kOpaque);
    } else {
      CHECK(graph_.node_map.count(op));
      Node* node = graph_.node_map.at(op);
      node->pattern = kInjective;
      // node->input_op_num = 1;
      this->Update(op->tuple, node, kInjective);
    }
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }

  void VisitExpr_(const VarNode* op) final { this->AddNode(op); }

  void VisitExpr_(const LetNode* op) final {
    // do not fuse through let.
    this->Update(op->var, nullptr, kOpaque);
    this->Update(op->value, nullptr, kOpaque);
    this->Update(op->body, nullptr, kOpaque);
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }

  void VisitExpr_(const IfNode* op) final {
    // do not fuse through if.
    this->Update(op->cond, nullptr, kOpaque);
    this->Update(op->true_branch, nullptr, kOpaque);
    this->Update(op->false_branch, nullptr, kOpaque);
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }

  void VisitExpr_(const RefCreateNode* op) final {
    this->Update(op->value, nullptr, kOpaque);
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }

  void VisitExpr_(const RefReadNode* op) final {
    this->Update(op->ref, nullptr, kOpaque);
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }

  void VisitExpr_(const RefWriteNode* op) final {
    this->Update(op->ref, nullptr, kOpaque);
    this->Update(op->value, nullptr, kOpaque);
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }

  void VisitExpr_(const MatchNode* op) final {
    this->Update(op->data, nullptr, kOpaque);
    for (const Clause& c : op->clauses) {
      this->Update(c->rhs, nullptr, kOpaque);
    }
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }
};

EdgexIndexedForwardGraph EdgexIndexedForwardGraph::Create(support::Arena* arena, const Expr& body) {
  return Creator(arena).Prepare(body);
}

EdgexDominatorTree EdgexDominatorTree::PostDom(support::Arena* arena,
                                               const EdgexIndexedForwardGraph& graph) {
  EdgexDominatorTree tree;
  tree.nodes.resize(graph.post_dfs_order.size(), nullptr);
  // reverse topo order
  for (size_t i = graph.post_dfs_order.size(); i != 0; --i) {
    size_t index = i - 1;
    Node* tnode = arena->make<Node>();
    auto* gnode = graph.post_dfs_order[index];
    tnode->gnode = gnode;
    if (gnode->extern_ref) {
      tnode->depth = 1;
      tnode->parent = nullptr;
      tnode->pattern = kOpaque;
    } else {
      // find the LCAs of all outputs.
      OpPatternKind pattern = kElemWise;
      Node* parent = nullptr;
      for (auto link = gnode->outputs.head; link != nullptr; link = link->next) {
        size_t oindex = link->value.node->index;
        CHECK_LT(oindex, tree.nodes.size());
        Node* onode = tree.nodes[oindex];
        CHECK(onode != nullptr);
        if (parent != nullptr) {
          parent = LeastCommonAncestor(parent, onode, &pattern);
        } else {
          parent = onode;
        }
        pattern = CombinePattern(pattern, link->value.pattern);
      }
      tnode->depth = parent ? parent->depth + 1 : 1;
      tnode->parent = parent;
      tnode->pattern = pattern;
    }
    tree.nodes[index] = tnode;
  }
  return tree;
}

// Creator of EdgexDependencyGraph
class EdgexDependencyGraph::Creator : private ExprFunctor<void(const Expr& e)> {
 public:
  explicit Creator(support::Arena* arena) : arena_(arena) {}

  EdgexDependencyGraph Create(const Expr& body) {
    this->VisitExpr(body);
    return std::move(graph_);
  }

 private:
  /*! \brief allocator of all the internal node object */
  support::Arena* arena_;
  // The output.
  EdgexDependencyGraph graph_;
  // Update the message stored at the node.
  void Depend(EdgexDependencyGraph::Node* parent, const Expr& child) {
    VisitExpr(child);

    if (graph_.found_node(child.get())) {
      Depend(parent, graph_.node_of(child.get()));
    }
  }

  void Depend(EdgexDependencyGraph::Node* parent, EdgexDependencyGraph::Node* child) {
    auto* parent_link = arena_->make<LinkNode<EdgexDependencyGraph::Node*>>();
    parent_link->value = parent;
    child->parents.Push(parent_link);

    auto* child_link = arena_->make<LinkNode<EdgexDependencyGraph::Node*>>();
    child_link->value = child;
    parent->children.Push(child_link);

    // fuse ops: rearrange Op::add and its operands next by next
    auto callnode_p = GetRef<ObjectRef>(parent->tvm_node).as<CallNode>();
    auto callnode_c = GetRef<ObjectRef>(child->tvm_node).as<CallNode>();
    if (child->parents_size() < 2 && callnode_p && callnode_p->op.same_as(Op::Get("add")) &&
        callnode_c && callnode_c->op.same_as(Op::Get("cast"))) {
      graph_.pop_back();
    }
    if (callnode_c && callnode_c->op.same_as(Op::Get("add"))) {
      graph_.pop_back();
      for (auto n = child->children.head; n != nullptr; n = n->next) {
        auto cn = GetRef<ObjectRef>(n->value->tvm_node).as<CallNode>();
        if (n->value->parents_size() < 2 && cn && cn->op.same_as(Op::Get("cast")))
          graph_.push_back(n->value);
      }
      graph_.push_back(child);
    }
  }

  std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> visited_;

  EdgexDependencyGraph::Node* NewNode(const Expr& e) {
    auto* ret = arena_->make<EdgexDependencyGraph::Node>();
    auto node = const_cast<tvm::RelayExprNode*>(e.get());
    ret->tvm_node = static_cast<tvm::Object*>(node);
    return ret;
  }

  void VisitExpr(const Expr& e) final {
    if (visited_.count(e) == 0) {
      // NOTE(cww): ignore Constant and Var and Function
      if (graph_.found_node(e.get()) == false && e.as<ConstantNode>() == nullptr &&
          e.as<VarNode>() == nullptr && e.as<FunctionNode>() == nullptr) {
        EdgexDependencyGraph::Node* node = NewNode(e);
        graph_.set_map(node);

        visited_.insert(e);
        ExprFunctor<void(const Expr&)>::VisitExpr(e);

        graph_.push_back(node);
      } else {
        visited_.insert(e);
        ExprFunctor<void(const Expr&)>::VisitExpr(e);
      }
    }
  }

  void VisitExpr_(const CallNode* c) final {
    EdgexDependencyGraph::Node* n = graph_.node_of(c);
    // Depend(n, c->op);  // NOTE(cww): ignore OpNode
    for (const auto& a : c->args) {
      Depend(n, a);
    }
  }

  void VisitExpr_(const TupleNode* t) final {
    EdgexDependencyGraph::Node* n = graph_.node_of(t);
    for (const auto& a : t->fields) {
      Depend(n, a);
    }
  }

  void VisitExpr_(const TupleGetItemNode* t) final {
    EdgexDependencyGraph::Node* n = graph_.node_of(t);
    Depend(n, t->tuple);
  }

  void VisitExpr_(const RefCreateNode* r) final {
    LOG(FATAL) << "Unsupported Relay IR: " << r->_type_key;
    EdgexDependencyGraph::Node* n = graph_.node_of(r);
    Depend(n, r->value);
  }

  void VisitExpr_(const RefReadNode* r) final {
    LOG(FATAL) << "Unsupported Relay IR: " << r->_type_key;
    EdgexDependencyGraph::Node* n = graph_.node_of(r);
    Depend(n, r->ref);
  }

  void VisitExpr_(const RefWriteNode* r) final {
    LOG(FATAL) << "Unsupported Relay IR: " << r->_type_key;
    EdgexDependencyGraph::Node* n = graph_.node_of(r);
    Depend(n, r->ref);
    Depend(n, r->value);
  }

  void VisitExpr_(const IfNode* i) final {
    LOG(FATAL) << "Unsupported Relay IR: " << i->_type_key;
    EdgexDependencyGraph::Node* n = graph_.node_of(i);
    Depend(n, i->cond);
    Depend(n, i->true_branch);
    Depend(n, i->false_branch);
  }

  void VisitExpr_(const FunctionNode* f) final { VisitExpr(f->body); }

  void VisitExpr_(const LetNode* l) final {
    LOG(FATAL) << "Unsupported Relay IR: " << l->_type_key;
    EdgexDependencyGraph::Node* n = graph_.node_of(l);
    Depend(n, l->value);
    Depend(n, l->body);
  }

  void VisitExpr_(const MatchNode* m) final {
    LOG(FATAL) << "Unsupported Relay IR: " << m->_type_key;
    EdgexDependencyGraph::Node* n = graph_.node_of(m);
    Depend(n, m->data);
    for (const Clause& c : m->clauses) {
      Depend(n, c->rhs);
    }
  }

  void VisitExpr_(const VarNode* v) final {}

  void VisitExpr_(const GlobalVarNode* v) final {
    LOG(FATAL) << "Unsupported Relay IR: " << v->_type_key;
  }

  void VisitExpr_(const ConstantNode* c) final {}

  void VisitExpr_(const OpNode* o) final {}

  void VisitExpr_(const ConstructorNode* c) final {
    LOG(FATAL) << "Unsupported Relay IR: " << c->_type_key;
  }
};

EdgexDependencyGraph EdgexDependencyGraph::Create(support::Arena* arena, const Expr& body) {
  return Creator(arena).Create(body);
}

}  // namespace relay
}  // namespace tvm
