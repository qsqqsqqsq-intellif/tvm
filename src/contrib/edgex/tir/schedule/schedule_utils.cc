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
#include "./schedule_utils.h"

#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tir {

Stmt Substitute(const Stmt& stmt, const Map<Stmt, Stmt>& replace_plan) {
  class Mutator : public StmtMutator {
   public:
    explicit Mutator(const Map<Stmt, Stmt>& replace_plan) : replace_plan(replace_plan) {}
    Stmt VisitStmt(const Stmt& stmt) override {
      auto it = replace_plan.find(stmt);
      if (it == replace_plan.end()) {
        return StmtMutator::VisitStmt(stmt);
      } else {
        return StmtMutator::VisitStmt((*it).second);
      }
    }
    const Map<Stmt, Stmt>& replace_plan;
  };
  return Mutator(replace_plan)(stmt);
}

namespace schedule {

For WithAnnotation(const ForNode* loop, const String& attr_key, const ObjectRef& attr_value) {
  Map<String, ObjectRef> annotations = loop->annotations;
  annotations.Set(attr_key, attr_value);
  ObjectPtr<ForNode> new_loop = make_object<ForNode>(*loop);
  new_loop->annotations = std::move(annotations);
  return For(new_loop);
}

StmtSRef LowestCommonAncestor(const std::vector<StmtSRef>& nodes, const StmtSRef& root) {
  // alg: count the visit times for each node from the bottom to the root
  ICHECK_GE(nodes.size(), 2);
  std::unordered_map<StmtSRef, size_t, ObjectHash, ObjectEqual> visit_cnt;

  auto f_visit = [&visit_cnt](const StmtSRef& node) {
    auto it = visit_cnt.find(node);
    if (it == visit_cnt.end()) {
      visit_cnt[node] = 1;
    } else {
      it->second++;
    }
  };

  for (auto node : nodes) {
    while (!node.same_as(root)) {
      f_visit(node);
      if (visit_cnt[node] == nodes.size()) {
        return node;
      }
      node = GetRef<StmtSRef>(node->parent);
    }
  }

  return root;
}

void ReplaceStmt(ScheduleState self, const StmtSRef& src, Stmt tgt_stmt,
                 const Map<Block, Block>& block_sref_reuse) {
  // Determine whether we can just replace on current src stmt
  Stmt src_stmt = GetRef<Stmt>(src->stmt);
  bool stmt_type_match =
      (src_stmt->IsInstance<ForNode>() && tgt_stmt->IsInstance<ForNode>()) ||
      (src_stmt->IsInstance<ForNode>() && tgt_stmt->IsInstance<BlockRealizeNode>()) ||
      (src_stmt->IsInstance<BlockNode>() && tgt_stmt->IsInstance<BlockNode>());
  if (stmt_type_match) {
    self->Replace(src, tgt_stmt, block_sref_reuse);
    return;
  }

  // Try replace on parent of src stmt
  StmtSRef parent_ref = GetRef<StmtSRef>(src->parent);
  Stmt parent = GetRef<Stmt>(parent_ref->stmt);
  Map<Stmt, Stmt> stmt_map;
  stmt_map.Set(src_stmt, tgt_stmt);
  Stmt new_parent = tvm::tir::Substitute(parent, stmt_map);
  self->Replace(parent_ref, new_parent, block_sref_reuse);
}

}  // namespace schedule
}  // namespace tir
}  // namespace tvm
