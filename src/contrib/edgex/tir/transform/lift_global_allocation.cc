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
 * \file decorate_device_scope.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../edgex_ir_utils.h"
#include "./edgex_transform.h"

namespace tvm {
namespace tir {

class GlobalAllocationLifter : public StmtExprMutator {
 private:
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::device_scope) {
      Stmt updated = StmtExprMutator::VisitStmt_(op);
      if (allocation_nests_.empty()) {
        return updated;
      }
      return MergeNest(allocation_nests_, updated);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AllocateNode* op) final {
    auto storage_scope = GetStorageScope(op->buffer_var);
    if (storage_scope.rank == runtime::StorageRank::kGlobal) {
      auto n = CopyOnWrite(op);
      Stmt body = std::move(n->body);
      allocation_nests_.push_back(Stmt(n));
      return VisitStmt(body);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  std::unordered_set<const VarNode*> device_storage_scope_vars_;
  std::vector<Stmt> allocation_nests_;
};

namespace transform {

Pass LiftGlobalAllocation() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = GlobalAllocationLifter()(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.edgex.LiftGlobalAllocation", {});
}

TVM_REGISTER_GLOBAL("tir.edgex.transform.LiftGlobalAllocation")
    .set_body_typed(LiftGlobalAllocation);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
