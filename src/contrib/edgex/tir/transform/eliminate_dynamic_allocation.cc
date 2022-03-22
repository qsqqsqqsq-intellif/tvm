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
 * \file eliminate_loop_dynamic_allocation.cc
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../edgex_ir_utils.h"
#include "./edgex_transform.h"

namespace tvm {
namespace tir {

class DynamicAllocationEliminator : public StmtExprMutator {
 private:
  Stmt VisitStmt_(const ForNode* op) final {
    dom_map_.Set(op->loop_var, arith::IntSet::FromMinExtent(op->min, op->extent));
    auto res = StmtExprMutator::VisitStmt_(op);
    dom_map_.erase(op->loop_var);
    return res;
  }

  Stmt VisitStmt_(const AllocateNode* op) final {
    auto storage_scope = GetStorageScope(op->buffer_var);
    if (storage_scope.rank == runtime::StorageRank::kGlobal) {
      // we support dynamic global allocation
      return StmtExprMutator::VisitStmt_(op);
    }
    bool need_update = false;
    Array<PrimExpr> static_extents;
    for (const PrimExpr& e : op->extents) {
      PrimExpr extent = analyzer_.Simplify(e);
      if (is_const_int(extent)) {
        static_extents.push_back(extent);
        continue;
      }
      extent = arith::EvalSet(extent, dom_map_).max();
      extent = analyzer_.Simplify(extent);
      ICHECK(is_const_int(extent)) << "Dynamic allocation for " << storage_scope.to_string()
                                   << " of buffer " << op->buffer_var << ": " << e;
      static_extents.push_back(extent);
      need_update = true;
    }
    if (need_update) {
      auto n = CopyOnWrite(op);
      n->extents = static_extents;
      n->body = VisitStmt(n->body);
      return std::move(Allocate(n));
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  arith::Analyzer analyzer_;
  Map<Var, arith::IntSet> dom_map_;
};

namespace transform {

Pass EliminateDynamicAllocation() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = DynamicAllocationEliminator()(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.edgex.EliminateDynamicAllocation", {});
}

TVM_REGISTER_GLOBAL("tir.edgex.transform.EliminateDynamicAllocation")
    .set_body_typed(EliminateDynamicAllocation);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
