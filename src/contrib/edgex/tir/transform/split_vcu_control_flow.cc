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
 * \file split_vcu_control_flow.cc
 * \brief Split vcu/cu code segments.
 */
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../attrs.h"
#include "../edgex_ir_utils.h"
#include "../op/builtin.h"
#include "./edgex_transform.h"

#define VCU_NUM 4

namespace tvm {
namespace tir {

using CodeState = tvm::tir::edgex::NNPUnitKind;
using runtime::StorageRank;
using runtime::StorageScope;

template <typename R, typename... Args>
class StmtExprFunctor : public StmtFunctor<R(const Stmt&, Args...)>,
                        public ExprFunctor<R(const PrimExpr&, Args...)> {
 public:
  using StmtFunctorT = StmtFunctor<R(const Stmt&, Args...)>;
  using ExprFunctorT = ExprFunctor<R(const PrimExpr&, Args...)>;
  using StmtFunctorT::operator();
  using ExprFunctorT::operator();

 protected:
  using ExprFunctorT::VisitExpr;
  using StmtFunctorT::VisitStmt;

  R VisitExpr(const PrimExpr& e, Args&&... args) override {
    return ExprFunctorT::VisitExpr(e, std::forward<Args>(args)...);
  }
};

#define UNSUPPORTED_STMT(T)                           \
  CodeState VisitStmt_(const T* op) final {           \
    LOG(FATAL) << "Do not support " << #T << " stmt"; \
    return CodeState::ALL;                            \
  }

#define UNSUPPORTED_EXPR(T)                           \
  CodeState VisitExpr_(const T* op) final {           \
    LOG(FATAL) << "Do not support " << #T << " expr"; \
    return CodeState::ALL;                            \
  }

#define DEFAULT_STATE_EXPR(T) \
  CodeState VisitExpr_(const T* op) final { return CodeState::ALL; }

#define BINARY_STATE_EXPR(T)                                   \
  CodeState VisitExpr_(const T* op) final {                    \
    return MergeExprState(VisitExpr(op->a), VisitExpr(op->b)); \
  }

class VcuControlFlowMarker : public StmtExprFunctor<CodeState> {
 public:
  /*! \brief query code state for stmt. */
  CodeState GetStmtState(const StmtNode* stmt) const {
    auto it = stmt_states_.find(stmt);
    if (it != stmt_states_.end()) {
      return it->second;
    } else {
      return CodeState::ALL;
    }
  }

  /*! \brief record whether there are vcu stmts in visited stmts. */
  bool HasVcuStmt() const { return has_vcu_; }

 private:
  CodeState VisitStmt_(const AttrStmtNode* op) final {
    CodeState value_state = VisitExpr(op->value);
    CodeState body_state = VisitStmt(op->body);
    if (value_state != CodeState::ALL) {
      CHECK_EQ(value_state, body_state);
      return MarkStmt(op, value_state);
    } else {
      return MarkStmt(op, body_state);
    }
  }

  CodeState VisitStmt_(const IfThenElseNode* op) final {
    CodeState cond_state = VisitExpr(op->condition);
    CodeState left_state = VisitStmt(op->then_case);
    CodeState right_state = op->else_case.defined() ? VisitStmt(op->else_case) : CodeState::ALL;
    if (cond_state != CodeState::ALL) {
      CHECK_EQ(cond_state, left_state);
      CHECK_EQ(cond_state, right_state);
      return MarkStmt(op, cond_state);
    }
    return MarkStmt(op, left_state == right_state ? left_state : CodeState::ALL);
  }

  CodeState VisitStmt_(const LetStmtNode* op) final {
    CodeState value_state = VisitExpr(op->value);
    MarkVar(op->var.get(), value_state);
    CodeState body_state = VisitStmt(op->body);
    ResetVar(op->var.get());
    return MarkStmt(op, body_state);
  }

  CodeState VisitStmt_(const ForNode* op) final {
    CodeState min_state = VisitExpr(op->min);
    CodeState ext_state = VisitExpr(op->extent);
    CodeState body_state = VisitStmt(op->body);
    if (min_state != CodeState::ALL || ext_state != CodeState::ALL) {
      CHECK_EQ(min_state, ext_state);
      CHECK_EQ(min_state, body_state);
      return MarkStmt(op, min_state);
    }
    return MarkStmt(op, body_state);
  }

  CodeState VisitStmt_(const AllocateNode* op) final {
    // allocation should dominate cu/vcu blocks, we do not process it here
    VisitStmt(op->body);
    return MarkStmt(op, CodeState::ALL);
  }

  CodeState VisitStmt_(const StoreNode* op) final {
    CodeState value_state = VisitExpr(op->value);
    CodeState index_state = VisitExpr(op->index);
    CodeState pred_state = VisitExpr(op->predicate);
    if (GetStorageScope(op->buffer_var).rank == StorageRank::kVM) {
      CHECK_NE(value_state, CodeState::CU);
      CHECK_NE(index_state, CodeState::CU);
      CHECK_NE(pred_state, CodeState::CU);
      return MarkStmt(op, CodeState::VCU);
    } else {
      CHECK_NE(value_state, CodeState::VCU);
      CHECK_NE(index_state, CodeState::VCU);
      CHECK_NE(pred_state, CodeState::VCU);
      return MarkStmt(op, CodeState::CU);
    }
  }

  CodeState VisitStmt_(const AssertStmtNode* op) final {
    CodeState cond_state = VisitExpr(op->condition);
    CodeState body_state = VisitStmt(op->body);
    if (cond_state != CodeState::ALL) {
      CHECK_EQ(cond_state, body_state);
      return MarkStmt(op, cond_state);
    }
    return MarkStmt(op, body_state);
  }

  CodeState VisitStmt_(const SeqStmtNode* op) final {
    bool has_vcu = false;
    bool has_cu = false;
    bool has_all = false;
    for (size_t i = 0; i < op->size(); ++i) {
      CodeState state = VisitStmt(op->operator[](i));
      has_vcu |= state == CodeState::VCU;
      has_cu |= state == CodeState::CU;
      has_all |= state == CodeState::ALL;
    }
    if (has_all || (has_vcu && has_cu)) {
      return MarkStmt(op, CodeState::ALL);
    } else if (has_vcu) {
      return MarkStmt(op, CodeState::VCU);
    } else {
      return MarkStmt(op, CodeState::CU);
    }
  }

  CodeState VisitStmt_(const EvaluateNode* op) final { return MarkStmt(op, VisitExpr(op->value)); }

  CodeState VisitExpr_(const VarNode* op) final { return GetVarState(op); }

  CodeState VisitExpr_(const LoadNode* op) final {
    CodeState index_state = VisitExpr(op->index);
    CodeState pred_state = VisitExpr(op->predicate);
    if (GetStorageScope(op->buffer_var).rank == StorageRank::kVM) {
      CHECK_NE(index_state, CodeState::CU);
      CHECK_NE(pred_state, CodeState::CU);
      return CodeState::VCU;
    } else {
      CHECK_NE(index_state, CodeState::VCU);
      CHECK_NE(pred_state, CodeState::VCU);
      return CodeState::CU;
    }
  }

  CodeState VisitExpr_(const BufferLoadNode* op) final {
    if (op->buffer.scope() == "vm") {
      return CodeState::VCU;
    } else {
      return CodeState::CU;
    }
  }

  CodeState VisitExpr_(const LetNode* op) final {
    CodeState value_state = VisitExpr(op->value);
    MarkVar(op->var.get(), value_state);
    CodeState body_state = VisitExpr(op->body);
    ResetVar(op->var.get());
    return body_state;
  }

  CodeState VisitExpr_(const CallNode* call) final {
    CodeState state = CodeState::ALL;
    for (const PrimExpr& e : call->args) {
      state = MergeExprState(state, VisitExpr(e));
    }
    if (call->op->IsInstance<OpNode>()) {
      const Op& op = Downcast<Op>(call->op);
      std::string name = op->name;
      if (name.find("nnp_") != std::string::npos) {
        // return registered state for op if it is on CU or VCU
        static auto unit_kind_dict_ = Op::GetAttrMap<edgex::TNNPUnitKind>("TNNPUnitKind");
        CodeState op_unit = (CodeState)unit_kind_dict_.get(op, Integer(CodeState::ALL))->value;
        if (op_unit != CodeState::ALL) {
          CHECK((op_unit == CodeState::CU && state != CodeState::VCU) ||
                (op_unit == CodeState::VCU && state != CodeState::CU));
          return op_unit;
        }
        // nnp_sync
        if (name.find("nnp_sync") != std::string::npos) {
          std::string sync_bb_name;
          if (auto bb_node = call->args[0].as<StringImmNode>()) {
            sync_bb_name = bb_node->value;
          }
          if (sync_bb_name.find("vidma") != std::string::npos ||
              sync_bb_name.find("vodma") != std::string::npos ||
              sync_bb_name.find("vcu") != std::string::npos) {
            return CodeState::VCU;
          } else {
            return CodeState::CU;
          }
        }
        // by default put on compute unit same with inputs
        // or put on cu if inputs state not specified.
        return state == CodeState::VCU ? state : CodeState::CU;
      }
    }
    return state;
  }

  BINARY_STATE_EXPR(AddNode);
  BINARY_STATE_EXPR(SubNode);
  BINARY_STATE_EXPR(MulNode);
  BINARY_STATE_EXPR(DivNode);
  BINARY_STATE_EXPR(ModNode);
  BINARY_STATE_EXPR(FloorDivNode);
  BINARY_STATE_EXPR(FloorModNode);
  BINARY_STATE_EXPR(MinNode);
  BINARY_STATE_EXPR(MaxNode);
  BINARY_STATE_EXPR(EQNode);
  BINARY_STATE_EXPR(NENode);
  BINARY_STATE_EXPR(LTNode);
  BINARY_STATE_EXPR(LENode);
  BINARY_STATE_EXPR(GTNode);
  BINARY_STATE_EXPR(GENode);
  BINARY_STATE_EXPR(AndNode);
  BINARY_STATE_EXPR(OrNode);

  CodeState VisitExpr_(const ReduceNode* op) final {
    CodeState state = VisitExpr(op->condition);
    for (const PrimExpr& v : op->init) {
      state = MergeExprState(state, VisitExpr(v));
    }
    for (const PrimExpr& v : op->source) {
      state = MergeExprState(state, VisitExpr(v));
    }
    return state;
  }

  CodeState VisitExpr_(const CastNode* op) final { return VisitExpr(op->value); }

  CodeState VisitExpr_(const NotNode* op) final { return VisitExpr(op->a); }

  CodeState VisitExpr_(const SelectNode* op) final {
    CodeState state = VisitExpr(op->condition);
    state = MergeExprState(state, VisitExpr(op->true_value));
    state = MergeExprState(state, VisitExpr(op->false_value));
    return state;
  }

  CodeState VisitExpr_(const RampNode* op) final {
    CodeState left_state = VisitExpr(op->base);
    CodeState right_state = VisitExpr(op->stride);
    return MergeExprState(left_state, right_state);
  }

  CodeState VisitExpr_(const BroadcastNode* op) final { return VisitExpr(op->value); }

  CodeState VisitExpr_(const ShuffleNode* op) final {
    CodeState state = CodeState::ALL;
    for (const PrimExpr& v : op->vectors) {
      state = MergeExprState(state, VisitExpr(v));
    }
    for (const PrimExpr& v : op->indices) {
      state = MergeExprState(state, VisitExpr(v));
    }
    return state;
  }

  UNSUPPORTED_STMT(WhileNode);
  UNSUPPORTED_STMT(BufferStoreNode);
  UNSUPPORTED_STMT(BufferRealizeNode);
  UNSUPPORTED_STMT(ProducerStoreNode);
  UNSUPPORTED_STMT(ProducerRealizeNode);
  UNSUPPORTED_STMT(PrefetchNode);
  UNSUPPORTED_STMT(BlockNode);
  UNSUPPORTED_STMT(BlockRealizeNode);

  UNSUPPORTED_EXPR(SizeVarNode);
  UNSUPPORTED_EXPR(ProducerLoadNode);
  DEFAULT_STATE_EXPR(AnyNode);
  DEFAULT_STATE_EXPR(IntImmNode);
  DEFAULT_STATE_EXPR(FloatImmNode);
  DEFAULT_STATE_EXPR(StringImmNode);

  CodeState MarkStmt(const StmtNode* stmt, CodeState s) {
    stmt_states_[stmt] = s;
    if (s == CodeState::VCU) {
      has_vcu_ = true;
    }
    return s;
  }
  CodeState MarkVar(const VarNode* var, CodeState s) {
    var_states_[var] = s;
    return s;
  }
  void ResetVar(const VarNode* var) { var_states_.erase(var); }
  CodeState GetVarState(const VarNode* var) {
    auto it = var_states_.find(var);
    if (it != var_states_.end()) {
      return it->second;
    } else {
      return CodeState::ALL;
    }
  }
  CodeState MergeExprState(CodeState s1, CodeState s2) {
    if (s1 != CodeState::ALL) {
      ICHECK(s2 == CodeState::ALL || s2 == s1) << "Merge inconsistent states";
      return s1;
    } else {
      return s2;
    }
  }

  /*! \brief denote whether vcu stmts exist. */
  bool has_vcu_ = false;

  /*! \brief map stmt to cu/vcu state. */
  std::map<const StmtNode*, CodeState> stmt_states_;

  /*! \brief map var to cu/vcu state. */
  std::map<const VarNode*, CodeState> var_states_;
};

class VcuControlFlowSplitter : public StmtExprMutator {
 public:
  Stmt Rewrite(Stmt stmt) {
    marker_(stmt);
    if (!marker_.HasVcuStmt()) {
      if (input_buffer_rebind_stmt_.defined()) {
        return SeqStmt::Flatten(input_buffer_rebind_stmt_, stmt);
      } else {
        return stmt;
      }
    }
    return VisitStmt(stmt);
  }

  bool HasVcuStmt() const { return marker_.HasVcuStmt(); }

  Stmt VisitStmt_(const AllocateNode* op) {
    auto storage_scope = GetStorageScope(op->buffer_var);
    if (storage_scope.rank == StorageRank::kVM) {
      ICHECK(!reach_vm_alloc_) << "VM memory buffer should be unique";
      reach_vm_alloc_ = true;
    } else if (storage_scope.rank == StorageRank::kDM) {
      ICHECK(!reach_dm_alloc_) << "DM memory buffer should be unique";
      reach_dm_alloc_ = true;
    }
    if (reach_vm_alloc_ && reach_dm_alloc_) {
      auto n = CopyOnWrite(op);
      n->body = DoSplit(op->body);
      return std::move(Allocate(n));
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  void SetInputBufferRebindStmt(Stmt stmt) { input_buffer_rebind_stmt_ = std::move(stmt); }

 private:
  class CodeEliminator : public StmtMutator {
   public:
    CodeEliminator(CodeState target, const VcuControlFlowMarker& marker)
        : target_(target), marker_(marker) {}
    Stmt VisitStmt(const Stmt& stmt) override {
      CodeState s = marker_.GetStmtState(stmt.get());
      if (s != CodeState::ALL && s != target_) {
        return std::move(Evaluate(0));
      } else {
        return StmtMutator::VisitStmt(stmt);
      }
    }

   private:
    CodeState target_;
    const VcuControlFlowMarker& marker_;
  };

  Stmt DoSplit(const Stmt& body) {
    // split cu/vcu block
    Stmt cu_block = CodeEliminator(CodeState::CU, marker_)(body);
    Stmt vcu_block = CodeEliminator(CodeState::VCU, marker_)(body);

    // acquire and release vcu resources
    Stmt lock_vcu = Evaluate(Call(DataType::Void(), edgex::builtin::nnp_lock_vcu(), {}));
    Stmt unlock_vcu = Evaluate(Call(DataType::Void(), edgex::builtin::nnp_unlock_vcu(), {}));

    // update for iss input buffer convention
    if (input_buffer_rebind_stmt_.defined()) {
      cu_block = SeqStmt::Flatten(lock_vcu, input_buffer_rebind_stmt_, cu_block);
    } else {
      cu_block = SeqStmt::Flatten(lock_vcu, cu_block);
    }
    vcu_block = SeqStmt::Flatten(vcu_block, unlock_vcu);

    PrimExpr cuid = Call(DataType::Int(32), edgex::builtin::nnp_cuid(), {});
    PrimExpr cond = cuid >= VCU_NUM;
    Stmt new_body = IfThenElse(std::move(cond), cu_block, vcu_block);
    return ConvertSSA(new_body);
  }

  /*! \brief cu/vcu state marker. */
  VcuControlFlowMarker marker_;

  /*! \brief record current dm buffer var. */
  const VarNode* dm_buf_ = nullptr;

  /*! \brief record current vm buffer var. */
  const VarNode* vm_buf_ = nullptr;

  /*! \brief whether a vm scope allocate is encountered. */
  bool reach_vm_alloc_ = false;

  /*! \brief whether a dm scope allocate is encountered. */
  bool reach_dm_alloc_ = false;

  /*! \brief optional buffer rebind stmt set by caller. */
  Stmt input_buffer_rebind_stmt_;
};

namespace transform {

Pass SplitVcuControlFlow() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();

    // deal with iss input buffer convention
    Stmt rebind_input_buffer =
        Evaluate(Call(DataType::Void(), edgex::builtin::nnp_iss_bind_input_buffer(), {}));
    VcuControlFlowSplitter splitter;
    splitter.SetInputBufferRebindStmt(rebind_input_buffer);
    n->body = splitter.Rewrite(std::move(n->body));
    if (splitter.HasVcuStmt()) {
      n->body = AttrStmt(ObjectRef(), attr::nnp_vcore_resource, make_const(DataType::Int(32), 0x01),
                         n->body);
    }
    return f;
  };
  Pass split_pass = CreatePrimFuncPass(pass_func, 0, "tir.edgex.SplitVcuControlFlow", {});
  return Sequential({split_pass, RemoveNoOp()});
}

TVM_REGISTER_GLOBAL("tir.edgex.transform.SplitVcuControlFlow").set_body_typed(SplitVcuControlFlow);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
