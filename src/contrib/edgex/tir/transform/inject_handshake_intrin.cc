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
 * \file inject_handshake_intrin.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"
#include "edgex_transform.h"

namespace tvm {
namespace tir {

class IntrinDetector : public StmtExprVisitor {
 public:
  void VisitStmt_(const EvaluateNode* op) final {
    if (op->value.as<CallNode>()->op.same_as(edgex::builtin::nnp_eidma_load())) {
      if (first_eidma_ == nullptr) {
        first_eidma_ = op;
      }
      // CHECK(eidma_depth_ < 0 || eidma_depth_ == cur_depth_);
      eidma_depth_ = cur_depth_;
    } else if (op->value.as<CallNode>()->op.same_as(edgex::builtin::nnp_eodma_store())) {
      if (last_eodma_ != op) {
        last_eodma_ = op;
      }
      // CHECK(eodma_depth_ < 0 || eodma_depth_ == cur_depth_);
      eodma_depth_ = cur_depth_;
    } else if (op->value.as<CallNode>()->op.same_as(edgex::builtin::nnp_ewdma_load())) {
      if (first_ewdma_ == nullptr) {
        first_ewdma_ = op;
      }
      // CHECK(ewdma_depth_ < 0 || ewdma_depth_ == cur_depth_);
      ewdma_depth_ = cur_depth_;
    } else if (op->value.as<CallNode>()->op.same_as(edgex::builtin::nnp_bdma_load())) {
      if (!has_bdma_) {
        has_bdma_ = true;
      }
      if (last_bdma_ != op) {
        last_bdma_ = op;
      }
      // CHECK(bdma_depth_ < 0 || bdma_depth_ == cur_depth_);
      bdma_depth_ = cur_depth_;
    } else if (op->value.as<CallNode>()->op.same_as(edgex::builtin::nnp_idma_load())) {
      if (!has_idma_) {
        has_idma_ = true;
      }
      if (last_idma_ != op) {
        last_idma_ = op;
      }
      // CHECK(idma_depth_ < 0 || idma_depth_ == cur_depth_);
      idma_depth_ = cur_depth_;
    } else if (op->value.as<CallNode>()->op.same_as(edgex::builtin::nnp_odma_store())) {
      if (!has_odma_) {
        has_odma_ = true;
      }
      if (last_odma_ != op) {
        last_odma_ = op;
      }
      // CHECK(odma_depth_ < 0 || odma_depth_ == cur_depth_);
      odma_depth_ = cur_depth_;
    } else if (op->value.as<CallNode>()->op.same_as(edgex::builtin::nnp_vidma_load())) {
      if (!has_vidma_) {
        has_vidma_ = true;
      }
      if (last_vidma_ != op) {
        last_vidma_ = op;
      }
      // CHECK(vidma_depth_ < 0 || vidma_depth_ == cur_depth_);
      vidma_depth_ = cur_depth_;
    } else if (op->value.as<CallNode>()->op.same_as(edgex::builtin::nnp_vodma_store())) {
      if (!has_vodma_) {
        has_vodma_ = true;
      }
      if (last_vodma_ != op) {
        last_vodma_ = op;
      }
      // CHECK(vodma_depth_ < 0 || vodma_depth_ == cur_depth_);
      vodma_depth_ = cur_depth_;
    } else if (op->value.as<CallNode>()->op.same_as(edgex::builtin::nnp_wdma_load())) {
      if (!has_wdma_) {
        has_wdma_ = true;
      }
      if (last_wdma_ != op) {
        last_wdma_ = op;
      }
      // CHECK(wdma_depth_ < 0 || wdma_depth_ == cur_depth_);
      wdma_depth_ = cur_depth_;
    }
  }

  void VisitStmt_(const ForNode* op) final {
    ++cur_depth_;
    StmtExprVisitor::VisitStmt_(op);
    --cur_depth_;
  }

  /*! \brief Detect the instrinsic whether is the first nnp_eidma_load. */
  bool IsFirstEidma(const EvaluateNode* op) { return op == first_eidma_; }
  /*! \brief Detect the instrinsic whether is the first nnp_ewdma_load. */
  bool IsFirstEwdma(const EvaluateNode* op) { return op == first_ewdma_; }
  /*! \brief Detect the instrinsic whether is the last nnp_bdma_load. */
  bool IsLastBdma(const EvaluateNode* op) { return op == last_bdma_; }
  /*! \brief Detect the instrinsic whether is the last nnp_eodma_store. */
  bool IsLastEodma(const EvaluateNode* op) { return op == last_eodma_; }
  /*! \brief Detect the instrinsic whether is the last nnp_idma_load. */
  bool IsLastIdma(const EvaluateNode* op) { return op == last_idma_; }
  /*! \brief Detect the instrinsic whether is the last nnp_odma_store. */
  bool IsLastOdma(const EvaluateNode* op) { return op == last_odma_; }
  /*! \brief Detect the instrinsic whether is the last nnp_vidma_load. */
  bool IsLastVidma(const EvaluateNode* op) { return op == last_vidma_; }
  /*! \brief Detect the instrinsic whether is the last nnp_vodma_store. */
  bool IsLastVodma(const EvaluateNode* op) { return op == last_vodma_; }
  /*! \brief Detect the instrinsic whether is the last nnp_wdma_load. */
  bool IsLastWdma(const EvaluateNode* op) { return op == last_wdma_; }
  /*! \brief Get bdma loop depth. */
  int GetBdmaDepth() const { return bdma_depth_; }
  /*! \brief Get eidma loop depth. */
  int GetEidmaDepth() const { return eidma_depth_; }
  /*! \brief Get eodma loop depth. */
  int GetEodmaDepth() const { return eodma_depth_; }
  /*! \brief Get ewdma loop depth. */
  int GetEwdmaDepth() const { return ewdma_depth_; }
  /*! \brief Get idma loop depth. */
  int GetIdmaDepth() const { return idma_depth_; }
  /*! \brief Get odma loop depth. */
  int GetOdmaDepth() const { return odma_depth_; }
  /*! \brief Get vidma loop depth. */
  int GetVidmaDepth() const { return vidma_depth_; }
  /*! \brief Get vodma loop depth. */
  int GetVodmaDepth() const { return vodma_depth_; }
  /*! \brief Get wdma loop depth. */
  int GetWdmaDepth() const { return wdma_depth_; }
  /*! \brief Detect whether the stmt contains bdma load. */
  bool HasBdma() const { return has_bdma_; }
  /*! \brief Detect whether the stmt contains idma load. */
  bool HasIdma() const { return has_idma_; }
  /*! \brief Detect whether the stmt contains odma store. */
  bool HasOdma() const { return has_odma_; }
  /*! \brief Detect whether the stmt contains vidma load. */
  bool HasVidma() const { return has_vidma_; }
  /*! \brief Detect whether the stmt contains vodma store. */
  bool HasVodma() const { return has_vodma_; }
  /*! \brief Detect whether the stmt contains wdma load. */
  bool HasWdma() const { return has_wdma_; }

 private:
  /*! \brief The first nnp_eidma_load intrinsic. */
  const EvaluateNode* first_eidma_{nullptr};
  /*! \brief The first nnp_ewdma_load intrinsic. */
  const EvaluateNode* first_ewdma_{nullptr};
  /*! \brief The last nnp_bdma_store intrinsic. */
  const EvaluateNode* last_bdma_{nullptr};
  /*! \brief The last nnp_eodma_store intrinsic. */
  const EvaluateNode* last_eodma_{nullptr};
  /*! \brief The last nnp_idma_store intrinsic. */
  const EvaluateNode* last_idma_{nullptr};
  /*! \brief The last nnp_odma_store intrinsic. */
  const EvaluateNode* last_odma_{nullptr};
  /*! \brief The last nnp_vidma_load intrinsic. */
  const EvaluateNode* last_vidma_{nullptr};
  /*! \brief The last nnp_vodma_store intrinsic. */
  const EvaluateNode* last_vodma_{nullptr};
  /*! \brief The last nnp_wdma_store intrinsic. */
  const EvaluateNode* last_wdma_{nullptr};
  /*! \brief Current for loop depth. */
  int cur_depth_{0};
  /*! \brief Bdma loop depth. */
  int bdma_depth_{-1};
  /*! \brief Eidma loop depth. */
  int eidma_depth_{-1};
  /*! \brief Eodma loop depth. */
  int eodma_depth_{-1};
  /*! \brief Ewdma loop depth. */
  int ewdma_depth_{-1};
  /*! \brief Idma loop depth. */
  int idma_depth_{-1};
  /*! \brief Odma loop depth. */
  int odma_depth_{-1};
  /*! \brief Vidma loop depth. */
  int vidma_depth_{-1};
  /*! \brief Vodma loop depth. */
  int vodma_depth_{-1};
  /*! \brief Wdma loop depth. */
  int wdma_depth_{-1};
  /*! \brief Whether the stmt contains bdma load. */
  bool has_bdma_{false};
  /*! \brief Whether the stmt contains idma load. */
  bool has_idma_{false};
  /*! \brief Whether the stmt contains odma store. */
  bool has_odma_{false};
  /*! \brief Whether the stmt contains vidma load. */
  bool has_vidma_{false};
  /*! \brief Whether the stmt contains vodma store. */
  bool has_vodma_{false};
  /*! \brief Whether the stmt contains wdma load. */
  bool has_wdma_{false};
};

class HandShakeIntrinInjector : public StmtExprMutator {
 public:
  Stmt Inject(Stmt stmt) {
    detector_(stmt);
    return operator()(std::move(stmt));
  }

  Stmt VisitStmt_(const EvaluateNode* op) final {
    std::vector<Stmt> seq;
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<EvaluateNode>();
    ICHECK(op != nullptr);
    if (op->value.as<CallNode>()->op.same_as(edgex::builtin::nnp_eidma_load())) {
      seq.emplace_back(stmt);
      seq.emplace_back(CreateSyncStmt({StringImm("eidma"), StringImm("ub"), StringImm("cu")}));
      seq.emplace_back(CreateSyncStmt({StringImm("cu"), StringImm("wo"), StringImm("eidma")}));
      /* if (detector_.HasIdma()) {
        seq.emplace_back(CreateSyncStmt({StringImm("eidma"), StringImm("ub"), StringImm("idma")}));
        seq.emplace_back(CreateSyncStmt({StringImm("idma"), StringImm("wo"), StringImm("eidma")}));
      } */
      if (detector_.HasVidma()) {
        seq.emplace_back(
            CreateSyncStmt({StringImm("eidma"), StringImm("ub"), StringImm("vidma0")}));
        seq.emplace_back(CreateSyncStmt({StringImm("vidma"), StringImm("wo"), StringImm("eidma")}));
        seq.emplace_back(
            CreateSyncStmt({StringImm("eidma"), StringImm("wo"), StringImm("vidma0")}));
        seq.emplace_back(CreateSyncStmt({StringImm("vidma"), StringImm("ub"), StringImm("eidma")}));
      }
      /* if (detector_.HasWdma()) {
        seq.emplace_back(CreateSyncStmt({StringImm("eidma"), StringImm("ub"), StringImm("wdma")}));
        seq.emplace_back(CreateSyncStmt({StringImm("wdma"), StringImm("wo"), StringImm("eidma")}));
      } */
      return SeqStmt::Flatten(seq);
    } else if (op->value.as<CallNode>()->op.same_as(edgex::builtin::nnp_eodma_store())) {
      if (detector_.HasVodma()) {
        seq.emplace_back(CreateSyncStmt({StringImm("vodma"), StringImm("ub"), StringImm("eodma")}));
        seq.emplace_back(
            CreateSyncStmt({StringImm("eodma"), StringImm("wo"), StringImm("vodma0")}));
      }
      if (detector_.HasOdma()) {
        seq.emplace_back(CreateSyncStmt({StringImm("odma"), StringImm("ub"), StringImm("eodma")}));
        seq.emplace_back(CreateSyncStmt({StringImm("eodma"), StringImm("wo"), StringImm("odma")}));
      }
      seq.emplace_back(stmt);
      seq.emplace_back(CreateSyncStmt({StringImm("eodma"), StringImm("ub"), StringImm("cu")}));
      seq.emplace_back(CreateSyncStmt({StringImm("cu"), StringImm("wo"), StringImm("eodma")}));
      return SeqStmt::Flatten(seq);
    } else if (op->value.as<CallNode>()->op.same_as(edgex::builtin::nnp_ewdma_load())) {
      seq.emplace_back(stmt);
      seq.emplace_back(CreateSyncStmt({StringImm("ewdma"), StringImm("ub"), StringImm("cu")}));
      seq.emplace_back(CreateSyncStmt({StringImm("cu"), StringImm("wo"), StringImm("ewdma")}));
      if (detector_.HasBdma()) {
        seq.emplace_back(CreateSyncStmt({StringImm("ewdma"), StringImm("ub"), StringImm("bdma")}));
        seq.emplace_back(CreateSyncStmt({StringImm("bdma"), StringImm("wo"), StringImm("ewdma")}));
      }
      if (detector_.HasWdma()) {
        seq.emplace_back(CreateSyncStmt({StringImm("ewdma"), StringImm("ub"), StringImm("wdma")}));
        seq.emplace_back(CreateSyncStmt({StringImm("wdma"), StringImm("wo"), StringImm("ewdma")}));
      }
      return SeqStmt::Flatten(seq);
    } else if (op->value.as<CallNode>()->op.same_as(edgex::builtin::nnp_vidma_load())) {
      seq.emplace_back(stmt);
      seq.emplace_back(CreateSyncStmt({StringImm("vidma"), StringImm("ub"), StringImm("vcu")}));
      seq.emplace_back(CreateSyncStmt({StringImm("vcu"), StringImm("wo"), StringImm("vidma")}));
      return SeqStmt::Flatten(seq);
    } else if (op->value.as<CallNode>()->op.same_as(edgex::builtin::nnp_vodma_store())) {
      seq.emplace_back(stmt);
      seq.emplace_back(CreateSyncStmt({StringImm("vodma"), StringImm("ub"), StringImm("vcu")}));
      seq.emplace_back(CreateSyncStmt({StringImm("vcu"), StringImm("wo"), StringImm("vodma")}));
      return SeqStmt::Flatten(seq);
    } else {
      return stmt;
    }
  }

  Stmt VisitStmt_(const ForNode* op) final {
    is_for_body_ = true;
    loop_var_ = op->loop_var;
    min_ = op->min;
    extent_ = op->extent;
    Stmt ret = StmtExprMutator::VisitStmt_(op);
    is_for_body_ = false;
    return ret;
  }

  /*!
   * \brief Create nnp_sync tir.
   */
  Stmt CreateSyncStmt(const Array<PrimExpr>& args) {
    return Evaluate(Call(DataType::Void(), edgex::builtin::nnp_sync(), args));
  }

 private:
  /*! \brief The IntrinPositionDetector. */
  IntrinDetector detector_;
  /*! \brief whether is the ForNode's body. */
  bool is_for_body_{false};
  /*! \brief ForNode loop_var. */
  Var loop_var_;
  /*! \brief ForNode min. */
  PrimExpr min_;
  /*! \brief ForNode extent. */
  PrimExpr extent_;
};

Stmt InjectHandShakeIntrin(Stmt stmt) { return HandShakeIntrinInjector().Inject(std::move(stmt)); }

namespace transform {

Pass InjectHandShakeIntrin() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = InjectHandShakeIntrin(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.edgex.InjectHandShakeIntrin", {});
}

TVM_REGISTER_GLOBAL("tir.edgex.transform.InjectHandShakeIntrin")
    .set_body_typed(InjectHandShakeIntrin);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
