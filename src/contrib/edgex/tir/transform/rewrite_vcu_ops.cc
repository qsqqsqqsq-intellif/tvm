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
 * \file rewrite_vcu_ops.cc
 */
#include <llvm/IR/Intrinsics.h>
#include <math.h>
#include <tvm/arith/analyzer.h>
#include <tvm/arith/pattern.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../attrs.h"
#include "../edgex_ir_utils.h"
#include "./edgex_pattern_match.h"

namespace tvm {
namespace tir {

using tvm::arith::Pattern;
using tvm::arith::PConst;
using tvm::arith::PVar;
using tvm::arith::PVarWithDataType;
using tvm::arith::PVecDataType;

constexpr const int VELTADD_ASR_MODE_ROUNDING = 4;
constexpr const int VELTADD_RELU_MODE_ENABLE = 1;
constexpr const int FP2FX_MODE_ROUNDING = 4;

/*! \brief detect vectorized relu */
static bool MatchVectorizedInt8Relu(const PrimExpr& expr) {
  PVecDataType i8_vec_ty(DataType::Int(8));
  PVar<PrimExpr> vec_len;
  PVarWithDataType<PrimExpr, PVecDataType> data(i8_vec_ty);
  auto broadcast_zero = broadcast(PConst<PrimExpr>(make_const(DataType::Int(8), 0)), vec_len);
  auto relu_pattern = max(data, broadcast_zero);
  return relu_pattern.Match(expr);
}

/*! \brief quantized pattern for veltadd */
static PrimExpr RewriteVeltaddPattern(const PrimExpr& expr) {
  // fast filter for root relu and cast to i8 pattern
  PrimExpr root = expr;
  arith::Analyzer ana;
  bool has_relu = MatchVectorizedInt8Relu(root);
  if (has_relu) {
    root = root.as<MaxNode>()->a;
  }
  const CastNode* castnode = root.as<CastNode>();
  if (!castnode || castnode->dtype.element_of() != DataType::Int(8) ||
      !castnode->value->dtype.is_int()) {
    return expr;
  }
  int64_t lanes = castnode->dtype.lanes();

  // match concrete patterns
  PVar<DataType> data_vec_ty;
  PVecDataType i8_vec_ty(DataType::Int(8));
  PVecDataType i64_vec_ty(DataType::Int(64));
  auto broadcast_upper =
      broadcast(PConst<PrimExpr>(make_const(DataType::Int(64), 127)), PConst<int64_t>(lanes));
  auto broadcast_lower =
      broadcast(PConst<PrimExpr>(make_const(DataType::Int(64), -128)), PConst<int64_t>(lanes));
  arith::Analyzer analyzer;

  // try get a vectorized input argument with smallest bits
  std::function<PrimExpr(const PrimExpr&, bool)> get_arg =
      [&get_arg, &analyzer](const PrimExpr& e, bool is_norm) -> PrimExpr {
    // optimzie for constant int
    if (e.dtype().lanes() == 1) {
      int64_t lower = is_norm ? 0 : -128;
      int64_t upper = is_norm ? 256 : 128;
      if (analyzer.CanProveGreaterEqual(e, lower) && analyzer.CanProveLess(e, upper)) {
        auto astype = is_norm ? DataType::UInt(8) : DataType::Int(8);
        return analyzer.Simplify(cast(astype, e));
      }
    }
    if (const BroadcastNode* broadcast = e.as<BroadcastNode>()) {
      return Broadcast(get_arg(broadcast->value, is_norm), broadcast->lanes);
    } else if (const CastNode* cast = e.as<CastNode>()) {
      if ((cast->dtype.is_int() || cast->dtype.is_uint()) &&
          (cast->value.dtype().is_int() || cast->value.dtype().is_uint()) &&
          cast->dtype.bits() >= cast->value.dtype().bits()) {
        return get_arg(cast->value, is_norm);
      }
    }
    return e;
  };

  // determin arg dtype validity
  auto is_valid_arg = [](const PrimExpr& e) {
    int bits = e.dtype().element_of().bits();
    return bits < 32 || (bits == 32 && e.dtype().is_int());
  };

  // define patterns, min/max maybe get eliminated by simplify
  PVarWithDataType<PrimExpr, PVecDataType> pat_input(i64_vec_ty);
  PVarWithDataType<PrimExpr, PVecDataType> pat_mulnorm(i64_vec_ty);
  PVarWithDataType<PrimExpr, PVecDataType> pat_shiftnorm(i64_vec_ty);
  auto veltadd_pattern0 =
      cast(i8_vec_ty,
           max(min(round_right_shift(pat_input, pat_shiftnorm), broadcast_upper), broadcast_lower));
  auto veltadd_pattern1 = cast(i8_vec_ty, round_right_shift(pat_input, pat_shiftnorm));
  if (!veltadd_pattern0.Match(root) && !veltadd_pattern1.Match(root)) {
    return expr;
  }
  PrimExpr shiftnorm = get_arg(pat_shiftnorm.Eval(), true);

  // general pattern matching
  auto dummy_mulnorm = Broadcast(make_const(DataType::UInt(8), 1), lanes);
  auto dummy_arg = Broadcast(make_const(DataType::Int(8), 0), lanes);
  PrimExpr arg0{dummy_arg}, arg1{dummy_arg}, mulnorm{dummy_mulnorm};
  PrimExpr input_part = pat_input.Eval();
  PVarWithDataType<PrimExpr, PVecDataType> pat_lhs(i64_vec_ty);
  PVarWithDataType<PrimExpr, PVecDataType> pat_rhs(i64_vec_ty);
  if (((pat_lhs + pat_rhs) * pat_mulnorm).Match(input_part) ||
      ((pat_mulnorm * (pat_lhs + pat_rhs)).Match(input_part))) {
    arg0 = get_arg(pat_lhs.Eval(), false);
    arg1 = get_arg(pat_rhs.Eval(), false);
    mulnorm = get_arg(pat_mulnorm.Eval(), true);
  } else if ((pat_lhs + pat_rhs).Match(input_part)) {
    arg0 = get_arg(pat_lhs.Eval(), false);
    arg1 = get_arg(pat_rhs.Eval(), false);
  } else if ((pat_lhs * pat_rhs).Match(input_part)) {
    PrimExpr lhs = get_arg(pat_lhs.Eval(), true);
    PrimExpr rhs = get_arg(pat_rhs.Eval(), true);
    if (analyzer.CanProveGreaterEqual(lhs, 0) && analyzer.CanProveLess(lhs, 256)) {
      arg0 = get_arg(pat_rhs.Eval(), false);
      mulnorm = lhs;
    } else if (analyzer.CanProveGreaterEqual(rhs, 0) && analyzer.CanProveLess(rhs, 256)) {
      arg0 = get_arg(pat_lhs.Eval(), false);
      mulnorm = rhs;
    } else {
      return expr;
    }
  } else {
    arg0 = get_arg(input_part, false);
  }
  if (mulnorm.dtype().element_of() != DataType::UInt(8) ||
      shiftnorm.dtype().element_of() != DataType::UInt(8)) {
    return expr;
  } else if (!is_valid_arg(arg0) || !is_valid_arg(arg1)) {
    return expr;
  }

  // create veltadd intrinsic call
  if (arg0.dtype().element_of() == DataType::Int(8) &&
      arg1.dtype().element_of() == DataType::Int(8)) {
    Call veltadd =
        Call(i8_vec_ty.Eval(), edgex::builtin::nnp_veltadd(), {mulnorm, arg0, arg1, shiftnorm});
    auto n = const_cast<CallNode*>(veltadd.get());
    edgex::NNPAddArg(n, "asr_rmode", VELTADD_ASR_MODE_ROUNDING);
    edgex::NNPAddArg(n, "veltadd_relu_mode", has_relu ? VELTADD_RELU_MODE_ENABLE : -1);
    return std::move(veltadd);
  } else {
    // fallback to vacc computation
    PrimExpr shiftnorm = pat_shiftnorm.Eval();
    PrimExpr input = pat_input.Eval();
    // cast shiftnorm to int32
    Call nnp_round_right_shift_call =
        Call(DataType::Int(64, lanes), edgex::builtin::nnp_round_right_shift(),
             {input, cast(DataType::Int(32, lanes), shiftnorm)});
    PrimExpr new_expr;
    if (veltadd_pattern0.Match(root)) {
      new_expr =
          cast(DataType::Int(8, lanes),
               Max(Min(nnp_round_right_shift_call, (make_const(DataType::Int(64, lanes), 127))),
                   (make_const(DataType::Int(64, lanes), -128))));
    } else if (veltadd_pattern1.Match(root)) {
      new_expr = cast(DataType::Int(8, lanes), nnp_round_right_shift_call);
    }
    return std::move(new_expr);
  }
}

/*! \brief rewrite round_clip_cast to vint. */
static PrimExpr RewriteVintPattern(const PrimExpr& expr) {
  PrimExpr root = expr;
  const CastNode* castnode = root.as<CastNode>();
  if (!castnode || castnode->dtype.element_of() != DataType::Int(8) ||
      !castnode->value->dtype.is_float16()) {
    return expr;
  }

  int64_t lanes = castnode->dtype.lanes();
  PVecDataType i8_vec_ty(DataType::Int(8));
  PVecDataType fp16_vec_ty(DataType::Float(16));
  PVarWithDataType<PrimExpr, PVecDataType> pat_input_fp16(fp16_vec_ty);
  auto broadcast_upper_fp16 =
      broadcast(PConst<PrimExpr>(make_const(DataType::Float(16), 127)), PConst<int64_t>(lanes));
  auto broadcast_lower_fp16 =
      broadcast(PConst<PrimExpr>(make_const(DataType::Float(16), -128)), PConst<int64_t>(lanes));

  auto vint_pattern =
      cast(i8_vec_ty, max(min(round(pat_input_fp16), broadcast_upper_fp16), broadcast_lower_fp16));
  if (vint_pattern.Match(root)) {
    PrimExpr input = pat_input_fp16.Eval();
    Call vint = Call(i8_vec_ty.Eval(), edgex::builtin::nnp_vint(), {input});
    auto n = const_cast<CallNode*>(vint.get());
    edgex::NNPAddArg(n, "fp2fx_rmode", FP2FX_MODE_ROUNDING);
    return std::move(vint);
  } else {
    return expr;
  }
}

/*! \brief Determine non-consecutive vectorized store indices. */
static bool IsScatteredStore(const BufferStoreNode* store) {
  ICHECK_EQ(store->indices.size(), 1U);
  PrimExpr index = store->indices[0];
  size_t lanes = index->dtype.lanes();
  if (lanes <= 1) {
    return false;
  }
  PVar<PrimExpr> offset;
  PVar<PrimExpr> lanes_ph;
  auto consecutive_access_pat = ramp(offset, 1, lanes);
  auto broadcast_access_pat = broadcast(offset, lanes_ph);
  return !consecutive_access_pat.Match(index) && !broadcast_access_pat.Match(index);
}

/*! \brief Rewrite non-consecutive vectorized store to masked_scatter intrin. */
static Stmt RewriteScatteredStore(const BufferStoreNode* store) {
  size_t lanes = store->indices[0]->dtype.lanes();

  Call access_ptrs(DataType::Handle(64, lanes), tir::builtin::tvm_access_ptr(),
                   {tir::TypeAnnotation(store->value->dtype.with_lanes(1)), store->buffer->data,
                    store->indices[0], Broadcast(1, lanes), StringImm("rw")});
  Call call_intrin(DataType::Void(), tir::builtin::call_llvm_intrin(),
                   {/*intrin=*/make_const(DataType::UInt(32), llvm::Intrinsic::masked_scatter),
                    /*arg_num=*/make_const(DataType::UInt(32), 4),
                    /*values=*/store->value,
                    /*addrs=*/access_ptrs,
                    /*alignment=*/make_const(DataType::Int(32), 0),
                    /*masks=*/Broadcast(make_const(DataType::Bool(), 1), lanes)});
  return std::move(Evaluate(call_intrin));
}

/*! \brief Determine non-consecutive vectorized load indices. */
static bool IsGatherLoad(const BufferLoadNode* load) {
  ICHECK_EQ(load->indices.size(), 1U);
  PrimExpr index = load->indices[0];
  size_t lanes = index->dtype.lanes();
  if (lanes <= 1) {
    return false;
  }
  PVar<PrimExpr> offset;
  PVar<PrimExpr> lanes_ph;
  auto consecutive_access_pat = ramp(offset, 1, lanes);
  auto broadcast_access_pat = broadcast(offset, lanes_ph);
  return !consecutive_access_pat.Match(index) && !broadcast_access_pat.Match(index);
}

/*! \brief Rewrite non-consecutive vectorized load to masked_gather intrin. */
static PrimExpr RewriteGatherLoad(const BufferLoadNode* load) {
  size_t lanes = load->indices[0]->dtype.lanes();

  Call access_ptrs(DataType::Handle(64, lanes), tir::builtin::tvm_access_ptr(),
                   {tir::TypeAnnotation(load->dtype.with_lanes(1)), load->buffer->data,
                    load->indices[0], Broadcast(1, lanes), StringImm("rw")});
  Call call_intrin(load->dtype, tir::builtin::call_llvm_intrin(),
                   {/*intrin=*/make_const(DataType::UInt(32), llvm::Intrinsic::masked_gather),
                    /*arg_num=*/make_const(DataType::UInt(32), 4),
                    /*addrs=*/access_ptrs,
                    /*alignment=*/make_const(DataType::Int(32), 0),
                    /*masks=*/Broadcast(make_const(DataType::Bool(), 1), lanes),
                    /*passthrough=*/Broadcast(make_const(load->dtype.element_of(), 0), lanes)});
  return std::move(call_intrin);
}

class VectorizedOpsRewritter : public StmtExprMutator {
 private:
  PrimExpr VisitExpr(const PrimExpr& expr) override {
    PrimExpr updated_veltadd = RewriteVeltaddPattern(expr);
    if (!expr.same_as(updated_veltadd)) {
      return updated_veltadd;
    }
    PrimExpr updated_vint = RewriteVintPattern(expr);
    if (!expr.same_as(updated_vint)) {
      return updated_vint;
    }
    return StmtExprMutator::VisitExpr(expr);
  }

  // transform nnp_nlfc_exp to nnp_nlfc_exp2
  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(edgex::builtin::nnp_nlfc_exp())) {
      auto dtype = op->dtype;
      // exp(x) = exp2(x/ln2), x -> x/ln2
      PrimExpr x = make_const(dtype, 1.0f / std::log(2)) * op->args[0];
      Array<PrimExpr> new_args = {x, op->args[1]};
      Call exp2 = Call(dtype, edgex::builtin::nnp_nlfc_exp2(), new_args);
      return std::move(exp2);
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* load) override {
    if ((allow_wildcard_gather_load_buffer_ ||
         gather_load_buffer_vars_.count(load->buffer->data)) &&
        IsGatherLoad(load)) {
      return VisitExpr(RewriteGatherLoad(load));
    }
    return StmtExprMutator::VisitExpr_(load);
  }

  Stmt VisitStmt_(const BufferStoreNode* store) override {
    if ((allow_wildcard_scatter_store_buffer_ ||
         scatter_store_buffer_vars_.count(store->buffer->data)) &&
        IsScatteredStore(store)) {
      return VisitStmt(RewriteScatteredStore(store));
    }
    return StmtExprMutator::VisitStmt_(store);
  }

  Stmt VisitStmt_(const AttrStmtNode* attr) override {
    if (attr->attr_key == attr::nnp_gather_load_scope) {
      bool tmp = allow_wildcard_gather_load_buffer_;
      bool has_var = attr->value->IsInstance<VarNode>();
      Var buffer_var;
      if (has_var) {
        buffer_var = Downcast<Var>(attr->value);
        gather_load_buffer_vars_.insert(buffer_var);
      } else {
        allow_wildcard_gather_load_buffer_ = true;
      }
      auto res = StmtExprMutator::VisitStmt(attr->body);
      if (has_var) {
        gather_load_buffer_vars_.erase(buffer_var);
      } else {
        allow_wildcard_gather_load_buffer_ = tmp;
      }
      return res;
    } else if (attr->attr_key == attr::nnp_scatter_store_scope) {
      bool tmp = allow_wildcard_scatter_store_buffer_;
      bool has_var = attr->value->IsInstance<VarNode>();
      Var buffer_var;
      if (has_var) {
        buffer_var = Downcast<Var>(attr->value);
        scatter_store_buffer_vars_.insert(buffer_var);
      } else {
        allow_wildcard_scatter_store_buffer_ = true;
      }
      auto res = StmtExprMutator::VisitStmt(attr->body);
      if (has_var) {
        scatter_store_buffer_vars_.erase(buffer_var);
      } else {
        allow_wildcard_scatter_store_buffer_ = tmp;
      }
      return res;
    }
    return StmtExprMutator::VisitStmt_(attr);
  }

  /*!
   * \brief buffer vars to allow vectorized scatter store.
   */
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> scatter_store_buffer_vars_;

  /*!
   * \brief buffer vars to allow vectorized gather load.
   */
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> gather_load_buffer_vars_;

  /*! \brief allow wildcard vectorized scatter store rewrite. */
  bool allow_wildcard_scatter_store_buffer_;

  /*! \brief allow wildcard vectorized gather load rewrite. */
  bool allow_wildcard_gather_load_buffer_;
};

namespace transform {

Pass RewriteVcuOps() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = VectorizedOpsRewritter()(std::move(f->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.edgex.RewriteVcuOps", {});
}

TVM_REGISTER_GLOBAL("tir.edgex.transform.RewriteVcuOps").set_body_typed(RewriteVcuOps);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
