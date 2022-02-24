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
#include <tvm/arith/analyzer.h>
#include <tvm/arith/pattern.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../../../arith/pattern_match.h"
#include "../edgex_ir_utils.h"
#include "../op/builtin.h"

namespace tvm {
namespace tir {

using tvm::arith::Pattern;
using tvm::arith::PConst;
using tvm::arith::PVar;
using tvm::arith::PVarWithDataType;
using tvm::arith::PVecDataType;

constexpr const int VELTADD_ASR_MODE_ROUNDING = 4;
constexpr const int VELTADD_RELU_MODE_ENABLE = 1;

// define match pattern of round right shift intrin
namespace builtin {
using tvm::tir::edgex::builtin::nnp_round_right_shift;
}  // namespace builtin
using arith::PCallExpr;
TVM_PATTERN_BINARY_INTRIN(round_right_shift, PRoundRightShiftOp, nnp_round_right_shift);

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

class VectorizedOpsRewritter : public StmtExprMutator {
 private:
  PrimExpr VisitExpr(const PrimExpr& expr) override {
    PrimExpr updated = RewriteVeltaddPattern(expr);
    if (!expr.same_as(updated)) {
      return updated;
    }
    return StmtExprMutator::VisitExpr(expr);
  }
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
