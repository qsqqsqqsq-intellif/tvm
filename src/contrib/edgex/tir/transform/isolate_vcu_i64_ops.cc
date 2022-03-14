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
 * \file isolate_vcu_i64_ops.cc
 */
#include <tvm/arith/analyzer.h>
#include <tvm/arith/pattern.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../edgex_ir_utils.h"
#include "./edgex_pattern_match.h"

namespace tvm {
namespace tir {

using tvm::arith::PConst;
using tvm::arith::PVarWithDataType;
using tvm::arith::PVecDataType;

/*! \brief isolate vcu round_right_shift input for qat pattern */
static PrimExpr IsolateVcuRoundRightShiftInput(const PrimExpr& expr) {
  // fast filter for cast to u8 pattern and cast to int32 pattern
  PrimExpr root = expr;
  const CastNode* castnode = root.as<CastNode>();
  if (!castnode ||
      (castnode->dtype.element_of() != DataType::UInt(8) &&
       castnode->dtype.element_of() != DataType::Int(32)) ||
      !castnode->value->dtype.is_int()) {
    return expr;
  }
  int64_t lanes = castnode->dtype.lanes();

  // match concrete patterns
  PVecDataType u8_vec_ty(DataType::UInt(8));
  PVecDataType i32_vec_ty(DataType::Int(32));
  PVecDataType i64_vec_ty(DataType::Int(64));

  // define patterns, min/max maybe get eliminated by simplify
  PVarWithDataType<PrimExpr, PVecDataType> pat_input(i64_vec_ty);
  PVarWithDataType<PrimExpr, PVecDataType> pat_quant_input(i32_vec_ty);
  PVarWithDataType<PrimExpr, PVecDataType> pat_mulnorm(i64_vec_ty);
  PVarWithDataType<PrimExpr, PVecDataType> pat_shiftnorm(i64_vec_ty);
  auto upper = PConst<PrimExpr>(make_const(DataType::Int(64), 255));
  auto lower = PConst<PrimExpr>(make_const(DataType::Int(64), 0));

  auto veltadd_pattern0 =
      cast(u8_vec_ty, max(min(round_right_shift(pat_input, pat_shiftnorm), upper), lower));
  auto veltadd_pattern1 = cast(u8_vec_ty, round_right_shift(pat_input, pat_shiftnorm));
  auto veltadd_pattern2 = cast(i32_vec_ty, round_right_shift(pat_input, pat_shiftnorm));

  if (!veltadd_pattern0.Match(root) && !veltadd_pattern1.Match(root) &&
      !veltadd_pattern2.Match(root)) {
    return expr;
  }

  PrimExpr shiftnorm = pat_shiftnorm.Eval();
  PrimExpr input = pat_input.Eval();
  if ((cast(i64_vec_ty, pat_quant_input) * pat_mulnorm).Match(input)) {
    PrimExpr quant_input = pat_quant_input.Eval();
    PrimExpr mulnorm = pat_mulnorm.Eval();

    // let binding to prevent the PushCastToChildren simplify
    Var wrapped_quant_input("wrapped_quant_input", DataType::Int(32, lanes));
    quant_input = Let(wrapped_quant_input, quant_input, wrapped_quant_input);

    // cast shiftnorm to int32
    Call nnp_round_right_shift_call =
        Call(DataType::Int(64, lanes), edgex::builtin::nnp_round_right_shift(),
             {cast(DataType::Int(64, lanes), quant_input) * mulnorm,
              cast(DataType::Int(32, lanes), shiftnorm)});

    PrimExpr new_expr;
    if (veltadd_pattern0.Match(root)) {
      new_expr =
          cast(DataType::UInt(8, lanes),
               Max(Min(nnp_round_right_shift_call, (make_const(DataType::Int(64, lanes), 127))),
                   (make_const(DataType::Int(64, lanes), -128))));
    } else if (veltadd_pattern1.Match(root)) {
      new_expr = cast(DataType::UInt(8, lanes), nnp_round_right_shift_call);
    } else if (veltadd_pattern2.Match(root)) {
      new_expr = cast(DataType::Int(32, lanes), nnp_round_right_shift_call);
    } else {
      return expr;
    }
    return std::move(new_expr);
  } else {
    return expr;
  }
}
class IsolateVcuI64OpsRewritter : public StmtExprMutator {
 private:
  PrimExpr VisitExpr(const PrimExpr& expr) override {
    PrimExpr updated = IsolateVcuRoundRightShiftInput(expr);
    if (!expr.same_as(updated)) {
      return updated;
    }
    return StmtExprMutator::VisitExpr(expr);
  }
};

namespace transform {

Pass IsolateVcuI64Ops() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = IsolateVcuI64OpsRewritter()(std::move(f->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.edgex.IsolateVcuI64Ops", {});
}

TVM_REGISTER_GLOBAL("tir.edgex.transform.IsolateVcuI64Ops").set_body_typed(IsolateVcuI64Ops);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
