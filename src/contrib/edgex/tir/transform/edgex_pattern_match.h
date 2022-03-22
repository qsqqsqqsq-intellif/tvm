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
 * \file edgex_pattern_match.h
 * \brief Helper functions to edgex patterns.
 */
#ifndef TVM_CONTRIB_EDGEX_TIR_TRANSFORM_EDGEX_PATTERN_MATCH_H_
#define TVM_CONTRIB_EDGEX_TIR_TRANSFORM_EDGEX_PATTERN_MATCH_H_

#include "../../../../arith/pattern_match.h"
#include "../op/builtin.h"

// pattern unary op by name
#define TVM_PATTERN_UNARY_INTRIN_BY_NAME(FuncName, OpName, IntrinOpName) \
  struct OpName {                                                        \
    static PrimExpr Eval(Array<PrimExpr> args) {                         \
      return tir::Call(args[0].dtype(), GetOp(), args);                  \
    }                                                                    \
    static const Op& GetOp() { return Op::Get(IntrinOpName); }           \
  };                                                                     \
  template <typename TA>                                                 \
  inline PCallExpr<OpName, TA> FuncName(const Pattern<TA>& a) {          \
    return PCallExpr<OpName, TA>(a.derived());                           \
  }

namespace tvm {
namespace tir {

using tvm::arith::Pattern;

namespace builtin {
using tvm::tir::edgex::builtin::nnp_round_right_shift;
}  // namespace builtin

using arith::PCallExpr;
TVM_PATTERN_BINARY_INTRIN(round_right_shift, PRoundRightShiftOp, nnp_round_right_shift);
TVM_PATTERN_UNARY_INTRIN_BY_NAME(round, PRoundOp, "tir.round");

}  // namespace tir
}  // namespace tvm

#endif  // TVM_CONTRIB_EDGEX_TIR_TRANSFORM_EDGEX_PATTERN_MATCH_H_
