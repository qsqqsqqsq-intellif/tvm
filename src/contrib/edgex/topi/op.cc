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
 * \brief Registration of edgex extension operators
 * \file op.cc
 */
#include "./op.h"

#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/topi/broadcast.h>
#include <tvm/topi/utils.h>

#include "../tir/op/builtin.h"

namespace tvm {
namespace topi {

#define TOPI_REGISTER_BCAST_OP(OpName, Op)                                              \
  TVM_REGISTER_GLOBAL(OpName).set_body([](TVMArgs args, TVMRetValue* rv) {              \
    bool lhs_is_tensor = args[0].IsObjectRef<tvm::te::Tensor>();                        \
    bool rhs_is_tensor = args[1].IsObjectRef<tvm::te::Tensor>();                        \
    if (lhs_is_tensor && rhs_is_tensor) {                                               \
      *rv = Op(args[0].operator tvm::te::Tensor(), args[1].operator tvm::te::Tensor()); \
    } else if (!lhs_is_tensor && rhs_is_tensor) {                                       \
      *rv = Op(args[0].operator tvm::PrimExpr(), args[1].operator tvm::te::Tensor());   \
    } else if (lhs_is_tensor && !rhs_is_tensor) {                                       \
      *rv = Op(args[0].operator tvm::te::Tensor(), args[1].operator tvm::PrimExpr());   \
    } else if (!lhs_is_tensor && !rhs_is_tensor) {                                      \
      *rv = Op(args[0].operator tvm::PrimExpr(), args[1].operator tvm::PrimExpr());     \
    }                                                                                   \
  });

TOPI_REGISTER_BCAST_OP("topi.round_right_shift", round_right_shift);
TOPI_REGISTER_BCAST_OP("topi.round_right_shift_intrin", round_right_shift_intrin);

tvm::te::Tensor cast_reinterpret(const tvm::te::Tensor& x, DataType type, const std::string& name,
                                 const std::string& tag) {
  const int expand = x->dtype.bits() / type.bits();
  ICHECK_GT(expand, 1) << "only support downcast, like int32 to int8.";
  ICHECK_EQ(x->dtype.bits() % type.bits(), 0)
      << type.bits() << "is not divisible into" << x->dtype.bits();

  Array<PrimExpr> new_shape;
  for (size_t i = 0; i < x->shape.size() - 1; i++) {
    new_shape.push_back(x->shape[i]);
  }
  // Caculate the last dimension.
  new_shape.push_back(x->shape.back() * expand);
  const PrimExpr bits = tvm::tir::make_const(x->dtype, type.bits());
  const PrimExpr operand =
      max_value(DataType::UInt(x->dtype.bits())) >> (x->dtype.bits() - type.bits());

  return tvm::te::compute(
      new_shape,
      [&](const Array<tir::Var>& indices) {
        Array<tvm::PrimExpr> idx;
        for (size_t i = 0; i < x->shape.size() - 1; i++) {
          idx.push_back(indices[i]);
        }
        idx.push_back(indexdiv(indices.back(), expand));
        return tvm::cast(type, (x(idx) >> (bits * indexmod(indices.back(), expand))) & operand);
      },
      name, tag);
}

TVM_REGISTER_GLOBAL("topi.cast_reinterpret").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = cast_reinterpret(args[0], args[1]);
});

}  // namespace topi
}  // namespace tvm
