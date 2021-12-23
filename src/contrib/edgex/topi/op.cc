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
#include <tvm/tir/buffer.h>
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
  Array<PrimExpr> new_shape;
  PrimExpr total_bytes = x->shape.back() * x->dtype.bytes();
  for (size_t i = 0; i < x->shape.size() - 1; i++) {
    new_shape.push_back(x->shape[i]);
    total_bytes *= x->shape[i];
  }
  bool is_downcast = x->dtype.bits() > type.bits();
  if (is_downcast) {
    ICHECK_EQ(x->dtype.bits() % type.bits(), 0)
        << type.bits() << "is not divisible into" << x->dtype.bits();
    const int factor = x->dtype.bits() / type.bits();
    new_shape.push_back(x->shape.back() * factor);
  } else {
    ICHECK_EQ(type.bits() % x->dtype.bits(), 0)
        << x->dtype.bits() << "is not divisible into" << type.bits();
    const int factor = type.bits() / x->dtype.bits();
    new_shape.push_back(floordiv(x->shape.back(), factor));
  }

  tvm::te::Buffer input_buffer = tvm::tir::decl_buffer(x->shape, x->dtype);
  tvm::te::Buffer output_buffer = tvm::tir::decl_buffer(new_shape, type);
  tvm::tir::Call call(DataType::Handle(), tvm::tir::builtin::call_extern(),
                      {tvm::tir::StringImm("cast_reinterpret_extern_data_copy"), total_bytes,
                       input_buffer.access_ptr(1), output_buffer.access_ptr(2)});
  tvm::te::ExternOp cast_reinterpret_extern("cast_reinterpret_extern", tag, {}, {x}, {input_buffer},
                                            {output_buffer}, tvm::tir::Evaluate(call));
  return cast_reinterpret_extern.output(0);
}

extern "C" void cast_reinterpret_extern_data_copy(int64_t bytes, void* src, void* dst) {
  memcpy(dst, src, bytes);
}

TVM_REGISTER_GLOBAL("topi.cast_reinterpret").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = cast_reinterpret(args[0], args[1]);
});

}  // namespace topi
}  // namespace tvm
