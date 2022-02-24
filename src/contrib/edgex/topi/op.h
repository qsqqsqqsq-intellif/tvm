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
 * \brief Edgex extension op constructions
 * \file topi/op.h
 */
#ifndef TVM_CONTRIB_EDGEX_TOPI_OP_H_
#define TVM_CONTRIB_EDGEX_TOPI_OP_H_

#include <tvm/topi/broadcast.h>

#include <string>

#include "../tir/op/builtin.h"

using tvm::tir::edgex::builtin::nnp_round_right_shift;

namespace tvm {
namespace topi {

TOPI_DEFINE_BCAST_OP(round_right_shift, {
  auto pos = 1l << (b - 1);
  auto neg = (1l << (b - 1)) - 1;
  auto round_param = tvm::tir::Select(a >= 0, pos, neg);
  return tvm::tir::Select(b > 0, (a + round_param) >> b, a);
});

TOPI_DEFINE_BCAST_OP(round_right_shift_intrin, {
  return tir::Call(a.dtype(), nnp_round_right_shift(), {a, b});
});

/*!
 * \brief Reinterpret_cast each element of x to the given type.

 * \param x The input tensor
 * \param type The type to cast to
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the reinterpret_cast operation
 */
tvm::te::Tensor cast_reinterpret(const tvm::te::Tensor& x, DataType type,
                                 const std::string& name = "cast_reinterpret",
                                 const std::string& tag = kElementWise);

}  // namespace topi
}  // namespace tvm

#endif  // TVM_CONTRIB_EDGEX_TOPI_OP_H_
