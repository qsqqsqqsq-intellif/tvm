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
 * \file op.cc
 * \brief quantization extension of relay operators.
 */
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/topi/broadcast.h>

#include "../../../relay/op/nn/pooling.h"
#include "../../../relay/op/op_common.h"
#include "../../../relay/transforms/infer_layout_utils.h"

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

TOPI_DEFINE_BCAST_OP(round_right_shift, {
  auto pos = 1ll << (b - 1);
  auto neg = (1ll << (b - 1)) - 1;
  auto round_param = tvm::tir::Select(a >= 0, pos, neg);
  return tvm::tir::Select(b > 0, (a + round_param) >> b, a);
});

TOPI_REGISTER_BCAST_OP("topi.round_right_shift", round_right_shift);

}  // namespace topi

namespace relay {

#define RELAY_BINARY_COMPUTE(FTOPI)                       \
  [](const Attrs& attrs, const Array<te::Tensor>& inputs, \
     const Type& out_type) -> Array<te::Tensor> {         \
    ICHECK_EQ(inputs.size(), 2U);                         \
    return {FTOPI(inputs[0], inputs[1])};                 \
  }

// relay.round_right_shift
RELAY_REGISTER_BINARY_OP("round_right_shift")
    .describe("Elementwise round and right shift with broadcasting")
    .set_support_level(1)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::round_right_shift));

}  // namespace relay
}  // namespace tvm
