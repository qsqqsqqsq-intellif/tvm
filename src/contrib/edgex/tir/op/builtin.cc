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
 * \file tir/edgex/op/builtin.cc
 *
 * edgex builtin intrinsic operators.
 * NOTE: Please add the intrinsic as follow order:
 *       1.Add intrinsics in alphabetical order;
 *       2.The intrinsics block order is:
 *         load&store -> handshake -> others
 */
#include "builtin.h"

#include <tvm/runtime/registry.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "../../topi/op.h"

namespace tvm {
namespace tir {
namespace edgex {
namespace builtin {

#define TIR_DEFINE_BUILTIN_FUNC(OpName)            \
  const Op& OpName() {                             \
    static const Op& op = Op::Get("tir." #OpName); \
    return op;                                     \
  }                                                \
  TVM_REGISTER_OP("tir." #OpName)

TIR_DEFINE_BUILTIN_FUNC(nnp_bdma_load)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .set_attr<TNNPUnitKind>("TNNPUnitKind", Integer(NNPUnitKind::CU));

TIR_DEFINE_BUILTIN_FUNC(nnp_eidma_load)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .set_attr<TNNPUnitKind>("TNNPUnitKind", Integer(NNPUnitKind::CU));

TIR_DEFINE_BUILTIN_FUNC(nnp_eodma_store)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .set_attr<TNNPUnitKind>("TNNPUnitKind", Integer(NNPUnitKind::CU));

TIR_DEFINE_BUILTIN_FUNC(nnp_ewdma_load)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .set_attr<TNNPUnitKind>("TNNPUnitKind", Integer(NNPUnitKind::CU));

TIR_DEFINE_BUILTIN_FUNC(nnp_idma_load)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .set_attr<TNNPUnitKind>("TNNPUnitKind", Integer(NNPUnitKind::CU));

TIR_DEFINE_BUILTIN_FUNC(nnp_odma_store)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .set_attr<TNNPUnitKind>("TNNPUnitKind", Integer(NNPUnitKind::CU));

TIR_DEFINE_BUILTIN_FUNC(nnp_vidma_load)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .set_attr<TNNPUnitKind>("TNNPUnitKind", Integer(NNPUnitKind::VCU));

TIR_DEFINE_BUILTIN_FUNC(nnp_vodma_store)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .set_attr<TNNPUnitKind>("TNNPUnitKind", Integer(NNPUnitKind::VCU));

TIR_DEFINE_BUILTIN_FUNC(nnp_wdma_load)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .set_attr<TNNPUnitKind>("TNNPUnitKind", Integer(NNPUnitKind::CU));

TIR_DEFINE_BUILTIN_FUNC(nnp_sync).set_attr<TCallEffectKind>("TCallEffectKind",
                                                            Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(nnp_cube)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .set_attr<Integer>("TNNPUnitKind", Integer(NNPUnitKind::CU));

TIR_DEFINE_BUILTIN_FUNC(nnp_cuid).set_attr<TCallEffectKind>("TCallEffectKind",
                                                            Integer(CallEffectKind::kOpaque));

TIR_DEFINE_BUILTIN_FUNC(nnp_iss_bind_input_buffer)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .set_attr<TNNPUnitKind>("TNNPUnitKind", Integer(NNPUnitKind::CU));

TIR_DEFINE_BUILTIN_FUNC(nnp_veltadd)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TNNPUnitKind>("TNNPUnitKind", Integer(NNPUnitKind::VCU));

TIR_DEFINE_BUILTIN_FUNC(nnp_vacc_madd_right_shift)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TNNPUnitKind>("TNNPUnitKind", Integer(NNPUnitKind::VCU));

TIR_DEFINE_BUILTIN_FUNC(nnp_round_right_shift)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TNNPUnitKind>("TNNPUnitKind", Integer(NNPUnitKind::ALL))
    .set_attr<TVectorizable>("TVectorizable", true)
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic", [](const PrimExpr& e) {
      Call call = Downcast<Call>(e);
      ICHECK_EQ(call->args.size(), 2U);
      const auto& x = call->args[0];
      const auto& y = call->args[1];
      return topi::round_right_shift(x, y);
    });

}  // namespace builtin
}  // namespace edgex
}  // namespace tir
}  // namespace tvm
