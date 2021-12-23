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
 * \brief edgex extension of relay operators.
 */
#include "../../topi/op.h"

#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/topi/broadcast.h>
#include <tvm/topi/nn/pooling.h>

#include "../../../../relay/op/nn/pooling.h"
#include "../../../../relay/op/op_common.h"
#include "../../../../relay/transforms/infer_layout_utils.h"

namespace tvm {
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

// relay.cast_reinterpret
bool CastReinterpretRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                        const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "cast: expect input type to be TensorType but get " << types[0];
    return false;
  }
  const auto* param = attrs.as<CastAttrs>();
  const int ndim = static_cast<int>(data->shape.size());
  std::vector<IndexExpr> oshape;
  oshape.reserve(ndim);
  for (int i = 0; i < ndim - 1; ++i) {
    oshape.emplace_back(data->shape[i]);
  }
  oshape.emplace_back(indexdiv(data->shape.back() * data->dtype.bits(), param->dtype.bits()));
  reporter->Assign(types[1], TensorType(oshape, param->dtype));
  return true;
}

Array<te::Tensor> CastReinterpretCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                         const Type& out_type) {
  const CastAttrs* param = attrs.as<CastAttrs>();
  ICHECK(param != nullptr);
  DataType dtype = param->dtype;
  return {topi::cast_reinterpret(inputs[0], dtype)};
}

Expr MakeCastReinterpret(Expr data, DataType dtype) {
  auto attrs = make_object<CastAttrs>();
  attrs->dtype = dtype;
  static const Op& op = Op::Get("cast_reinterpret");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.cast_reinterpret").set_body_typed(MakeCastReinterpret);

RELAY_REGISTER_OP("cast_reinterpret")
    .describe(R"code(Reinterpret_cast the data into a new data type.
)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_attrs_type<CastAttrs>()
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(3)
    .add_type_rel("CastReinterpret", CastReinterpretRel)
    .set_attr<FTVMCompute>("FTVMCompute", CastReinterpretCompute)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

}  // namespace relay
}  // namespace tvm
