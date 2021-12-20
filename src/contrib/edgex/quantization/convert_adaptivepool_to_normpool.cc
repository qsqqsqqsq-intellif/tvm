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
 * \file convert_adaptivepool_to_normpool_ops.cc
 * \brief Convert adaptive_pool to max_pool or avg_pool.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/transform.h>

#include "../../../relay/transforms/pattern_utils.h"

namespace tvm {
namespace relay {

class ConvertAdaptivePoolToNormPool : public ExprMutator {
 public:
  Expr MakeMaxPool(Expr data, const Array<IndexExpr>& pool_size, const Array<IndexExpr>& strides,
                   const Array<IndexExpr>& padding, const std::string& layout,
                   const std::string& out_layout) {
    auto attrs = make_object<MaxPool2DAttrs>();
    attrs->pool_size = std::move(pool_size);
    attrs->strides = std::move(strides);
    attrs->layout = std::move(layout);
    attrs->out_layout = std::move(out_layout);
    attrs->padding = std::move(padding);
    attrs->ceil_mode = false;
    attrs->dilation = {tir::make_const(DataType::Int(32), 1),
                       tir::make_const(DataType::Int(32), 1)};
    static const Op& op = Op::Get("nn.max_pool2d");
    return Call(op, {data}, Attrs(attrs), {});
  }

  Expr MakeAvgPool(Expr data, const Array<IndexExpr>& pool_size, const Array<IndexExpr>& strides,
                   const Array<IndexExpr>& padding, const std::string& layout,
                   const std::string& out_layout) {
    auto attrs = make_object<AvgPool2DAttrs>();
    attrs->pool_size = std::move(pool_size);
    attrs->strides = std::move(strides);
    attrs->layout = std::move(layout);
    attrs->out_layout = std::move(out_layout);
    attrs->padding = std::move(padding);
    attrs->dilation = {tir::make_const(DataType::Int(32), 1),
                       tir::make_const(DataType::Int(32), 1)};
    static const Op& op = Op::Get("nn.avg_pool2d");
    return Call(op, {data}, Attrs(attrs), {});
  }

  Expr VisitExpr_(const CallNode* n) {
    static const Op& ada_avg_pool = Op::Get("nn.adaptive_avg_pool2d");
    static const Op& ada_max_pool = Op::Get("nn.adaptive_max_pool2d");
    auto new_n = ExprMutator::VisitExpr_(n);
    if (n->op.same_as(ada_avg_pool) || n->op.same_as(ada_max_pool)) {
      Call call = Downcast<Call>(new_n);
      CHECK_EQ(call->args.size(), 1);
      const AdaptivePool2DAttrs* param = call->attrs.as<AdaptivePool2DAttrs>();
      std::string layout = param->layout;
      int h_idx = 2;
      int w_idx = 3;
      if (layout == "NHWC") {
        h_idx = 1;
        w_idx = 2;
      }
      auto data = call->args[0];
      auto data_shape = n->args[0]->type_as<TensorTypeNode>()->shape;
      CHECK_EQ(data_shape.size(), 4);
      int64_t data_h = data_shape[h_idx].as<IntImmNode>()->value;
      int64_t data_w = data_shape[w_idx].as<IntImmNode>()->value;
      int64_t out_h = 0;
      int64_t out_w = 0;
      auto output_size = param->output_size;
      if (output_size.size() == 2) {
        out_h = output_size[0].as<IntImmNode>()->value;
        out_w = output_size[1].as<IntImmNode>()->value;
      } else if (output_size.size() == 1) {
        out_h = output_size[0].as<IntImmNode>()->value;
        out_w = output_size[0].as<IntImmNode>()->value;
      } else {
        LOG(FATAL) << "adaptive pool out_size must be 2 or 1,current size is "
                   << output_size.size();
      }
      ICHECK_GT(out_h, 0);
      ICHECK_GT(out_w, 0);
      if (data_h % out_h == 0 && data_w % out_w == 0) {
        int stride_h = data_h / out_h;
        int stride_w = data_w / out_w;
        int kh = stride_h;
        int kw = stride_w;
        Array<IndexExpr> pool_size = Array<IndexExpr>({kh, kw});
        Array<IndexExpr> strides = Array<IndexExpr>({stride_h, stride_w});
        Array<IndexExpr> padding = Array<IndexExpr>({0, 0});
        const std::string layout = param->layout;
        const std::string out_layout = param->out_layout;
        if (n->op.same_as(ada_max_pool)) {
          return this->MakeMaxPool(data, pool_size, strides, padding, layout, out_layout);
        } else {
          return this->MakeAvgPool(data, pool_size, strides, padding, layout, out_layout);
        }
      }
    }
    return new_n;
  }
};

Expr ConvertAdaptivePoolToNormPoolOps(const Expr& e) {
  return ConvertAdaptivePoolToNormPool().Mutate(e);
}

namespace transform {

Pass ConvertAdaptivePoolToNormPoolOps() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(relay::ConvertAdaptivePoolToNormPoolOps(f));
      };
  return CreateFunctionPass(pass_func, 3, "ConvertAdaptivePoolToNormPoolOps", {});
}

TVM_REGISTER_GLOBAL("relay._transform.ConvertAdaptivepoolToNormpool")
    .set_body_typed(ConvertAdaptivePoolToNormPoolOps);

}  // namespace transform
}  // namespace relay
}  // namespace tvm
