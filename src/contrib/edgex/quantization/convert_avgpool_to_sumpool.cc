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
 * \file convert_avgpool_to_sumpool.cc
 * \brief Convert avgpool to sumpool + multiply.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/transform.h>

#include "../../../relay/transforms/pattern_utils.h"

namespace tvm {
namespace relay {

class ConvertAvgToSumpool : public ExprMutator {
 public:
  ConvertAvgToSumpool()
      : avg_pool2d_(Op::Get("nn.avg_pool2d")),
        global_avg_pool2d_(Op::Get("nn.global_avg_pool2d")),
        sum_pool2d_(Op::Get("nn.sum_pool2d")),
        global_sum_pool2d_(Op::Get("nn.global_sum_pool2d")) {}

  Expr VisitExpr_(const CallNode* n) {
    auto new_n = ExprMutator::VisitExpr_(n);
    if (n->op == avg_pool2d_) {
      auto pool_attrs = n->attrs.as<AvgPool2DAttrs>();
      std::string layout = pool_attrs->layout;
      std::string out_layout = pool_attrs->out_layout;
      auto pool_size = pool_attrs->pool_size;
      auto strides = pool_attrs->strides;
      auto dilation = pool_attrs->dilation;
      auto padding = pool_attrs->padding;
      auto ceil_mode = pool_attrs->ceil_mode;
      auto in_shape = n->args[0]->checked_type().as<TensorTypeNode>()->shape;
      ICHECK(in_shape.size() == 4);

      auto attrs = make_object<SumPool2DAttrs>();
      attrs->pool_size = std::move(pool_size);
      attrs->strides = std::move(strides);
      attrs->padding = std::move(padding);
      attrs->layout = std::move(layout);
      attrs->out_layout = std::move(out_layout);
      attrs->ceil_mode = ceil_mode;
      attrs->dilation = std::move(dilation);
      Expr sumpool = Call(sum_pool2d_, {new_n.as<CallNode>()->args[0]}, Attrs(attrs));

      int kh = attrs->pool_size[0].as<IntImmNode>()->value;
      int kw = attrs->pool_size[1].as<IntImmNode>()->value;
      float kernel_coef = 1. / kh / kw;
      auto mul_weight = MakeConstantScalar(DataType::Float(32), kernel_coef);
      return Multiply(sumpool, mul_weight);

    } else if (n->op == global_avg_pool2d_) {
      auto in_shape = n->args[0]->checked_type().as<TensorTypeNode>()->shape;
      auto pool_attrs = n->attrs.as<GlobalPool2DAttrs>();
      std::string layout = pool_attrs->layout;

      Expr sumpool = Call(global_sum_pool2d_, {new_n.as<CallNode>()->args[0]}, n->attrs);

      int h_index = layout.find('H');
      int w_index = layout.find('W');
      int height = in_shape[h_index].as<IntImmNode>()->value;
      int width = in_shape[w_index].as<IntImmNode>()->value;
      float kernel_coef = 1. / height / width;
      auto mul_weight = MakeConstantScalar(DataType::Float(32), kernel_coef);
      return Multiply(sumpool, mul_weight);
    }

    return new_n;
  }

  Expr Mutate(const Expr& expr) { return this->VisitExpr(expr); }

 private:
  const Op& avg_pool2d_;
  const Op& global_avg_pool2d_;
  const Op& sum_pool2d_;
  const Op& global_sum_pool2d_;
};

Expr ConvertAvgpool(const Expr& e) { return relay::ConvertAvgToSumpool().Mutate(e); }

namespace transform {

Pass ConvertAvgpoolToSumpool() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return Downcast<Function>(ConvertAvgpool(f)); };
  return CreateFunctionPass(pass_func, 3, "ConvertAvgpoolToSumpool", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.ConvertAvgpoolToSumpool")
    .set_body_typed(ConvertAvgpoolToSumpool);

}  // namespace transform
}  // namespace relay
}  // namespace tvm
