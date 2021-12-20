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
 * \file fuse_reshape_squeeze.cc
 * \brief fuse reshape + squeeze.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/transform.h>

#include "../../../relay/transforms/pattern_utils.h"

namespace tvm {
namespace relay {

class FuseReshapeSqueeze : public ExprMutator {
 public:
  FuseReshapeSqueeze() {}

 public:
  Expr VisitExpr_(const CallNode* n) {
    static const Op& reshape = Op::Get("reshape");
    static const Op& squeeze = Op::Get("squeeze");

    auto new_n = ExprMutator::VisitExpr_(n);
    Expr arg;
    if (n->op.same_as(reshape)) {
      auto squeeze_node = n->args[0].as<CallNode>();
      if (squeeze_node && squeeze_node->op.same_as(squeeze)) {
        ICHECK(new_n.as<CallNode>()->args[0].as<CallNode>());
        arg = new_n.as<CallNode>()->args[0].as<CallNode>()->args[0];
      } else {
        return new_n;
      }
    } else if (n->op.same_as(squeeze)) {
      auto reshape_node = n->args[0].as<CallNode>();
      if (reshape_node && reshape_node->op.same_as(reshape)) {
        ICHECK(new_n.as<CallNode>()->args[0].as<CallNode>());
        arg = new_n.as<CallNode>()->args[0].as<CallNode>()->args[0];
      } else {
        return new_n;
      }
    } else {
      return new_n;
    }
    auto shape = n->checked_type().as<TensorTypeNode>()->shape;
    Array<tvm::Integer> new_shape;
    for (auto a : shape) {
      new_shape.push_back(a.as<IntImmNode>()->value);
    }
    return Reshape(arg, new_shape);
  }

  Expr Mutate(const Expr& expr) { return this->VisitExpr(expr); }
};

Expr FuseReshapeSqueezeOps(const Expr& e) { return FuseReshapeSqueeze().Mutate(e); }

namespace transform {

Pass FuseReshapeSqueeze() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(relay::FuseReshapeSqueezeOps(f));
      };

  return CreateFunctionPass(pass_func, 3, "FuseReshapeSqueeze", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.FuseReshapeSqueeze").set_body_typed(FuseReshapeSqueeze);

}  // namespace transform
}  // namespace relay
}  // namespace tvm
