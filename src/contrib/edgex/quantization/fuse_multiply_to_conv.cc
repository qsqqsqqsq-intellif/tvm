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
 * \file fuse_multiply_to_conv.cc
 * \brief fuse multiply into conv2d or dense.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/transform.h>

#include "../../../relay/transforms/pattern_utils.h"

namespace tvm {
namespace relay {

static Expr FoldConstantOpt(const Expr& expr) {
  auto mod = IRModule::FromExpr(expr);
  mod = transform::FoldConstant()(mod);
  auto entry_func = Downcast<Function>(mod->Lookup("main"));
  return expr.as<FunctionNode>() == nullptr ? entry_func->body : entry_func;
}

class FuseMultiplyToConv : public ExprMutator {
 public:
  FuseMultiplyToConv() {}

 public:
  Expr VisitExpr_(const CallNode* n) {
    static const Op& conv2d = Op::Get("nn.conv2d");
    static const Op& dense = Op::Get("nn.dense");
    static const Op& reshape = Op::Get("reshape");
    static const Op& squeeze = Op::Get("squeeze");
    static const Op& multiply = Op::Get("multiply");
    static const Op& flatten = Op::Get("nn.batch_flatten");
    static const Op& transpose = Op::Get("transpose");

    auto new_n = ExprMutator::VisitExpr_(n);
    if (n->op.same_as(conv2d)) {
      auto mul_node = new_n.as<CallNode>()->args[0].as<CallNode>();
      if (!mul_node || !mul_node->op.same_as(multiply)) {
        return new_n;
      }
      auto attr = n->attrs;
      auto mul_w = mul_node->args[1].as<ConstantNode>();
      if (!mul_w) {
        return new_n;
      }
      if (!mul_w->is_scalar()) {
        return new_n;
      }

      auto new_w = Multiply(new_n.as<CallNode>()->args[1], mul_node->args[1]);
      auto new_arg0 = new_n.as<CallNode>()->args[0].as<CallNode>()->args[0];
      auto new_arg1 = FoldConstantOpt(new_w);
      return Call(conv2d, {new_arg0, new_arg1}, attr);
    }

    if (n->op.same_as(dense)) {
      auto arg0 = new_n.as<CallNode>()->args[0].as<CallNode>();
      ICHECK(arg0);
      Expr mul_w;
      Expr new_arg0;
      if (arg0->op.same_as(multiply)) {
        auto mul_arg1 = arg0->args[1].as<ConstantNode>();
        if (!mul_arg1) {
          return new_n;
        }
        mul_w = arg0->args[1];
        if (!mul_w.as<ConstantNode>()->is_scalar()) {
          return new_n;
        }
        new_arg0 = new_n.as<CallNode>()->args[0];
      } else if (arg0->op.same_as(reshape) || arg0->op.same_as(squeeze) ||
                 arg0->op.same_as(flatten)) {
        Expr arg0_expr = arg0->args[0];
        const CallNode* a = arg0_expr.as<CallNode>();
        if (a && a->op.same_as(transpose)) {
          Expr tran_arg0 = a->args[0];
          auto t_arg0 = tran_arg0.as<CallNode>();
          if (t_arg0 && t_arg0->op.same_as(multiply)) {
            auto transpose_attrs = a->attrs.as<TransposeAttrs>();
            auto axes = transpose_attrs->axes;
            Expr new_tran = MakeTranspose(t_arg0->args[0], axes);
            arg0_expr = Multiply(new_tran, t_arg0->args[1]);
          } else {
            return new_n;
          }
        }
        auto b = arg0_expr.as<CallNode>();
        if (b && b->op.same_as(multiply)) {
          mul_w = b->args[1];
          auto mul_const = mul_w.as<ConstantNode>();
          if (!mul_const) {
            return new_n;
          }
          if (!mul_const->is_scalar()) {
            return new_n;
          }

          auto arg = b->args[0];
          new_arg0 = Call(arg0->op, {arg}, arg0->attrs);
        } else {
          return new_n;
        }
      } else {
        return new_n;
      }
      auto new_w = Multiply(n->args[1], mul_w);
      return Call(dense, {new_arg0, new_w}, n->attrs);
    }

    return new_n;
  }

  Expr Mutate(const Expr& expr) { return this->VisitExpr(expr); }
};

Expr FuseMultiplyToConvOps(const Expr& e) { return FuseMultiplyToConv().Mutate(e); }

namespace transform {

Pass FuseMultiplyToConv() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(relay::FuseMultiplyToConvOps(f));
      };

  return CreateFunctionPass(pass_func, 3, "FuseMultiplyToConv", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.FuseMultiplyToConv").set_body_typed(FuseMultiplyToConv);

}  // namespace transform
}  // namespace relay
}  // namespace tvm
