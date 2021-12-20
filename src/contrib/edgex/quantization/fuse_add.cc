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
 *
 * \file src/relay/transforms/fuse_add.cc
 *
 * \brief This is a backend-aware optimization pass.
 *   Fuse add op into a bias_add or convert it to bias_add.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/transform.h>

#include "../../../relay/transforms/pattern_utils.h"

namespace tvm {
namespace relay {

using namespace relay::transform;

static Expr FoldConstantOpt(const Expr& expr) {
  auto mod = IRModule::FromExpr(expr);
  mod = transform::FoldConstant()(mod);
  auto entry_func = Downcast<Function>(mod->Lookup("main"));
  return expr.as<FunctionNode>() == nullptr ? entry_func->body : entry_func;
}

class FuseAdd : public ExprMutator {
 public:
  FuseAdd()
      : add_op_(Op::Get("add")),
        dense_op_(Op::Get("nn.dense")),
        conv2d_op_(Op::Get("nn.conv2d")),
        bias_add_op_(Op::Get("nn.bias_add")) {}

  Expr VisitExpr_(const CallNode* n) {
    auto new_n = ExprMutator::VisitExpr_(n);
    if (n->op == add_op_ && (new_n.as<CallNode>()->args[1].as<ConstantNode>() ||
                             new_n.as<CallNode>()->args[1].as<VarNode>())) {
      if (ref_counter_[n->args[0].as<CallNode>()] > 1) {
        return new_n;
      }

      auto shape = n->args[1]->checked_type().as<TensorTypeNode>()->shape;

      auto arg0 = new_n.as<CallNode>()->args[0].as<CallNode>();
      if (!arg0) return new_n;
      if (arg0->op == bias_add_op_) {
        int axis = arg0->attrs.as<BiasAddAttrs>()->axis;
        if ((shape.size() == 3 && axis == 1 && shape[1].as<tvm::IntImmNode>()->value == 1 &&
             shape[2].as<tvm::IntImmNode>()->value == 1) ||
            (shape.size() == 1 && axis == 3)) {
          auto bias_weight = arg0->args[1];
          auto add_weight = new_n.as<CallNode>()->args[1];
          add_weight =
              Reshape(add_weight, {static_cast<int>(shape[0].as<tvm::IntImmNode>()->value)});
          auto new_weight = FoldConstantOpt(Add(bias_weight, add_weight));
          return Call(bias_add_op_, {arg0->args[0], new_weight}, arg0->attrs, arg0->type_args);
        }
      } else if (arg0->op == dense_op_ || arg0->op == conv2d_op_) {
        auto op_shape = n->args[0]->checked_type().as<TensorTypeNode>()->shape;
        int axis = op_shape.size() - shape.size();
        auto add_weight = new_n.as<CallNode>()->args[1];
        if (shape.size() > 1) {
          for (uint i = 1; (i + axis) < op_shape.size(); i++) {
            if (shape[i].as<tvm::IntImmNode>()->value != 1) {
              return new_n;
            }
          }
          add_weight =
              Reshape(add_weight, {static_cast<int>(shape[0].as<tvm::IntImmNode>()->value)});
          add_weight = FoldConstantOpt(add_weight);
        }
        auto attrs = make_object<BiasAddAttrs>();
        attrs->axis = axis;
        return Call(bias_add_op_, {new_n.as<CallNode>()->args[0], add_weight}, Attrs(attrs), {});
      }
    }
    return new_n;
  }

  Expr Mutate(const Expr& expr) {
    ref_counter_ = GetExprRefCount(expr);
    return this->VisitExpr(expr);
  }

 private:
  // Cache the following ops. They will be used in the passes repeatedly for
  // operator equivalence checking so that the registry lookup overhead can be
  // reduced.
  const Op& add_op_;
  const Op& dense_op_;
  const Op& conv2d_op_;
  const Op& bias_add_op_;

  std::unordered_map<Expr, Type, ObjectPtrHash, ObjectPtrEqual> ty_map_;
  // reference counter of an internal expr
  std::unordered_map<const Object*, size_t> ref_counter_;
};

Expr FuseAddFunc(const Expr& e) { return relay::FuseAdd().Mutate(e); }

namespace transform {

Pass FuseAdd() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return Downcast<Function>(FuseAddFunc(f)); };
  return CreateFunctionPass(pass_func, 3, "FuseAdd", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.FuseAdd").set_body_typed(FuseAdd);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
