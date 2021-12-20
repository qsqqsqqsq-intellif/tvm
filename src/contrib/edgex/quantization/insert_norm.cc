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
 * \file src/relay/transforms/insert_norm.cc
 *
 * \brief This is a backend-aware optimization pass.
 *   Insert batchnorm in front of graph.
 */

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/transform.h>

#include <cmath>
#include <vector>

#include "../../../relay/transforms/pattern_utils.h"

namespace tvm {
namespace relay {

class InsertNorm : public ExprMutator {
 public:
  InsertNorm(Array<PrimExpr> mean, Array<PrimExpr> scale) {
    for (PrimExpr data : mean) {
      auto d = data.as<FloatImmNode>();
      if (d) {
        mean_.push_back(d->value);
      } else if (auto d = data.as<IntImmNode>()) {
        mean_.push_back(static_cast<float>(d->value));
      } else {
        ICHECK(0) << "mean must be float or int";
      }
    }

    for (PrimExpr data : scale) {
      auto d = data.as<FloatImmNode>();
      if (d) {
        scale_.push_back(std::pow(d->value, 2));
      } else if (auto d = data.as<IntImmNode>()) {
        scale_.push_back(static_cast<float>(std::pow(d->value, 2)));
      } else {
        ICHECK(0) << "scale must be float or int";
      }
    }
  }

  Expr VisitExpr_(const CallNode* n) {
    auto new_n = ExprMutator::VisitExpr_(n);
    auto new_call = new_n.as<CallNode>();

    int c_index = 1;
    if (auto arg0 = new_call->args[0].as<TupleNode>()) {
      bool hit = false;
      tvm::Array<Expr> new_args;
      for (Expr field : arg0->fields) {
        if (bn_insert_map_.count(field)) {
          new_args.push_back(bn_insert_map_[field]);
          hit = true;
        } else {
          new_args.push_back(field);
        }
      }

      if (hit) {
        return Call(new_call->op, new_args, new_call->attrs);
      }
    }

    if (bn_insert_map_.empty()) {
      int64_t chs = 1;
      if (auto arg0 = new_call->args[0].as<TupleNode>()) {
        ICHECK(new_call->op.as<OpNode>()->name == "concatenate");
        for (Expr field : arg0->fields) {
          if (auto input_value = field.as<VarNode>()) {
            auto shape = input_value->checked_type().as<TensorTypeNode>()->shape;
            if ((int64_t)this->mean_.size() == shape[3].as<IntImmNode>()->value) {
              c_index = 3;
            }
            chs = input_value->checked_type()
                      .as<TensorTypeNode>()
                      ->shape[c_index]
                      .as<IntImmNode>()
                      ->value;
            break;
          }
        }
      } else {
        ICHECK(n->args[0].as<VarNode>());
        auto shape = n->args[0].as<VarNode>()->checked_type().as<TensorTypeNode>()->shape;
        if ((int64_t)this->mean_.size() == shape[3].as<IntImmNode>()->value) {
          c_index = 3;
        }
        chs = n->args[0]
                  .as<VarNode>()
                  ->checked_type()
                  .as<TensorTypeNode>()
                  ->shape[c_index]
                  .as<IntImmNode>()
                  ->value;
      }

      // prepare attributes and args
      auto attrs = make_object<BatchNormAttrs>();
      attrs->axis = c_index;
      attrs->epsilon = 0.00001;
      attrs->center = true;
      attrs->scale = true;

      std::vector<float> gama_data(chs, 1.f);
      std::vector<float> beta_data(chs, 0.f);
      auto gamma = MakeConstantTensor(DataType::Float(32), {chs}, gama_data);
      auto beta = MakeConstantTensor(DataType::Float(32), {chs}, beta_data);
      auto mean = MakeConstantTensor(DataType::Float(32), {chs}, mean_);
      auto var = MakeConstantTensor(DataType::Float(32), {chs}, scale_);

      if (auto arg0 = new_call->args[0].as<TupleNode>()) {
        Array<Expr> new_args;
        for (Expr field : arg0->fields) {
          if (field.as<VarNode>()) {
            if (bn_insert_map_.count(field)) {
              Expr new_arg = bn_insert_map_[field];
              new_args.push_back(new_arg);
            } else {
              Expr bn = Call(Op::Get("nn.batch_norm"), {new_call->args[0], gamma, beta, mean, var},
                             tvm::Attrs(attrs));
              bn_insert_map_[n->args[0]] = GetField(bn, 0);
            }
          } else {
            new_args.push_back(field);
          }
        }
        return Call(new_call->op, new_args, new_call->attrs);
      }

      ICHECK(new_call->args[0].as<VarNode>());
      Expr bn = Call(Op::Get("nn.batch_norm"), {new_call->args[0], gamma, beta, mean, var},
                     tvm::Attrs(attrs));
      bn_insert_map_[n->args[0]] = GetField(bn, 0);

      Array<Expr> orig_left_args;
      orig_left_args.push_back(GetField(bn, 0));
      for (size_t i = 1; i < new_call->args.size(); i++) {
        orig_left_args.push_back(new_call->args[i]);
      }
      return Call(new_call->op, orig_left_args, new_call->attrs);
    }

    return new_n;
  }

  Expr Mutate(const Expr& expr) { return this->VisitExpr(expr); }

 private:
  std::vector<float> mean_;
  std::vector<float> scale_;
  std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> bn_insert_map_;
};

namespace transform {

Pass InsertNorm(Array<PrimExpr> mean, Array<PrimExpr> scale) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(relay::InsertNorm(mean, scale).Mutate(f));
      };
  return CreateFunctionPass(pass_func, 3, "InsertNorm", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.InsertNorm").set_body_typed(InsertNorm);

}  // namespace transform
}  // namespace relay
}  // namespace tvm
