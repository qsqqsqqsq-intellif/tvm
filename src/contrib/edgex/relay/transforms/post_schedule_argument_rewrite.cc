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
 * \file post_schedule_argument_rewrite.cc
 * \brief Rewrite the relay arguments after tir schedule.
 */
#include <tvm/node/serialization.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/packed_func.h>

#include <deque>
#include <functional>
#include <vector>

#include "../../../../relay/transforms/device_aware_visitors.h"
#include "../../../../relay/transforms/expr_subst.h"
#include "../backend/schedule_cache.h"

namespace tvm {
namespace relay {

static const char* TARGET_PRIMFUNC_ATTR = "post_schedule_argument_rewrite";
static const char* TARGET_FORWARD_FUNCNAME = "forward";
static const char* TARGET_BACKWARD_FUNCNAME = "backward";

class PostScheduleArgumentRewriter : public transform::DeviceAwareExprMutator {
 public:
  explicit PostScheduleArgumentRewriter(IRModule m)
      : transform::DeviceAwareExprMutator(m), cache_(tec::ScheduleCache::Current()) {
    ICHECK(cache_.defined()) << "Must specify external schedule cache for this pass";
  }

 private:
  Expr DeviceAwareVisitExpr_(const CallNode* call) final {
    std::vector<Expr> call_args;
    bool args_unchanged = true;
    for (const auto& arg : call->args) {
      Expr e = this->VisitExpr(arg);
      call_args.push_back(e);
      args_unchanged &= e.same_as(arg);
    }
    auto default_result = [args_unchanged, call, &call_args](const Expr& op) {
      return args_unchanged && op.same_as(call->op)
                 ? GetRef<Call>(call)
                 : Call(op, call_args, call->attrs, call->type_args, call->span);
    };
    if (!call->op->IsInstance<FunctionNode>()) {
      return default_result(call->op);
    }
    Function origin_function = Downcast<Function>(call->op);

    // execute tir schedule ahead and extract primfunc annotation
    // note that we should use rewritten function as key
    SEScope se_scope = GetSEScope(GetRef<Call>(call));
    ICHECK(!se_scope->IsFullyUnconstrained())
        << "Can not determine target, should run PlanDevices() first";
    Target target = se_scope->target;
    ICHECK(target.defined());

    auto pair = cache_.Lower(origin_function, target);
    int64_t cache_key = pair.first;
    origin_function = WithAttr(origin_function, "ScheduleCacheKey", Integer(cache_key));

    auto cfunc = pair.second;
    if (!cfunc->prim_func.defined()) {
      return default_result(origin_function);
    }
    tir::PrimFunc prim_func = cfunc->prim_func.value();
    if (!prim_func->attrs.defined()) {
      return default_result(origin_function);
    }
    auto it = prim_func->attrs->dict.find(TARGET_PRIMFUNC_ATTR);
    if (it == prim_func->attrs->dict.end()) {
      return default_result(origin_function);
    }

    // extract relay forward/backward transform function for arguments
    ICHECK((*it).second->IsInstance<runtime::StringObj>());
    std::string json = Downcast<runtime::String>((*it).second);
    ObjectRef rewrite_spec = LoadJSON(json);
    ICHECK(rewrite_spec->IsInstance<IRModuleNode>());
    auto mod = Downcast<IRModule>(rewrite_spec);
    Function forward_func = Downcast<Function>(mod->Lookup(TARGET_FORWARD_FUNCNAME));
    Function backward_func = Downcast<Function>(mod->Lookup(TARGET_BACKWARD_FUNCNAME));

    // extract updated argument types
    auto tuple_type = forward_func->ret_type;
    ICHECK(tuple_type.defined() && tuple_type->IsInstance<TupleTypeNode>());
    Array<Type> new_arg_types = Downcast<TupleType>(tuple_type)->fields;

    // transform functions take output buffer as a function param
    // thus we should subtract the relay output num from the count
    size_t output_num = cfunc->outputs.size();
    size_t origin_arg_num = call_args.size();
    size_t new_arg_num = new_arg_types.size() - output_num;

    Array<Expr> new_args = GetTransformedArgs(forward_func, call_args, origin_arg_num, new_arg_num);
    std::unordered_set<Expr, StructuralHash, StructuralEqual> new_arg_dict(new_args.begin(),
                                                                           new_args.end());
    for (const Expr& arg : call_args) {
      // we do not support rewrite non-leaf argument now
      if (!arg->IsInstance<ConstantNode>() && !arg->IsInstance<VarNode>()) {
        ICHECK(new_arg_dict.count(arg))
            << "Non-leaf argument " << arg << " is rewritten by post schedule transform";
      }
    }

    // Convert to:
    //     new_v0, new_v1, ... = forward(origin_v0, origin_v1, ...)
    //     def new_func(new_p0, new_p1, ...):
    //         recover_v0, recover_v1, ... = backward(new_p0, new_p1, ...)
    //         return origin_func(recover_v0, recover_v1, ...)
    //     return new_func(new_v0, new_v1, ...)

    Array<Var> new_arg_vars;
    for (size_t i = 0; i < new_arg_num; ++i) {
      // prefix `p` is fuse convention and required by passes like defuse
      new_arg_vars.push_back(Var("p" + std::to_string(i), new_arg_types[i]));
    }
    Array<Expr> recovered_origin_vars =
        GetTransformedArgs(backward_func, Array<Expr>(new_arg_vars.begin(), new_arg_vars.end()),
                           new_arg_num, origin_arg_num);
    auto new_func_node = make_object<FunctionNode>(*origin_function.get());
    new_func_node->params = new_arg_vars;

    // remove kPrimitive tag for origin function
    if (origin_function->HasNonzeroAttr(attr::kPrimitive)) {
      auto origin_n = origin_function.CopyOnWrite();
      origin_n->attrs.CopyOnWrite()->dict.erase(attr::kPrimitive);
    }
    new_func_node->body = Call(origin_function, recovered_origin_vars);

    // mark transformed relay function as fused: kPrimitive
    if (new_func_node->attrs.defined()) {
      new_func_node->attrs.CopyOnWrite()->dict.Set(attr::kPrimitive, tvm::Integer(1));
    } else {
      Map<String, ObjectRef> dict = {{attr::kPrimitive, tvm::Integer(1)}};
      new_func_node->attrs = DictAttrs(dict);
    }

    // drop TARGET_PRIMFUNC_ATTR
    prim_func.CopyOnWrite()->attrs.CopyOnWrite()->dict.erase(TARGET_PRIMFUNC_ATTR);

    // update relay function ccache
    Array<Type> new_relay_arg_types;
    auto cn = cfunc.CopyOnWrite();
    cn->prim_func = prim_func;
    cn->inputs = Array<te::Tensor>();
    for (size_t i = 0; i < new_arg_num; ++i) {
      auto ttype = new_arg_types[i].as<TensorTypeNode>();
      cn->inputs.push_back(
          te::placeholder(ttype->shape, ttype->dtype, new_arg_vars[i]->name_hint()));
      new_relay_arg_types.push_back(new_arg_types[i]);
    }
    auto new_ftype = make_object<FuncTypeNode>(*origin_function->checked_type().as<FuncTypeNode>());
    new_ftype->arg_types = new_relay_arg_types;
    cn->prim_fn_var->checked_type_ = FuncType(new_ftype);

    // add schedule cache for new relay function
    cache_.AddSchedule(cache_key, cfunc);

    return std::move(Call(Function(new_func_node), new_args));
  }

  /*! \brief do lambda beta reduction, note the function can take more than `n_before` params and
   * more than `n_after` return fields */
  Array<Expr> GetTransformedArgs(Function arg_transform_func, const Array<Expr>& args,
                                 size_t n_before, size_t n_after) {
    ICHECK(arg_transform_func->params.size() >= args.size() && args.size() == n_before)
        << "Illegal transform function: \n"
        << AsText(arg_transform_func) << "\nargs: " << args;
    std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> subst_map;
    for (size_t i = 0; i < n_before; ++i) {
      subst_map[arg_transform_func->params[i]] = args[i];
    }
    ICHECK(arg_transform_func->body->IsInstance<TupleNode>());
    Tuple tuple = Downcast<Tuple>(arg_transform_func->body);
    ICHECK(tuple->fields.size() >= n_after);
    Array<Expr> results;
    for (size_t i = 0; i < n_after; ++i) {
      Expr expr = ExprSubst(tuple->fields[i], subst_map);
      results.push_back(expr);
    }
    return results;
  }

  tec::ScheduleCache cache_;
};

namespace transform {

Pass PostScheduleArgumentRewrite() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(relay::PostScheduleArgumentRewriter(m).Mutate(f));
      };
  return CreateFunctionPass(pass_func, 3, "PostScheduleArgumentRewrite", {});
}

TVM_REGISTER_GLOBAL("relay.edgex.transform.PostScheduleArgumentRewrite")
    .set_body_typed(PostScheduleArgumentRewrite);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
