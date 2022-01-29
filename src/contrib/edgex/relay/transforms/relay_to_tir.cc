
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
#include <tvm/ir/function.h>
#include <tvm/relay/attrs/call.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/function.h>
#include <tvm/relay/op_strategy.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/memory.h>
#include <tvm/target/target_kind.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/topi/tags.h>

#include "../../../../relay/backend/te_compiler_cache.h"
#include "../../../../relay/backend/utils.h"
#include "../../../../relay/op/call/call.h"
#include "../../../../relay/op/memory/device_copy.h"
#include "../../../../relay/transforms/device_aware_visitors.h"
#include "../../tir/attrs.h"
#include "../../tir/transform/edgex_transform.h"

namespace tvm {
namespace relay {

struct RelayToTIRConfig {
  /*! \brief Whether inline primfuncs of sub-op in fused function */
  bool inline_primfunc{true};
  /*! \brief Whether raise error on multi-anchor ops in fused function */
  bool allow_multi_anchor{true};
  /*! \brief verbosity */
  bool verbose{false};
  /*! \brief Whether also rewrite primitives on non-device target */
  bool rewrite_device_only{true};
  /*! \brief unique name getter */
  std::function<std::string(const std::string& cand)> renamer;
};

/*! \brief Converter from relay op attrs to an annotation dict acceptable by tir */
class RelayOpAttrToDict : public AttrVisitor {
 public:
  Map<String, ObjectRef> Convert(Attrs op_attrs) {
    dict_.clear();
    if (op_attrs.defined()) {
      auto n = const_cast<BaseAttrsNode*>(op_attrs.get());
      n->VisitAttrs(this);
    }
    return dict_;
  }

 private:
  void Visit(const char* key, double* value) final {
    dict_.Set(key, FloatImm(DataType::Float(64), *value));
  }
  void Visit(const char* key, int64_t* value) final {
    dict_.Set(key, IntImm(DataType::Int(64), *value));
  }
  void Visit(const char* key, uint64_t* value) final {
    dict_.Set(key, IntImm(DataType::UInt(64), *value));
  }
  void Visit(const char* key, int* value) final {
    dict_.Set(key, IntImm(DataType::Int(sizeof(int) * 8), *value));
  }
  void Visit(const char* key, bool* value) final { dict_.Set(key, Bool(*value)); }
  void Visit(const char* key, std::string* value) final { dict_.Set(key, String(*value)); }
  void Visit(const char* key, void** value) final {
    // skip ptr attribute
  }
  void Visit(const char* key, DataType* value) final { dict_.Set(key, PrimType(*value)); }
  void Visit(const char* key, runtime::NDArray* value) final {
    // skip NDArray attribute
  }
  void Visit(const char* key, runtime::ObjectRef* value) final {
    if (const ArrayNode* arr = value->as<ArrayNode>()) {
      dict_.Set(key, Array<ObjectRef>(arr->begin(), arr->end()));
    }
    // skip general objects
  }
  Map<String, ObjectRef> dict_;
};

/*! \brief helper mutator to add relay subcall op information to primfunc */
class SubfuncBlockAnnotator : public tir::StmtExprMutator {
 public:
  explicit SubfuncBlockAnnotator(const Map<String, ObjectRef>& attrs) : attrs_(attrs) {}

  tir::PrimFunc Mutate(tir::PrimFunc subfunc) {
    if (const auto* root = subfunc->body.as<tir::BlockRealizeNode>()) {
      root_block_ = root->block.get();
    }
    auto n = subfunc.CopyOnWrite();
    n->body = VisitStmt(subfunc->body);
    return GetRef<tir::PrimFunc>(n);
  }

 private:
  tir::Stmt VisitStmt_(const tir::BlockNode* block) {
    if (block == root_block_) {
      return tir::StmtExprMutator::VisitStmt_(block);
    }
    auto n = CopyOnWrite(block);
    for (const auto& p : attrs_) {
      n->annotations.Set(p.first, p.second);
    }
    return tir::Block(n);
  }

  const Map<String, ObjectRef>& attrs_;
  const tir::BlockNode* root_block_{nullptr};
};

/*! \brief Convert primitive relay function call to call lowered
 * During visiting, we presume each relay expr coresponds to one of tir objects below
 * (1) relay constant -> IntImm/FloatImm
 * (2) relay tuple -> list of corresponding tir objects of each tuple field
 * (3) other -> var representing one buffer's data
 * The implementation try to encode these cases into `Array<PrimExpr>`
 */
class PrimitiveCallLowering : public backend::MemoizedExprTranslator<Array<PrimExpr>> {
 public:
  explicit PrimitiveCallLowering(IRModule module, Target target, const RelayToTIRConfig& config)
      : module_(module), target_(target), config_(config) {}

  /*! \brief rewrite entrance */
  Expr operator()(const Function& relay_func, const Array<Expr>& args, const Call& origin_call) {
    GlobalVar gv = CreatePrimFunc(relay_func);
    CallLoweredAttrs call_lowered_attrs;
    call_lowered_attrs.metadata.Set("relay_attrs", origin_call->attrs);
    call_lowered_attrs.metadata.Set("relay_anchor_op", anchor_op_);
    call_lowered_attrs.metadata.Set("relay_anchor_attrs", anchor_attrs_);
    call_lowered_attrs.metadata.Set("EdgeXRelayToTIR", Integer(1));
    return std::move(
        CallLowered(std::move(gv), args, std::move(call_lowered_attrs), origin_call->span));
  }

 private:
  /*! \brief convert relay fused function to tir PrimFunc, return corresponding GlobalVar in module
   */
  GlobalVar CreatePrimFunc(const Function& relay_func) {
    // Step (1): build input primfunc parameters
    size_t n_input_params = 0;
    std::unordered_set<const tir::VarNode*> global_buffer_vars;
    for (Var relay_param : relay_func->params) {
      Array<PrimExpr> tir_args;
      for (const auto& ty : FlattenTupleType(relay_param->checked_type())) {
        tir::Var param;
        if (const TensorTypeNode* tensor_ty = ty.as<TensorTypeNode>()) {
          param = tir::Var("arg_" + std::to_string(n_input_params), DataType::Handle());
          tir::Buffer buffer =
              tir::decl_buffer(tensor_ty->shape, tensor_ty->dtype,
                               "placeholder_" + std::to_string(n_input_params), "global");
          global_buffer_map_.Set(param, buffer);
          var2buffer_.Set(buffer->data, buffer);
          global_buffer_vars.insert(buffer->data.get());
          tir_args.push_back(buffer->data);
        } else if (const PrimTypeNode* prim_ty = ty.as<PrimTypeNode>()) {
          param = tir::Var("arg_" + std::to_string(n_input_params), prim_ty->dtype);
          tir_args.push_back(param);
        }
        global_params_.push_back(param);
        n_input_params += 1;
      }
      memo_[relay_param] = tir_args;
    }

    // Step (2): visit function body
    readable_name_stream_ << "edgex_fused";
    auto outputs = this->VisitExpr(relay_func->body);
    auto candidate_name = readable_name_stream_.str();
    constexpr static size_t kMaxFuncNameLength = 80;
    if (candidate_name.size() > kMaxFuncNameLength) {
      std::stringstream truncated_name;
      truncated_name << candidate_name.substr(0, kMaxFuncNameLength);
      truncated_name << "_" << std::hash<std::string>{}(candidate_name) << "_";
      candidate_name = truncated_name.str();
    }

    // Step (3): build output primfunc parameters
    size_t n_output_params = 0;
    for (const PrimExpr& e : outputs) {
      ICHECK(e->IsInstance<tir::VarNode>()) << "Lower result should be a buffer var";
      tir::Var buffer_var = Downcast<tir::Var>(e);
      auto it = var2buffer_.find(buffer_var);
      ICHECK(it != var2buffer_.end());
      tir::Var param = tir::Var("res_" + std::to_string(n_output_params), DataType::Handle());
      global_buffer_map_.Set(param, (*it).second);
      if (global_buffer_vars.count(buffer_var.get())) {
        // currently aliased buffer is not supported
        LOG(FATAL) << "Aliased param buffer: " << (*it).second << " param is " << param;
      }
      global_buffer_vars.insert(buffer_var.get());
      global_params_.push_back(param);
      n_output_params += 1;
    }
    for (const auto& p : var2buffer_) {
      if (!global_buffer_vars.count(p.first.get())) {
        alloc_buffers_.push_back(p.second);
      }
    }

    // Step (3): build global primfunc object
    tir::SeqStmt seq = tir::SeqStmt(sub_calls_);
    tir::Block root_block({}, {}, {}, "root", seq, NullOpt, alloc_buffers_);
    tir::BlockRealize root_realize({}, tir::const_true(), root_block);
    tir::PrimFunc primfunc(global_params_, root_realize, VoidType(), global_buffer_map_);

    // Step (4): inline all invocations to primfuncs of subops
    if (config_.inline_primfunc) {
      primfunc = tvm::tir::transform::InlinePrimFuncCalls(primfunc, module_);
    }

    // Step (5): add function annotations and update module
    ICHECK(config_.renamer);
    std::string symbol_name = config_.renamer(candidate_name);
    return AddPrimFunc(/*name=*/symbol_name, /*primfunc=*/primfunc,
                       /*checked_type=*/relay_func->checked_type_, /*is_extern=*/true);
  }

  Array<PrimExpr> VisitExpr_(const VarNode* op) final {
    LOG(FATAL) << "Unexpected free variable " << PrettyPrint(GetRef<Var>(op));
    return {};
  }

  Array<PrimExpr> VisitExpr_(const ConstantNode* op) final {
    using tir::make_const;
    ICHECK(op->is_scalar());
    void* data = op->data->data;
    DataType dtype = DataType(op->data->dtype);
    PrimExpr imm;
    if (dtype == DataType::Int(32)) {
      imm = make_const(dtype, static_cast<const int32_t*>(data)[0]);
    } else if (dtype == DataType::Int(64)) {
      imm = make_const(dtype, static_cast<const int64_t*>(data)[0]);
    } else if (dtype == DataType::Float(32)) {
      imm = make_const(dtype, static_cast<const float*>(data)[0]);
    } else if (dtype == DataType::Float(64)) {
      imm = make_const(dtype, static_cast<const double*>(data)[0]);
    } else if (dtype == DataType::Bool()) {
      imm = make_const(dtype, static_cast<const uint8_t*>(data)[0]);
    } else {
      LOG(FATAL) << "can not handle data type of constant: " << dtype;
    }
    return {imm};
  }

  Array<PrimExpr> VisitExpr_(const CallNode* call_node) final {
    ICHECK(call_node->op.as<OpNode>()) << "Primitive function only allows call into primitive ops";
    Op op = Downcast<Op>(call_node->op);
    Array<PrimExpr> tir_inputs;

    // visit arguments
    for (Expr arg : call_node->args) {
      if (arg->checked_type().as<TupleTypeNode>()) {
        ICHECK_EQ(call_node->args.size(), 1U)
            << "Only functions with a single tuple input are allowed";
      }
      for (PrimExpr e : VisitExpr(arg)) {
        tir_inputs.push_back(e);
      }
    }

    // currently we can only use TE based op implementation to create primfunc of subcall
    auto te_result = ConvertRelayToTE(call_node, op, tir_inputs);
    tir::Call subcall = te_result.first;
    const Array<PrimExpr>& tir_outputs = te_result.second;

    // record current subcall
    sub_calls_.push_back(tir::Evaluate(subcall));
    readable_name_stream_ << "_" << op->name;
    return tir_outputs;
  }

  Array<PrimExpr> VisitExpr_(const FunctionNode* op) final {
    LOG(FATAL) << "Primitive Functions can not contain nested functions.";
    return {};
  }

  Array<PrimExpr> VisitExpr_(const LetNode* op) final {
    Array<PrimExpr> val = VisitExpr(op->value);
    ICHECK(!memo_.count(op->var));
    memo_[op->var] = val;
    return VisitExpr(op->body);
  }

  Array<PrimExpr> VisitExpr_(const TupleNode* op) final {
    Array<PrimExpr> fields;
    for (Expr field : op->fields) {
      ICHECK(field->checked_type().as<TensorTypeNode>()) << "Only allow Tuple of Tensor";
      Array<PrimExpr> res = VisitExpr(field);
      ICHECK_EQ(res.size(), 1);
      fields.push_back(res[0]);
    }
    return fields;
  }

  Array<PrimExpr> VisitExpr_(const TupleGetItemNode* op) final {
    const auto* tuple_type = op->tuple->type_as<TupleTypeNode>();
    Array<PrimExpr> tuple = VisitExpr(op->tuple);
    ICHECK_EQ(tuple_type->fields.size(), tuple.size());
    ICHECK_GE(op->index, 0);
    ICHECK_LT(static_cast<size_t>(op->index), tuple.size());
    return {tuple[op->index]};
  }

  /*! \brief Run te level conversion for relay sub-op, return tir level subcall and output arguments
   */
  std::pair<tir::Call, Array<PrimExpr>> ConvertRelayToTE(const CallNode* relay_call, const Op& op,
                                                         const Array<PrimExpr>& tir_inputs) {
    // get lower implementation: "relay.backend.lower_call"
    static auto fpattern = Op::GetAttrMap<TOpPattern>("TOpPattern");
    static auto flower_call = tvm::runtime::Registry::Get("relay.backend.lower_call");
    ICHECK(flower_call) << "relay.backend.lower_call is not registered.";

    // create te input placeholders
    Array<te::Tensor> te_inputs;
    for (const PrimExpr& e : tir_inputs) {
      te::Tensor tensor;
      if (tir::is_const_number(e)) {
        tensor = te::compute(
            {}, [&](const Array<tvm::tir::Var>&) { return e; }, "compile_engine_const",
            topi::kBroadcast);
      } else {
        tir::Var buffer_var = Downcast<tir::Var>(e);
        tir::Buffer buffer = var2buffer_.at(buffer_var);
        tensor = tvm::te::placeholder(buffer->shape, buffer->dtype);
      }
      te_inputs.push_back(tensor);
    }

    OpImplementation impl;
    tec::LoweredOutput lowered_out;
    {
      With<Target> target_scope(target_);
      lowered_out = (*flower_call)(GetRef<Call>(relay_call), te_inputs, target_);
    }
    Array<te::Tensor> te_outputs = lowered_out->outputs;

    // update anchor op info
    int op_pattern = fpattern[op];
    if (op_pattern >= kCommReduce && !config_.allow_multi_anchor) {
      ICHECK(!anchor_op_.defined() || anchor_op_pattern_ < kCommReduce)
          << "Cannot apply TOPI schedule to a primitive function with two complicated ops"
          << " anchor=" << anchor_op_ << " current=" << op;
    }
    if (op_pattern >= anchor_op_pattern_) {
      anchor_op_ = op;
      anchor_attrs_ = relay_call->attrs;
      anchor_op_pattern_ = op_pattern;
      anchor_implementation_ = impl;
    }
    if (te_outputs.size() != 1) {
      const auto* tuple_type = relay_call->checked_type().as<TupleTypeNode>();
      ICHECK(tuple_type) << "Expected output to be a tuple type "
                         << PrettyPrint(relay_call->checked_type());
      ICHECK_EQ(tuple_type->fields.size(), te_outputs.size());
    }

    // collect te/tir call arguments
    Array<te::Tensor> all_te_tensors;
    Array<PrimExpr> all_tir_args;
    Array<PrimExpr> tir_outputs;
    for (size_t i = 0; i < tir_inputs.size(); ++i) {
      if (tir::is_const_number(tir_inputs[i])) continue;  // constants are folded
      all_te_tensors.push_back(te_inputs[i]);
      all_tir_args.push_back(tir_inputs[i]);
    }
    for (const te::Tensor& tensor : te_outputs) {
      all_te_tensors.push_back(tensor);
      tir::Buffer buffer =
          tir::decl_buffer(tensor->shape, tensor->dtype, tensor->GetNameHint(), "global");
      tir::Var buffer_var = buffer->data;
      var2buffer_.Set(buffer_var, buffer);
      all_tir_args.push_back(buffer_var);
      tir_outputs.push_back(buffer_var);
    }

    // create sub-op's primfunc
    const auto* f_create_func = runtime::Registry::Get("te.CreatePrimFunc");
    ICHECK(f_create_func) << "te.CreatePrimFunc is not registered";
    tir::PrimFunc subfunc = (*f_create_func)(all_te_tensors);

    // annotate sub-op's primfunc with relay graph level information
    subfunc = AnnotateSubPrimFunc(subfunc, op, relay_call->attrs);

    ICHECK(config_.renamer);
    std::string subfunc_name = config_.renamer("subcall_" + op->name);
    GlobalVar gv = AddPrimFunc(/*name=*/subfunc_name, /*primfunc=*/subfunc, /*checked_type=*/Type(),
                               /*is_extern=*/!config_.inline_primfunc);
    tir::Call subcall(DataType::Void(), gv, all_tir_args);
    return {subcall, tir_outputs};
  }

  /*! \brief annotate sub-op's primfunc with relay graph level information */
  tir::PrimFunc AnnotateSubPrimFunc(tir::PrimFunc subfunc, const Op& relay_op,
                                    const Attrs& op_attrs) {
    // clear all attrs of subfunc
    subfunc.CopyOnWrite()->attrs = DictAttrs();

    // collect extra attrs to annotate on subfunc's blocks
    Map<String, ObjectRef> annotations;
    annotations.Set(tir::attr::relay_op_name, relay_op->name);

    Map<String, ObjectRef> op_attr_dict = RelayOpAttrToDict().Convert(op_attrs);
    // TODO(bxq): support print and parse dict
    // annotate_attrs.Set(tir::attr::relay_op_attrs, op_attr_dict);
    for (const auto& p : op_attr_dict) {
      annotations.Set(std::string(tir::attr::relay_op_attrs) + "." + p.first, p.second);
    }

    // relay_call->attrs
    SubfuncBlockAnnotator annotator(annotations);
    subfunc = annotator.Mutate(subfunc);
    return subfunc;
  }

  /*! \brief helper to update a tir primfunc into module */
  GlobalVar AddPrimFunc(const std::string& name, tir::PrimFunc primfunc, Type checked_type,
                        bool is_extern) {
    if (is_extern) {
      primfunc = WithAttr(primfunc, tvm::attr::kGlobalSymbol, String(name));
      primfunc = WithAttr(primfunc, tvm::attr::kTarget, target_);
    }
    GlobalVar gv(name);
    if (checked_type.defined()) {
      gv->checked_type_ = checked_type;
    }
    module_->Update(gv, primfunc);
    return gv;
  }

  /*! \brief global primfunc params */
  Array<tir::Var> global_params_;

  /*! \brief global buffer map of the */
  Map<tir::Var, tir::Buffer> global_buffer_map_;

  /*! \brief subcall to primfuncs in dfs order */
  std::vector<tir::Stmt> sub_calls_;

  /*! \brief internally allocated buffers */
  Array<tir::Buffer> alloc_buffers_;

  /*! \brief var to buffer map */
  Map<tir::Var, tir::Buffer> var2buffer_;

  /*! \brief IRModule object */
  IRModule module_;

  /*! \brief Lower target */
  tvm::Target target_;

  // anchor informations
  Op anchor_op_;
  Attrs anchor_attrs_;
  int anchor_op_pattern_{0};
  OpImplementation anchor_implementation_;

  /*! \brief lower name of fused relay function */
  std::ostringstream readable_name_stream_;

  /*! \brief pass config */
  const RelayToTIRConfig& config_;

  /*! \brief function name counter */
  std::unordered_map<std::string, int> funcname_counter_;
};

class Relay2TIRConverter : public transform::DeviceAwareExprMutator {
 public:
  Relay2TIRConverter(IRModule module, const RelayToTIRConfig& config)
      : transform::DeviceAwareExprMutator(module), module_(module), config_(config) {
    auto tgt_kind_opt = TargetKind::Get("edgex");
    ICHECK(tgt_kind_opt.defined()) << "Target kind edgex not registered";
    device_target_ = tgt_kind_opt.value();
  }

  IRModule MutateModule(String entry_name) {
    GlobalVar main_gv = module_->GetGlobalVar(entry_name);
    Function main_func = Downcast<Function>(module_->Lookup(main_gv));
    Function new_func = Downcast<Function>(Mutate(main_func));
    auto new_module = module_.CopyOnWrite();
    new_module->Update(main_gv, new_func);
    return GetRef<IRModule>(new_module);
  }

 private:
  Expr DeviceAwareVisitExpr_(const CallNode* call) final {
    Array<Expr> call_args;
    bool unchanged = true;
    for (const auto& arg : call->args) {
      Expr e = VisitExpr(arg);
      call_args.push_back(e);
      unchanged &= e.same_as(arg);
    }

    // special case for device copy
    if (const auto* fused_device_copy = call->op.as<FunctionNode>()) {
      DeviceCopyProps device_copy_props = GetDeviceCopyProps(fused_device_copy->body);
      if (device_copy_props.body.defined()) {
        ICHECK_EQ(call_args.size(), 1);
        return DeviceCopy(call_args[0], device_copy_props.src_virtual_device,
                          device_copy_props.dst_virtual_device);
      }
    }

    // process fused primitive function
    for (;;) {
      if (!call->op->IsInstance<FunctionNode>()) {
        break;
      }
      if (!call->type_args.empty()) {
        // lowered functions cannot be polymorphic
        break;
      }
      Function func = Downcast<Function>(call->op);
      if (!func->HasNonzeroAttr(attr::kPrimitive)) {
        break;
      }

      VirtualDevice virtual_device = GetVirtualDevice(GetRef<Call>(call));
      ICHECK(!virtual_device->IsFullyUnconstrained())
          << "Can not get device target of current primitive function, maybe PlanDevices not run";
      Target target = virtual_device->target;
      ICHECK(target.defined());
      if (config_.rewrite_device_only && target->kind != device_target_) {
        // skip non-device function, delete kCompiler tag to fallback to normal compile
        Map<String, ObjectRef> new_dict = func->attrs->dict;
        new_dict.erase(attr::kCompiler);
        func.CopyOnWrite()->attrs.CopyOnWrite()->dict.erase(attr::kCompiler);
        return Call(func, call_args, call->attrs, call->type_args, call->span);
      }

      PrimitiveCallLowering lowerer(module_, target, config_);
      Expr lowered_call = lowerer(func, call_args, GetRef<Call>(call));
      // type annotation is neccesary since it could not be inferred
      lowered_call->checked_type_ = call->checked_type_;
      return lowered_call;
    }

    // fallback return for other cases
    if (unchanged) {
      return GetRef<Call>(call);
    } else {
      return Call(call->op, call->args, call->attrs, call->type_args, call->span);
    }
  }

  IRModule module_;
  const RelayToTIRConfig& config_;
  TargetKind device_target_;
};

namespace transform {

TVM_REGISTER_PASS_CONFIG_OPTION("edgex.relay_to_tir.verbose", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("edgex.relay_to_tir.allow_multi_anchor", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("edgex.relay_to_tir.inline_primfunc", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("edgex.relay_to_tir.rewrite_device_only", Bool);

// fwd declaration
Pass PostScheduleArgumentRewrite(bool is_legacy);

// default rename helper
static std::string DefaultRenamer(const std::string& candidate_name,
                                  std::unordered_map<std::string, int>* name_counter) {
  std::string name = candidate_name;
  for (size_t i = 0; i < name.length(); ++i) {
    if (name[i] == '.') name[i] = '_';
  }
  while (true) {
    auto it = name_counter->find(name);
    if (it == name_counter->end()) {
      (*name_counter)[name] = 1;
      return name;
    } else {
      std::ostringstream os;
      os << name << "_" << it->second;
      ++(it->second);
      name = os.str();
    }
  }
  return name;
}

transform::Pass EdgeXRelayToTIR(String entry_name, PackedFunc renamer, bool post_schedule_rewrite,
                                bool fold_constants) {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [=](IRModule module, transform::PassContext ctx) {
        RelayToTIRConfig config;
        config.allow_multi_anchor =
            ctx->GetConfig<Bool>("edgex.relay_to_tir.allow_multi_anchor").value_or(Bool(true));
        config.inline_primfunc =
            ctx->GetConfig<Bool>("edgex.relay_to_tir.inline_primfunc").value_or(Bool(true));
        config.verbose = ctx->GetConfig<Bool>("edgex.relay_to_tir.verbose").value_or(Bool(false));
        config.rewrite_device_only =
            ctx->GetConfig<Bool>("edgex.relay_to_tir.rewrite_device_only").value_or(Bool(true));

        std::unordered_map<std::string, int> name_counter;
        if (renamer != nullptr) {
          config.renamer = [renamer](const std::string& name) { return renamer(name); };
        } else {
          config.renamer = [&name_counter](const std::string& name) {
            return DefaultRenamer(name, &name_counter);
          };
        }
        auto relay_to_tir = Relay2TIRConverter(module, config);
        return relay_to_tir.MutateModule(entry_name);
      };

  std::vector<Pass> passes;
  // basic conversions to call_lowered
  passes.push_back(tvm::transform::CreateModulePass(pass_func, 0, "EdgeXRelayToTIR", {}));
  // remove inlined tir funcs
  passes.push_back(RemoveUnusedFunctions({entry_name}));
  // sanity type check
  passes.push_back(InferType());
  if (post_schedule_rewrite) {
    // schedule -> lower -> rewrite relay arguments
    passes.push_back(PostScheduleArgumentRewrite(false));
    // sanity type check
    passes.push_back(InferType());
  }
  // eliminate argument conversions
  if (fold_constants) {
    passes.push_back(FoldConstant());
  }
  return tvm::transform::Sequential(passes);
}

TVM_REGISTER_GLOBAL("relay.edgex.transform.EdgeXRelayToTIR").set_body_typed(EdgeXRelayToTIR);

}  // namespace transform
}  // namespace relay
}  // namespace tvm
