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
 * \file decorate_device_scope.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../edgex_ir_utils.h"
#include "./edgex_transform.h"

namespace tvm {
namespace tir {

/*! \brief Substituter used to replace buffer in inlined primfunc body to global func's buffer. */
class PrimFuncBufferSubstituter : public StmtExprMutator {
 public:
  explicit PrimFuncBufferSubstituter(std::unordered_map<std::string, size_t>* buffer_name_cnt)
      : buffer_name_cnt_(buffer_name_cnt) {}

  /*! \brief Record buffer replacement */
  void AddBufferRemap(const Buffer& origin_buffer, const Buffer& new_buffer) {
    buffer_remap_[origin_buffer.get()] = new_buffer;
  }

  /*! \brief Record var replacement */
  void AddVarRemap(const Var& var, const PrimExpr& expr) { var_remap_[var.get()] = expr; }

  /*! \brief record updated buffer alloc and buffer match at root block. */
  Block root_block_;

 private:
  PrimExpr VisitExpr_(const VarNode* v) final {
    auto it = var_remap_.find(v);
    if (it != var_remap_.end()) {
      return it->second;
    }
    return StmtExprMutator::VisitExpr_(v);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* load) final {
    auto it = buffer_remap_.find(load->buffer.get());
    if (it != buffer_remap_.end()) {
      auto n = make_object<BufferLoadNode>(*load);
      n->buffer = it->second;
      return std::move(BufferLoad(n));
    }
    return StmtExprMutator::VisitExpr_(load);
  }

  Stmt VisitStmt_(const BufferStoreNode* store) final {
    auto it = buffer_remap_.find(store->buffer.get());
    if (it != buffer_remap_.end()) {
      auto n = CopyOnWrite(store);
      n->buffer = it->second;
      n->value = StmtExprMutator::VisitExpr(n->value);
      return std::move(BufferStore(n));
    }
    return StmtExprMutator::VisitStmt_(store);
  }

  Stmt VisitStmt_(const BlockRealizeNode* block_realize) final {
    // visit block realize fields
    Array<PrimExpr> iter_values;
    for (const PrimExpr& e : block_realize->iter_values) {
      iter_values.push_back(VisitExpr(e));
    }
    PrimExpr predicate = VisitExpr(block_realize->predicate);

    const BlockNode* block = block_realize->block.get();
    auto n = CopyOnWrite(block);
    // update block match buffers
    for (size_t i = 0; i < block->match_buffers.size(); ++i) {
      const auto& match = block->match_buffers[i];
      Buffer target = match->buffer;
      Buffer source = match->source->buffer;
      const Region& region = match->source->region;
      auto it = buffer_remap_.find(source.get());
      if (it != buffer_remap_.end()) {
        source = it->second;
      }
      std::string name = GetUniqueBufferName(target->name);
      if (name != target->name) {
        auto new_target = make_object<BufferNode>(*target.get());
        new_target->name = name;
        AddBufferRemap(target, Buffer(new_target));
        target = Buffer(new_target);
      }
      n->match_buffers.Set(i, MatchBufferRegion(target, BufferRegion(source, region)));
    }
    // update block allocations, avoid same name with global func's buffers
    for (size_t i = 0; i < block->alloc_buffers.size(); ++i) {
      Buffer buffer = block->alloc_buffers[i];
      std::string name = GetUniqueBufferName(buffer->name);
      if (name != buffer->name) {
        auto new_buffer = make_object<BufferNode>(*buffer.get());
        new_buffer->name = name;
        AddBufferRemap(buffer, Buffer(new_buffer));
        n->alloc_buffers.Set(i, Buffer(new_buffer));
      }
    }
    if (root_block_.get() == block) {
      // drop root block of inlined primfunc
      root_block_ = Block(n);
      return StmtExprMutator::VisitStmt(n->body);
    }
    // update reads and writes
    for (size_t i = 0; i < block->reads.size(); ++i) {
      const auto& read_region = block->reads[i];
      auto it = buffer_remap_.find(read_region->buffer.get());
      if (it != buffer_remap_.end()) {
        n->reads.Set(i, BufferRegion(it->second, read_region->region));
      }
    }
    for (size_t i = 0; i < block->writes.size(); ++i) {
      const auto& write_region = block->writes[i];
      auto it = buffer_remap_.find(write_region->buffer.get());
      if (it != buffer_remap_.end()) {
        n->writes.Set(i, BufferRegion(it->second, write_region->region));
      }
    }
    n->body = StmtExprMutator::VisitStmt(n->body);
    return std::move(BlockRealize(iter_values, predicate, std::move(Block(n))));
  }

  Stmt VisitStmt_(const StoreNode* store) final {
    auto it = var_remap_.find(store->buffer_var.get());
    if (it != var_remap_.end()) {
      auto n = CopyOnWrite(store);
      ICHECK(it->second->IsInstance<VarNode>())
          << "Can not inline opaque accessed buffer argument with non-var argument " << it->second;
      n->buffer_var = Downcast<Var>(it->second);
      return std::move(Store(n));
    }
    return StmtExprMutator::VisitStmt_(store);
  }

  PrimExpr VisitExpr_(const LoadNode* load) final {
    auto it = var_remap_.find(load->buffer_var.get());
    if (it != var_remap_.end()) {
      auto n = make_object<LoadNode>(*load);
      ICHECK(it->second->IsInstance<VarNode>())
          << "Can not inline opaque accessed buffer argument with non-var argument " << it->second;
      n->buffer_var = Downcast<Var>(it->second);
      return std::move(Load(n));
    }
    return StmtExprMutator::VisitExpr_(load);
  }

  std::string GetUniqueBufferName(const std::string& name) {
    auto it = buffer_name_cnt_->find(name);
    if (it != buffer_name_cnt_->end()) {
      size_t cnt = it->second;
      std::string new_name = name + "_" + std::to_string(cnt);
      while (buffer_name_cnt_->find(new_name) != buffer_name_cnt_->end()) {
        cnt += 1;
        new_name = name + "_" + std::to_string(cnt);
      }
      it->second = cnt;
      buffer_name_cnt_->insert(it, {new_name, 1});
      return new_name;
    } else {
      buffer_name_cnt_->insert(it, {name, 1});
      return name;
    }
  }

  std::unordered_map<const VarNode*, PrimExpr> var_remap_;
  std::unordered_map<const BufferNode*, Buffer> buffer_remap_;
  std::unordered_map<std::string, size_t>* buffer_name_cnt_;
};

/*! \brief PrimFunc call inline mutator */
class PrimFuncInliner : public StmtExprMutator {
 public:
  PrimFunc Rewrite() {
    Stmt new_body = VisitStmt(origin_body_);
    if (global_new_root_block_.defined()) {
      global_new_root_block_.CopyOnWrite()->body = new_body;
      new_body = BlockRealize({}, make_const(DataType::Bool(), true), global_new_root_block_);
    }
    new_main_func_->body = ConvertSSA(new_body);
    return GetRef<PrimFunc>(new_main_func_);
  }

  PrimFuncInliner(IRModule m, PrimFuncNode* new_main_func, Map<String, PrimFunc> extern_prim_funcs)
      : module_(m), new_main_func_(new_main_func), extern_prim_funcs_(extern_prim_funcs) {
    for (const auto& p : new_main_func->buffer_map) {
      global_buffer_map_.Set(p.second->data, p.second);
      buffer_name_cnt_[p.second->name] = 1;
    }
    origin_body_ = new_main_func_->body;
    if (const BlockRealizeNode* root_realize = new_main_func->body.as<BlockRealizeNode>()) {
      global_new_root_block_ = root_realize->block;
      origin_body_ = global_new_root_block_->body;
    }
  }

 private:
  Stmt VisitStmt_(const BlockNode* block) final {
    for (const Buffer& buffer : block->alloc_buffers) {
      global_buffer_map_.Set(buffer->data, buffer);
      buffer_name_cnt_[buffer->name] = 1;
    }
    return StmtExprMutator::VisitStmt_(block);
  }

  Stmt VisitStmt_(const EvaluateNode* evaluate) final {
    if (const CallNode* call = evaluate->value.as<CallNode>()) {
      // if the call op is the global var, try inline the referenced primfunc into body.
      if (const GlobalVarNode* gv = call->op.as<GlobalVarNode>()) {
        auto basefunc = module_->Lookup(GetRef<GlobalVar>(gv));
        if (const PrimFuncNode* primfunc = basefunc.as<PrimFuncNode>()) {
          return InlinePrimFunc(primfunc, call->args);
        }
      } else if (extern_prim_funcs_.defined() && call->op.same_as(builtin::call_extern())) {
        // if the call op is "call_extern" and funcname is bind by a primfunc,
        // try inline the primfunc into body.
        String extern_funcname = Downcast<StringImm>(call->args[0])->value;
        auto it = extern_prim_funcs_.find(extern_funcname);
        if (it != extern_prim_funcs_.end()) {
          Array<PrimExpr> call_args(call->args.begin() + 1, call->args.end());
          return InlinePrimFunc((*it).second.get(), call_args);
        }
      }
    }
    return StmtExprMutator::VisitStmt_(evaluate);
  }

  Stmt InlinePrimFunc(const PrimFuncNode* primfunc, const Array<PrimExpr>& args) {
    ICHECK_EQ(primfunc->params.size(), args.size());
    PrimFuncBufferSubstituter substituter(&buffer_name_cnt_);

    // associate buffer map and params of inlined primfunc with actual call args
    for (size_t i = 0; i < args.size(); ++i) {
      auto param = primfunc->params[i];
      substituter.AddVarRemap(param, args[i]);
      auto it = primfunc->buffer_map.find(param);
      if (it == primfunc->buffer_map.end()) {
        continue;
      }
      Buffer matched_buffer = (*it).second;
      const VarNode* argvar = args[i].as<VarNode>();
      ICHECK(argvar) << "This inliner only accept PrimFunc call with buffer var argument";
      ICHECK(global_buffer_map_.count(GetRef<Var>(argvar)))
          << "Can not find buffer bind to " << argvar->name_hint;
      Buffer global_buffer = global_buffer_map_[GetRef<Var>(argvar)];

      auto n = make_object<BufferNode>(*global_buffer.get());
      n->data = matched_buffer->data;
      ICHECK(StructuralEqual()(matched_buffer, Buffer(n)))
          << "Buffer mismatch, expect " << i << "th argument buffer of primfunc call to be "
          << matched_buffer << ", but get " << global_buffer << " in global scope";
      substituter.AddBufferRemap(matched_buffer, global_buffer);
    }

    // determine whether the inlined primfunc has a root block, we can try drop it if so.
    if (const BlockRealizeNode* root_realize = primfunc->body.as<BlockRealizeNode>()) {
      if (root_realize->iter_values.empty()) {
        substituter.root_block_ = root_realize->block;
        if (!global_new_root_block_.defined()) {
          // since inlined primfunc has root block, global func also need a root block
          global_new_root_block_ = Block({}, {}, {}, "root", Stmt());
        }
      }
    }

    // update inlined func's root block buffer defs to global
    Stmt inlined = substituter(primfunc->body);
    if (substituter.root_block_.defined()) {
      auto new_root_node = global_new_root_block_.CopyOnWrite();
      for (const MatchBufferRegion& match : substituter.root_block_->match_buffers) {
        new_root_node->match_buffers.push_back(match);
      }
      for (const Buffer& alloc : substituter.root_block_->alloc_buffers) {
        new_root_node->alloc_buffers.push_back(alloc);
      }
    }
    if (primfunc->attrs.defined()) {
      for (const auto& p : primfunc->attrs->dict) {
        inlined = AttrStmt(p.second, p.first, PrimExpr(), inlined);
      }
    }
    return inlined;
  }

  /*! \brief map var to the buffer of main func */
  Map<Var, Buffer> global_buffer_map_;
  /*! \brief rewritten module */
  IRModule module_;
  /*! \brief the new PrimFunc to return after inline */
  PrimFuncNode* new_main_func_;
  /*! \brief map call_extern funcname to PrimFunc */
  Map<String, PrimFunc> extern_prim_funcs_;
  /*! \brief if defined, the new root block of main func */
  Block global_new_root_block_;
  /*! \brief the function body of original main func, without root block */
  Stmt origin_body_;
  /*! \brief helper dict to unique the buffer names */
  std::unordered_map<std::string, size_t> buffer_name_cnt_;
};

namespace transform {

Pass InlinePrimFuncCalls(Map<String, PrimFunc> extern_prim_funcs) {
  auto pass_func = [extern_prim_funcs](PrimFunc func, IRModule m, PassContext ctx) {
    auto* n = func.CopyOnWrite();
    PrimFuncInliner inliner(m, n, extern_prim_funcs);
    IRModule new_mod = IRModule::FromExpr(inliner.Rewrite());

    // use RemoveNoOp to flatten the seq stmts
    return Downcast<PrimFunc>(RemoveNoOp()(new_mod)->Lookup("main"));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.edgex.InlinePrimFuncCalls", {});
}

TVM_REGISTER_GLOBAL("tir.edgex.transform.InlinePrimFuncCalls").set_body_typed(InlinePrimFuncCalls);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
