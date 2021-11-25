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
 * \file storage_rewrite_nnp400.cc
 * \brief Memory access pattern analysis and optimization.
 *  Re-write data access to enable memory sharing when possible.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/ir/type.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target_info.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <map>
#include <unordered_map>
#include <unordered_set>

#include "../../../../runtime/thread_storage_scope.h"
#include "../../../../tir/transforms/ir_utils.h"
#include "../edgex_ir_utils.h"
#include "../op/builtin.h"

namespace tvm {
namespace tir {

using runtime::StorageRank;
using runtime::StorageScope;

class NNP400LinearAccessPatternFinder final : public StmtExprVisitor {
 public:
  /*! \brief record the touch hist of statement. */
  struct StmtEntry {
    /*! \brief The statement. */
    const StmtNode* stmt;
    // The index in the linear_seq_ to point to end of the nested scope.
    // This is only set to non-zero if stmt is a nested scope.
    // if offset > 0, means this is the begin, the end entry is current_index + offset
    // if offset < 0, means this is the end, the begin entry is current_index + offset
    int64_t scope_pair_offset{0};
    // The buffer variables this statement touched.
    std::vector<const VarNode*> touched;
  };
  // The scope of each allocation
  struct AllocEntry {
    // Scope used for allocation.
    StorageScope storage_scope;
    // scope level
    size_t level{0};
    // allocation alignment in bytes
    size_t align_bytes{0};
    // allocation stmt
    const AllocateNode* alloc{nullptr};
  };

  // linearized access sequence.
  std::vector<StmtEntry> linear_seq_;
  // The storage scope of each buffer
  std::unordered_map<const VarNode*, AllocEntry> alloc_info_;

 private:
  void VisitStmt_(const AllocateNode* op) final {
    size_t level = scope_.size();
    AllocEntry& entry = alloc_info_[op->buffer_var.get()];
    entry.storage_scope = GetStorageScope(op->buffer_var);
    entry.alloc = op;
    entry.level = level;
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitBufferAccess(const VarNode* buf) {
    auto it = alloc_info_.find(buf);
    if (it != alloc_info_.end() && it->second.alloc) {
      CHECK_LT(it->second.level, scope_.size()) << "Access memory buffer out of allocation scope.";
      scope_[it->second.level].touched.push_back(buf);
    }
  }
  void VisitStmt_(const StoreNode* op) final {
    scope_.push_back(StmtEntry());
    // visit subexpr
    StmtExprVisitor::VisitStmt_(op);
    // Add write access.
    const VarNode* buf = op->buffer_var.get();
    VisitBufferAccess(buf);
    StmtEntry e = scope_.back();
    scope_.pop_back();
    if (e.touched.size() != 0) {
      e.stmt = op;
      linear_seq_.push_back(e);
    }
  }
  void VisitStmt_(const EvaluateNode* op) final {
    scope_.push_back(StmtEntry());
    // visit subexpr
    StmtExprVisitor::VisitStmt_(op);
    StmtEntry e = scope_.back();
    scope_.pop_back();
    if (!e.touched.empty()) {
      e.stmt = op;
      linear_seq_.push_back(e);
    }
  }
  void VisitExpr_(const LoadNode* op) final {
    // Add read access.
    StmtExprVisitor::VisitExpr_(op);
    const VarNode* buf = op->buffer_var.get();
    VisitBufferAccess(buf);
  }
  void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::address_of())) {
      const LoadNode* l = op->args[0].as<LoadNode>();
      this->VisitExpr(l->index);
    } else if (op->op.same_as(builtin::tvm_access_ptr())) {
      const VarNode* v = op->args[1].as<VarNode>();
      CHECK(v);
      VisitBufferAccess(v);
    } else {
      StmtExprVisitor::VisitExpr_(op);
      // For cube dmas (bdma/idma/wdma), the accessed dm buffer
      // should be alive till odma finished, but this is implicit
      // in linear ir program since the dma are async. Thus we
      // store and revisit them to annotate a proper lifetime.
      // Currently we assume there are 1-1 respondances.
      if (op->op.same_as(edgex::builtin::nnp_bdma_load()) ||
          op->op.same_as(edgex::builtin::nnp_wdma_load()) ||
          op->op.same_as(edgex::builtin::nnp_idma_load())) {
        alive_cube_dmas_.push_back(op);
      } else if (op->op.same_as(edgex::builtin::nnp_odma_store())) {
        for (auto alive_call : alive_cube_dmas_) {
          // arg[1]=dst, arg[2]=src
          VisitExpr(alive_call->args[1]);
          VisitExpr(alive_call->args[2]);
        }
        alive_cube_dmas_.clear();
      }
    }
  }
  void VisitExpr_(const VarNode* buf) final {
    // Directly reference to the variable count as a read.
    auto it = alloc_info_.find(buf);
    if (it != alloc_info_.end() && it->second.alloc) {
      ICHECK_LT(it->second.level, scope_.size()) << " buf=" << buf->name_hint;
      scope_[it->second.level].touched.push_back(buf);
    }
  }
  template <typename T>
  void VisitNewScope(const T* op) {
    scope_.push_back(StmtEntry());
    StmtEntry e;
    e.stmt = op;
    int64_t begin_index = static_cast<int64_t>(linear_seq_.size());
    // before scope.
    linear_seq_.push_back(e);
    StmtExprVisitor::VisitStmt_(op);
    // after scope.
    e.touched = std::move(scope_.back().touched);
    scope_.pop_back();
    int64_t end_index = static_cast<int64_t>(linear_seq_.size());
    ICHECK_GT(end_index, begin_index);
    e.scope_pair_offset = begin_index - end_index;
    linear_seq_.push_back(e);
    // record the pointer to end index.
    ICHECK_NE(end_index, 0U);
    linear_seq_[begin_index].scope_pair_offset = end_index - begin_index;
  }
  void VisitStmt_(const AttrStmtNode* op) final {
    // Only record the outer most thread extent.
    if (op->attr_key == attr::thread_extent && !in_thread_env_) {
      in_thread_env_ = true;
      VisitNewScope(op);
      in_thread_env_ = false;
    } else if (op->attr_key == attr::extern_scope) {
      VisitNewScope(op);
    } else if (op->attr_key == attr::virtual_thread) {
      VisitNewScope(op);
    } else if (op->attr_key == attr::storage_alignment) {
      const VarNode* buf = op->node.as<VarNode>();
      ICHECK(buf);
      alloc_info_[buf].align_bytes = op->value.as<IntImmNode>()->value;
      StmtExprVisitor::VisitStmt_(op);
    } else {
      StmtExprVisitor::VisitStmt_(op);
    }
  }
  void VisitStmt_(const IfThenElseNode* op) final { VisitNewScope(op); }

  void VisitStmt_(const ForNode* op) final { VisitNewScope(op); }

  void VisitStmt_(const WhileNode* op) final { VisitNewScope(op); }

  void VisitStmt_(const AssertStmtNode* op) final { VisitNewScope(op); }

  // Whether already in thread env.
  bool in_thread_env_{false};
  // The scope stack.
  std::vector<StmtEntry> scope_;
  // The current async idma/wdma/bdma intrin with their dm buffer alive still
  std::vector<const CallNode*> alive_cube_dmas_;
};

// Verify if the statement can be run safely via inplace fashion
//
// Detect pattern: dst[index] = f(src[index])
//
// WARNING: the current detection algorithm cannot handle the case
// when a location in an array is written multiple times
//
// For example, the following program will pass the check,
// but we cannot make A and B to be the same array.
//
//  A[0] = B[0] + 1
//  A[0] = B[0] + 1
//
// The high level code generator needs to ensure that the generated
// code only write each location of the target array once.
//
// This is the case with IR generated by the current compute schedule.
// We explicitly return false if we find there is an extern block
// which can be arbitrary IR.
//
// Nevertheless, inplace detector should be used with care in mind.
// We may also consider introduce a condition checker that checks
// if every index only visited once for an absolute sufficient condition.
//
// The code after inplace transformation is no longer idempotent.
//
class InplaceOpVerifier : public StmtExprVisitor {
 public:
  bool Check(const Object* stmt, const VarNode* dst, const VarNode* src) {
    dst_ = dst;
    src_ = src;
    result_ = true;
    if (stmt->IsInstance<AttrStmtNode>()) {
      VisitStmt_(static_cast<const AttrStmtNode*>(stmt));
    } else if (stmt->IsInstance<ForNode>()) {
      VisitStmt_(static_cast<const ForNode*>(stmt));
    } else if (stmt->IsInstance<IfThenElseNode>()) {
      VisitStmt_(static_cast<const IfThenElseNode*>(stmt));
    } else if (stmt->IsInstance<WhileNode>()) {
      VisitStmt_(static_cast<const WhileNode*>(stmt));
    } else if (stmt->IsInstance<StoreNode>()) {
      VisitStmt_(static_cast<const StoreNode*>(stmt));
    } else {
      return false;
    }
    return result_;
  }

  using StmtExprVisitor::VisitStmt_;

  void VisitStmt(const Stmt& n) final {
    if (!result_) return;
    StmtExprVisitor::VisitStmt(n);
  }
  void VisitExpr(const PrimExpr& n) final {
    if (!result_) return;
    StmtExprVisitor::VisitExpr(n);
  }

  void VisitExpr_(const VarNode* op) final {
    // assume all opaque access is unsafe
    if (op == dst_ || op == src_) {
      result_ = false;
      return;
    }
  }

  void VisitStmt_(const StoreNode* op) final {
    ++mem_nest_;
    this->VisitExpr(op->index);
    --mem_nest_;
    if (op->buffer_var.get() == dst_) {
      store_ = op;
      this->VisitExpr(op->value);
      this->VisitExpr(op->predicate);
      store_ = nullptr;
    } else {
      this->VisitExpr(op->value);
      this->VisitExpr(op->predicate);
    }
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    // always reject extern code
    if (op->attr_key == attr::extern_scope || op->attr_key == attr::volatile_scope) {
      result_ = false;
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const LoadNode* op) final {
    const VarNode* buf = op->buffer_var.get();
    // cannot read from dst_ (no reduction)
    if (buf == dst_) {
      result_ = false;
      return;
    }
    // do not allow indirect memory load
    if (mem_nest_ != 0) {
      result_ = false;
      return;
    }
    if (src_ == buf) {
      if (store_ == nullptr || store_->value.dtype() != op->dtype ||
          !tir::ExprDeepEqual()(store_->index, op->index)) {
        result_ = false;
        return;
      }
    }
    ++mem_nest_;
    StmtExprVisitor::VisitExpr_(op);
    --mem_nest_;
  }

 private:
  // result of the check
  bool result_{true};
  // destination memory
  const VarNode* dst_;
  // source variable
  const VarNode* src_;
  // counter of load,
  // it is not safe to inplace when there is nested load like A[B[i]]
  int mem_nest_{0};
  // The current store to be inspected
  const StoreNode* store_{nullptr};
};

// Planner to plan and rewrite memory allocation.
class NNP400StoragePlanRewriter : public StmtExprMutator {
 public:
  using StmtEntry = NNP400LinearAccessPatternFinder::StmtEntry;
  using AllocEntry = NNP400LinearAccessPatternFinder::AllocEntry;

  explicit NNP400StoragePlanRewriter(bool verbose) : verbose_(verbose) {}

  Stmt Rewrite(Stmt stmt, bool detect_inplace) {
    detect_inplace_ = detect_inplace;
    // plan the rewrite
    NNP400LinearAccessPatternFinder finder;
    finder(stmt);
    this->LivenessAnalysis(finder.linear_seq_);
    this->PlanMemory(finder.linear_seq_, finder.alloc_info_);
    this->PrepareNewAlloc();
    // start rewrite
    stmt = operator()(std::move(stmt));
    if (attach_map_.count(nullptr)) {
      return MakeAttach(attach_map_.at(nullptr), stmt);
    }
    return stmt;
  }
  Stmt VisitStmt_(const StoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<StoreNode>();
    ICHECK(op != nullptr);
    auto it = alloc_map_.find(op->buffer_var.get());
    if (it == alloc_map_.end()) return stmt;
    return Store(it->second->alloc_var, op->value,
                 RemapIndex(op->value.dtype(), op->index, it->second), op->predicate);
  }
  PrimExpr VisitExpr_(const LoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<LoadNode>();
    auto it = alloc_map_.find(op->buffer_var.get());
    if (it == alloc_map_.end()) return expr;
    return Load(op->dtype, it->second->alloc_var, RemapIndex(op->dtype, op->index, it->second),
                op->predicate);
  }
  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = alloc_map_.find(op);
    if (it != alloc_map_.end()) {
      if (it->second->bits_offset != 0) {
        LOG(ERROR) << "Use a merged buffer variable address";
      }
      return it->second->alloc_var;
    } else {
      return GetRef<PrimExpr>(op);
    }
  }

  void SetStaticDMAAddr(const std::string& begin_key, const std::string& end_key, CallNode* call,
                        PrimExpr access) {
    const CallNode* access_call = access.as<CallNode>();
    ICHECK(access_call->op.same_as(builtin::tvm_access_ptr()));
    arith::Analyzer analyzer;
    const int64_t* begin = as_const_int(analyzer.Simplify(access_call->args[2]));
    const int64_t* extent = as_const_int(analyzer.Simplify(access_call->args[3]));
    if (begin == nullptr || extent == nullptr) {
      return;  // skip dynamic buffer offset
    }
    DataType dtype = access_call->args[0].dtype();
    int elem_bytes = dtype.bits() * dtype.lanes() / 8;
    CHECK_EQ(elem_bytes * 8, dtype.bits() * dtype.lanes());
    int64_t bytes_begin = (*begin) * elem_bytes;
    int64_t bytes_end = bytes_begin + (*extent) * elem_bytes - 1;
    // the dma addr should LE 0x3fffff.
    CHECK_LE(bytes_end, 0x3fffff);
    auto to_hex = [](size_t i) -> std::string {
      std::stringstream ss;
      ss << "0x" << std::hex << i;
      return ss.str();
    };
    edgex::NNPAddArg(call, begin_key, to_hex(bytes_begin));
    edgex::NNPAddArg(call, end_key, to_hex(bytes_end));
    return;
  }

  PrimExpr VisitNNPIntrinsic(const CallNode* call) {
    auto op = call->op;
    auto n = make_object<CallNode>(*call);
    if (op.same_as(edgex::builtin::nnp_eidma_load())) {
      PrimExpr dst_access = call->args[1];
      SetStaticDMAAddr("ei_start_addr1", "ei_end_addr1", n.get(), dst_access);
      SetStaticDMAAddr("ei_start_addr2", "ei_end_addr2", n.get(), dst_access);
    } else if (op.same_as(edgex::builtin::nnp_ewdma_load())) {
      PrimExpr dst_access = call->args[1];
      SetStaticDMAAddr("ew_start_addr1", "ew_end_addr1", n.get(), dst_access);
      SetStaticDMAAddr("ew_start_addr2", "ew_end_addr2", n.get(), dst_access);
    } else if (op.same_as(edgex::builtin::nnp_eodma_store())) {
      PrimExpr src_access = call->args[2];
      SetStaticDMAAddr("eo_start_addr1", "eo_end_addr1", n.get(), src_access);
      SetStaticDMAAddr("eo_start_addr2", "eo_end_addr2", n.get(), src_access);
    } else if (op.same_as(edgex::builtin::nnp_bdma_load())) {
      PrimExpr src_access = call->args[2];
      SetStaticDMAAddr("st_addr1_bdma", "end_addr1_bdma", n.get(), src_access);
      SetStaticDMAAddr("st_addr2_bdma", "end_addr2_bdma", n.get(), src_access);
    } else if (op.same_as(edgex::builtin::nnp_idma_load())) {
      PrimExpr src_access = call->args[2];
      SetStaticDMAAddr("feat_st_addr1_idma", "feat_end_addr1_idma", n.get(), src_access);
      SetStaticDMAAddr("feat_st_addr2_idma", "feat_end_addr2_idma", n.get(), src_access);
    } else if (op.same_as(edgex::builtin::nnp_odma_store())) {
      PrimExpr dst_access = call->args[1];
      SetStaticDMAAddr("rslt_st_addr1_odma", "rslt_end_addr1_odma", n.get(), dst_access);
      SetStaticDMAAddr("rslt_st_addr2_odma", "rslt_end_addr2_odma", n.get(), dst_access);
    } else if (op.same_as(edgex::builtin::nnp_wdma_load())) {
      PrimExpr src_access = call->args[2];
      SetStaticDMAAddr("wt_st_addr1_wdma", "wt_end_addr1_wdma", n.get(), src_access);
      SetStaticDMAAddr("wt_st_addr2_wdma", "wt_end_addr2_wdma", n.get(), src_access);
    } else if (op.same_as(edgex::builtin::nnp_vidma_load())) {
      PrimExpr src_access = call->args[2];
      PrimExpr dst_access = call->args[1];
      SetStaticDMAAddr("start_addr1_dm_vidma", "end_addr1_dm_vidma", n.get(), src_access);
      SetStaticDMAAddr("start_addr2_dm_vidma", "end_addr2_dm_vidma", n.get(), src_access);
      SetStaticDMAAddr("cb_buf_start_addr_vm_vidma", "cb_buf_end_addr_vm_vidma", n.get(),
                       dst_access);
    } else if (op.same_as(edgex::builtin::nnp_vodma_store())) {
      PrimExpr src_access = call->args[2];
      PrimExpr dst_access = call->args[1];
      SetStaticDMAAddr("start_addr1_dm_vodma", "end_addr1_dm_vodma", n.get(), dst_access);
      SetStaticDMAAddr("start_addr2_dm_vodma", "end_addr2_dm_vodma", n.get(), dst_access);
      SetStaticDMAAddr("cb_buf_start_addr_vm_vodma", "cb_buf_end_addr_vm_vodma", n.get(),
                       src_access);
    }
    return std::move(Call(n));
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      CHECK_EQ(op->args.size(), 5U);
      DataType dtype = op->args[0].dtype();
      const VarNode* buffer = op->args[1].as<VarNode>();
      auto it = alloc_map_.find(buffer);
      if (it == alloc_map_.end()) {
        return StmtExprMutator::VisitExpr_(op);
      }
      const StorageEntry* se = it->second;
      PrimExpr offset = this->VisitExpr(op->args[2]);
      PrimExpr extent = this->VisitExpr(op->args[3]);
      uint64_t elem_bits = dtype.bits() * dtype.lanes();
      ICHECK_EQ(se->bits_offset % elem_bits, 0U);
      if (se->bits_offset != 0) {
        offset = make_const(offset.dtype(), se->bits_offset / elem_bits) + offset;
      }
      return Call(op->dtype, op->op, {op->args[0], se->alloc_var, offset, extent, op->args[4]});
    } else if (edgex::IsNNPIntrinsic(op->op)) {
      auto updated = StmtExprMutator::VisitExpr_(op);
      return VisitNNPIntrinsic(updated.as<CallNode>());
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent || op->attr_key == attr::virtual_thread ||
        attr::IsPragmaKey(op->attr_key)) {
      // remake all the allocation at the attach scope.
      if (attach_map_.count(op)) {
        auto& s_vec = attach_map_[op];
        Stmt stmt = StmtExprMutator::VisitStmt_(op);
        op = stmt.as<AttrStmtNode>();
        return AttrStmt(op->node, op->attr_key, op->value, MakeAttach(s_vec, op->body));
      } else {
        return StmtExprMutator::VisitStmt_(op);
      }
    } else if (op->attr_key == attr::volatile_scope) {
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      op = stmt.as<AttrStmtNode>();
      auto it = alloc_map_.find(op->node.as<VarNode>());
      if (it == alloc_map_.end()) return stmt;
      return AttrStmt(it->second->alloc_var, op->attr_key, op->value, op->body);
    } else if (op->attr_key == attr::storage_alignment) {
      return this->VisitStmt(op->body);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const ForNode* op) final {
    ICHECK(op->kind != ForKind::kVectorized) << "VectorizeLoop before LiftStorageAlloc";
    // remake all the allocation at the attach scope.
    if (attach_map_.count(op)) {
      auto& s_vec = attach_map_[op];
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      op = stmt.as<ForNode>();
      return For(op->loop_var, op->min, op->extent, op->kind, MakeAttach(s_vec, op->body),
                 op->thread_binding, op->annotations);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const AllocateNode* op) final { return this->VisitStmt(op->body); }

 private:
  struct StorageEntry {
    // The scope that this alloc attaches after
    // For shared/local memory it is beginning of the thread extent.
    // for global memory it is nullptr, means beginning of everything.
    const Object* attach_scope_{nullptr};
    // The constant size of the buffer in bits, only used if it is constant
    uint64_t const_nbits{0};
    // The alignment requirement bits
    uint64_t align_bytes{0};
    // The storage scope.
    StorageScope scope;
    // Allocs that shares this entry.
    std::vector<const AllocateNode*> allocs;
    // The children of this entry, not including itself.
    std::vector<StorageEntry*> merged_children;
    // The replacement allocation, if any.
    Stmt new_alloc;
    // The var expr of new allocation.
    Var alloc_var;
    // The allocation element type.
    DataType elem_type;
    // This is non-zero if this allocate is folded into another one
    // the address(in bits) becomes alloc_var + bits_offset;
    // can be effectively converted to the element type.
    // We need to convert bit_offset to offset of specific element type later.
    //
    // We use bits(instead of bytes) to support non-conventional indexing in hardware.
    // When we are merging buffer together, the bits_offset are set to be aligned
    // to certain value given by the max_simd_bits property of the special memory.
    //
    // This allows effective sharing among different types as long as their alignment
    // requirement fits into the max_simd_bits.
    uint64_t bits_offset{0};
  };

  // Alllocate entry of node.
  // Event entry in liveness analysis
  struct EventEntry {
    // variables we generate
    std::vector<const VarNode*> gen;
    // variables we kill
    std::vector<const VarNode*> kill;
  };

  bool IsNNPScope(const StorageScope& scope) const {
    return scope.rank == StorageRank::kVM || scope.rank == StorageRank::kDM;
  }

  Stmt MakeAttach(const std::vector<StorageEntry*>& s_vec, Stmt body) {
    std::vector<Stmt> nest;
    for (StorageEntry* e : s_vec) {
      if (e->new_alloc.defined()) {
        if (e->align_bytes > 0 && !IsNNPScope(e->scope)) {
          nest.emplace_back(AttrStmt(e->alloc_var, attr::storage_alignment,
                                     make_const(DataType::Int(32), e->align_bytes), Evaluate(0)));
        }
        nest.push_back(e->new_alloc);
      }
    }
    return MergeNest(nest, body);
  }
  // Remap the index
  PrimExpr RemapIndex(DataType dtype, PrimExpr index, StorageEntry* e) {
    if (e->bits_offset == 0) return index;
    uint64_t elem_bits = dtype.bits();
    ICHECK_EQ(e->bits_offset % elem_bits, 0U);
    return make_const(index.dtype(), e->bits_offset / elem_bits) + index;
  }
  // Prepare the new allocations
  void PrepareNewAlloc() {
    for (size_t i = 0; i < alloc_vec_.size(); ++i) {
      StorageEntry* e = alloc_vec_[i].get();
      attach_map_[e->attach_scope_].push_back(e);
    }
    // find allocation via attach map.
    for (auto& kv : attach_map_) {
      // find the element with the most amount of bytes.
      std::vector<StorageEntry*>& vec = kv.second;
      // try to find merge, for tagged memory
      for (size_t i = 0; i < vec.size(); ++i) {
        StorageEntry* e = vec[i];
        bool do_merge = false;
        if (e->scope.tag.length() != 0) {
          ICHECK_NE(e->const_nbits, 0U) << "Special tagged memory must be const size";
          do_merge = true;
        } else if (IsNNPScope(e->scope)) {
          do_merge = true;
        }
        if (do_merge) {
          for (size_t j = 0; j < i; ++j) {
            if (e->scope == vec[j]->scope) {
              vec[j]->merged_children.push_back(e);
              break;
            }
          }
        }
      }
      // Start allocation
      for (size_t i = 0; i < vec.size(); ++i) {
        StorageEntry* e = vec[i];
        // already merged
        if (e->bits_offset != 0) continue;
        if (e->merged_children.size() != 0) {
          NewAllocTagMerged(e);
          continue;
        }
        // Get the allocation size;
        e->alloc_var = e->allocs[0]->buffer_var;
        DataType alloc_type = e->allocs[0]->dtype;
        for (const AllocateNode* op : e->allocs) {
          if (op->dtype.lanes() > alloc_type.lanes()) {
            alloc_type = op->dtype;
          }
        }

        if (e->allocs.size() == 1) {
          // simply use the original allocation.
          PrimExpr sz = foldl([](PrimExpr a, PrimExpr b, Span span) { return mul(a, b, span); },
                              make_const(DataType::Int(32), 1), e->allocs[0]->extents);
          sz = Align(sz);
          e->new_alloc =
              Allocate(e->alloc_var, alloc_type, {sz}, e->allocs[0]->condition, Evaluate(0));
          if (e->scope.tag.length() != 0) {
            MemoryInfo info = GetMemoryInfo(e->scope.to_string());
            uint64_t total_elem = e->const_nbits / e->elem_type.bits();
            ICHECK_LE(total_elem * e->elem_type.bits(), info->max_num_bits)
                << "Allocation exceed bound of memory tag " << e->scope.to_string();
          }
        } else {
          // Build a merged allocation
          PrimExpr combo_size;
          for (const AllocateNode* op : e->allocs) {
            PrimExpr sz = foldl([](PrimExpr a, PrimExpr b, Span span) { return mul(a, b, span); },
                                make_const(DataType::Int(32), 1), op->extents);
            auto nbits = op->dtype.bits() * op->dtype.lanes();
            if (const auto* imm = sz.as<IntImmNode>()) {
              if (imm->value > std::numeric_limits<int>::max() / nbits) {
                LOG(WARNING) << "The allocation requires : " << imm->value << " * " << nbits
                             << " bits, which is greater than the maximum of"
                                " int32. The size is cast to int64."
                             << "\n";
                sz = make_const(DataType::Int(64), imm->value);
              }
            }
            // transform to bits
            auto sz_nbits = sz * nbits;
            if (combo_size.defined()) {
              combo_size = max(combo_size, sz_nbits);
            } else {
              combo_size = sz_nbits;
            }
          }
          // transform to alloc bytes
          auto type_bits = alloc_type.bits() * alloc_type.lanes();
          bool divided = analyzer_.CanProve(indexmod(combo_size, type_bits) == 0);
          combo_size = indexdiv(combo_size, type_bits);
          // round up for can not divided
          if (!divided) {
            combo_size = combo_size + make_const(DataType::Int(32), 1);
          }
          combo_size = Align(combo_size);
          combo_size = analyzer_.Simplify(combo_size);
          e->new_alloc =
              Allocate(e->alloc_var, alloc_type, {combo_size}, const_true(), Evaluate(0));
          if (e->scope.tag.length() != 0) {
            MemoryInfo info = GetMemoryInfo(e->scope.to_string());
            uint64_t total_elem = e->const_nbits / e->elem_type.bits();
            ICHECK_LE(total_elem * e->elem_type.bits(), info->max_num_bits)
                << "Allocation exceed bound of memory tag " << e->scope.to_string();
          }
        }
      }
    }
  }
  // New allocation for merged data
  void NewAllocTagMerged(StorageEntry* e) {
    CHECK(e->scope.tag.length() != 0 || IsNNPScope(e->scope));
    // allocate with element type.
    ICHECK_NE(e->const_nbits, 0U);
    MemoryInfo info = GetMemoryInfo(e->scope.to_string());
    uint64_t total_bits = e->const_nbits;

    // By default, align to 32 bits at each buffer end.
    size_t tail_align = 32;
    if (!IsNNPScope(e->scope) && info.defined()) {
      // Always align to max_simd_bits
      // so we can remap types by keeping this property
      tail_align = info->max_simd_bits;
    }
    if (total_bits % tail_align != 0) {
      total_bits += tail_align - (total_bits % tail_align);
    }
    e->alloc_var = e->allocs[0]->buffer_var;
    for (StorageEntry* child : e->merged_children) {
      ICHECK_NE(child->const_nbits, 0U);
      ICHECK_NE(total_bits, 0U);
      size_t align_bits = child->align_bytes * 8;
      if (align_bits > 0 && total_bits % align_bits != 0) {
        total_bits += align_bits - (total_bits % align_bits);
      }
      child->bits_offset = total_bits;
      child->alloc_var = e->alloc_var;
      total_bits += child->const_nbits;
      if (total_bits % tail_align != 0) {
        total_bits += tail_align - (total_bits % tail_align);
      }
    }
    uint64_t type_bits = e->elem_type.bits() * e->elem_type.lanes();
    uint64_t size = Align((total_bits + type_bits - 1) / type_bits);
    PrimExpr alloc_size = make_const(e->allocs[0]->extents[0].dtype(), size);
    e->new_alloc = Allocate(e->alloc_var, e->elem_type, {alloc_size}, const_true(), Evaluate(0));
    if (info.defined()) {
      ICHECK_LE(total_bits, info->max_num_bits)
          << "Allocation exceed bound of memory tag " << e->scope.to_string();
    }
  }
  // Liveness analysis to find gen and kill point of each variable.
  void LivenessAnalysis(const std::vector<StmtEntry>& seq) {
    // find kill point, do a reverse linear scan.
    std::unordered_set<const VarNode*> touched;
    for (size_t i = seq.size(); i != 0; --i) {
      const StmtEntry& s = seq[i - 1];
      for (const VarNode* buffer : s.touched) {
        if (!touched.count(buffer)) {
          touched.insert(buffer);
          event_map_[s.stmt].kill.push_back(buffer);
        }
      }
    }
    // find gen point, do forward scan
    touched.clear();
    for (size_t i = 0; i < seq.size(); ++i) {
      int64_t offset = seq[i].scope_pair_offset;
      if (offset < 0) continue;
      const StmtEntry& s = seq[i + offset];
      for (const VarNode* buffer : s.touched) {
        if (!touched.count(buffer)) {
          touched.insert(buffer);
          event_map_[s.stmt].gen.push_back(buffer);
        }
      }
    }
    if (verbose_) {
      std::stringstream ss;
      ss << "Liveness analysis result:\n";
      for (size_t i = 0; i < seq.size(); ++i) {
        int64_t offset = seq[i].scope_pair_offset;
        if (offset < 0) continue;
        const StmtEntry& s = seq[i + offset];
        const auto& events = event_map_[s.stmt];
        ss << s.stmt->GetTypeKey() << "\n"
           << "[TOUCH] ";
        for (auto v : s.touched) ss << GetRef<Var>(v) << ", ";
        ss << "\n[GEN] ";
        for (auto v : events.gen) ss << GetRef<Var>(v) << ", ";
        ss << "\n[KILL] ";
        for (auto v : events.kill) ss << GetRef<Var>(v) << ", ";
      }
      LOG(INFO) << ss.str();
    }
  }
  void PlanNewScope(const Object* op) {
    if (thread_scope_ != nullptr) {
      ICHECK(thread_scope_ == op);
      // erase all memory attached to this scope.
      for (auto it = const_free_map_.begin(); it != const_free_map_.end();) {
        if (it->second->attach_scope_ == op) {
          it = const_free_map_.erase(it);
        } else {
          ++it;
        }
      }
      for (auto it = sym_free_list_.begin(); it != sym_free_list_.end();) {
        if ((*it)->attach_scope_ == op) {
          it = sym_free_list_.erase(it);
        } else {
          ++it;
        }
      }
      thread_scope_ = nullptr;
    } else {
      thread_scope_ = op;
    }
  }

  // Memory plan algorithm
  void PlanMemory(const std::vector<StmtEntry>& seq,
                  const std::unordered_map<const VarNode*, AllocEntry>& alloc_info) {
    std::unordered_set<const VarNode*> inplace_flag;

    for (size_t i = 0; i < seq.size(); ++i) {
      const StmtEntry& s = seq[i];
      auto it = event_map_.find(seq[i].stmt);
      // scope_pair_offset >= 0 means it is either
      // - leaf stmt(offset = 0)
      // - beginning of scope(offset < 0)
      // In both cases, we need to handle the gen event correctly
      if (it != event_map_.end() && seq[i].scope_pair_offset >= 0) {
        // Inplace operation detection
        // specially handle this
        bool detect_inplace = detect_inplace_ && (it->second.gen.size() <= 2);

        for (const VarNode* var : it->second.gen) {
          ICHECK(alloc_info.count(var));
          const AllocEntry& ae = alloc_info.at(var);
          StorageEntry* dst_entry = nullptr;
          // inplace detection
          if (detect_inplace) {
            // only one inplace var for s.stmt
            bool inplace_found = false;
            for (const VarNode* src : it->second.kill) {
              if (!inplace_flag.count(src) && alloc_map_.count(src)) {
                InplaceOpVerifier visitor;
                StorageEntry* src_entry = alloc_map_.at(src);
                if (src_entry->scope == ae.storage_scope &&
                    src_entry->attach_scope_ == thread_scope_ &&
                    src_entry->elem_type == ae.alloc->dtype.element_of() &&
                    visitor.Check(s.stmt, var, src)) {
                  uint64_t const_nbits =
                      static_cast<uint64_t>(ae.alloc->constant_allocation_size()) *
                      ae.alloc->dtype.bits() * ae.alloc->dtype.lanes();
                  if (src_entry->const_nbits == const_nbits &&
                      src_entry->align_bytes == ae.align_bytes && !inplace_found) {
                    // successfully inplace
                    dst_entry = src_entry;
                    inplace_flag.insert(src);
                    inplace_found = true;
                  }
                }
              }
            }
          }
          if (dst_entry == nullptr) {
            dst_entry = FindAlloc(ae, thread_scope_);
          }
          dst_entry->allocs.emplace_back(ae.alloc);
          alloc_map_[var] = dst_entry;
        }
      }
      // enter/exit new scope
      if (s.stmt->IsInstance<AttrStmtNode>()) {
        const auto* op = static_cast<const AttrStmtNode*>(s.stmt);
        if (op->attr_key == attr::thread_extent || op->attr_key == attr::virtual_thread ||
            attr::IsPragmaKey(op->attr_key)) {
          PlanNewScope(op);
        } else {
          ICHECK(op->attr_key == attr::extern_scope);
        }
      } else if (s.stmt->IsInstance<ForNode>()) {
        const auto* op = static_cast<const ForNode*>(s.stmt);
        if (op->kind == ForKind::kParallel) {
          if (thread_scope_ == nullptr || thread_scope_ == op) {
            PlanNewScope(op);
          }
        }
      }
      // scope_pair_offset <= 0 means it is either
      // - leaf stmt(offset = 0)
      // - end of scope(offset < 0)
      // In both cases, we need to handle the kill event correctly
      if (it != event_map_.end() && seq[i].scope_pair_offset <= 0) {
        for (const VarNode* var : it->second.kill) {
          // skip space which are already replaced by inplace
          if (!inplace_flag.count(var)) {
            this->Free(var);
          }
        }
      }
    }
  }
  // Allocate new storage entry.
  StorageEntry* NewAlloc(const AllocateNode* op, const Object* attach_scope,
                         const StorageScope& scope, size_t const_nbits, size_t align_bytes) {
    ICHECK(op != nullptr);
    // Re-use not successful, allocate a new buffer.
    std::unique_ptr<StorageEntry> entry(new StorageEntry());
    entry->attach_scope_ = attach_scope;
    entry->scope = scope;
    entry->elem_type = op->dtype.element_of();
    entry->const_nbits = const_nbits;
    entry->align_bytes = align_bytes;
    StorageEntry* e = entry.get();
    alloc_vec_.emplace_back(std::move(entry));
    return e;
  }

  StorageEntry* FindAlloc(const AllocEntry& alloc_entry, const Object* attach_scope) {
    const StorageScope& scope = alloc_entry.storage_scope;
    const AllocateNode* op = alloc_entry.alloc;
    ICHECK(op != nullptr);
    // skip plan for local variable,
    // compiler can do a better job with register allocation.
    const uint64_t match_range = 16;
    uint64_t op_elem_bits = op->dtype.bits() * op->dtype.lanes();
    uint64_t const_nbits = static_cast<uint64_t>(op->constant_allocation_size() * op_elem_bits);
    uint64_t align_bytes = alloc_entry.align_bytes;
    // disable reuse of small arrays, they will be lowered to registers in LLVM
    // This rules only apply if we are using non special memory
    if (scope.tag.length() == 0 && !IsNNPScope(scope)) {
      if (scope.rank >= StorageRank::kWarp || op->dtype.is_handle()) {
        return NewAlloc(op, attach_scope, scope, const_nbits, align_bytes);
      }
      if (const_nbits > 0 && const_nbits <= 32) {
        return NewAlloc(op, attach_scope, scope, const_nbits, align_bytes);
      }
    }
    if (const_nbits != 0) {
      // constant allocation.
      auto begin = const_free_map_.lower_bound(const_nbits / match_range);
      auto mid = const_free_map_.lower_bound(const_nbits);
      auto end = const_free_map_.upper_bound(const_nbits * match_range);
      // start looking at the buffer that is bigger than the required size first
      for (auto it = mid; it != end; ++it) {
        StorageEntry* e = it->second;
        if (e->attach_scope_ != attach_scope) continue;
        if (e->scope != scope) continue;
        // when not divided, no reuse, eg, float4 vs float3
        if (e->bits_offset % op_elem_bits != 0) continue;
        if (align_bytes != 0 && e->align_bytes % align_bytes != 0 &&
            align_bytes % e->align_bytes != 0)
          continue;
        e->const_nbits = std::max(const_nbits, e->const_nbits);
        e->align_bytes = std::max(align_bytes, e->align_bytes);
        const_free_map_.erase(it);
        return e;
      }
      // then start looking at smaller buffers.
      for (auto it = mid; it != begin;) {
        --it;
        StorageEntry* e = it->second;
        if (e->attach_scope_ != attach_scope) continue;
        if (e->scope != scope) continue;
        if (e->elem_type != op->dtype.element_of()) continue;
        if (align_bytes != 0 && e->align_bytes % align_bytes != 0 &&
            align_bytes % e->align_bytes != 0)
          continue;
        e->const_nbits = std::max(const_nbits, e->const_nbits);
        e->align_bytes = std::max(align_bytes, e->align_bytes);
        const_free_map_.erase(it);
        return e;
      }
    } else {
      // Simple strategy: round robin.
      for (auto it = sym_free_list_.begin(); it != sym_free_list_.end(); ++it) {
        StorageEntry* e = *it;
        if (e->attach_scope_ != attach_scope) continue;
        if (e->scope != scope) continue;
        if (e->elem_type != op->dtype.element_of()) continue;
        if (align_bytes != 0 && e->align_bytes % align_bytes != 0 &&
            align_bytes % e->align_bytes != 0)
          continue;
        sym_free_list_.erase(it);
        return e;
      }
    }
    return NewAlloc(op, attach_scope, scope, const_nbits, align_bytes);
  }
  // simulated free.
  void Free(const VarNode* var) {
    auto it = alloc_map_.find(var);
    ICHECK(it != alloc_map_.end());
    StorageEntry* e = it->second;
    ICHECK_NE(e->allocs.size(), 0U);

    // disable reuse of small arrays, they will be lowered to registers in LLVM
    // This rules only apply if we are using non special memory
    if (e->scope.tag.length() == 0 && !IsNNPScope(e->scope)) {
      // Disable sharing of local memory.
      if (e->scope.rank >= StorageRank::kWarp || e->allocs[0]->dtype.is_handle()) return;
      // disable reuse of small arrays
      if (e->const_nbits > 0 && e->const_nbits <= 32) return;
    }
    // normal free.
    if (e->const_nbits != 0) {
      const_free_map_.insert({e->const_nbits, e});
    } else {
      sym_free_list_.push_back(e);
    }
  }
  // thread scope.
  const Object* thread_scope_{nullptr};
  // whether enable inplace detection.
  bool detect_inplace_{false};
  // Locations of free ops.
  std::unordered_map<const Object*, EventEntry> event_map_;
  // constant size free map.
  std::multimap<uint64_t, StorageEntry*> const_free_map_;
  // symbolic free list, for non constant items.
  std::list<StorageEntry*> sym_free_list_;
  // The allocation attach map
  std::unordered_map<const Object*, std::vector<StorageEntry*> > attach_map_;
  // The allocation assign map
  std::unordered_map<const VarNode*, StorageEntry*> alloc_map_;
  // The allocations
  std::vector<std::unique_ptr<StorageEntry> > alloc_vec_;
  // analyzer
  arith::Analyzer analyzer_;
  // verbosity
  bool verbose_{false};
};

namespace transform {

static constexpr const char* VERBOSE_OPTION = "tir.edgex.StorageRewriteNNP400.verbose";
TVM_REGISTER_PASS_CONFIG_OPTION(VERBOSE_OPTION, Bool);

Pass StorageRewriteNNP400() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    bool verbose = ctx->GetConfig<Bool>(VERBOSE_OPTION, Bool(false)).value();
    auto* n = f.CopyOnWrite();
    n->body = NNP400StoragePlanRewriter(verbose).Rewrite(std::move(n->body), true);
    return std::move(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.edgex.StorageRewriteNNP400", {});
}

TVM_REGISTER_GLOBAL("tir.edgex.transform.StorageRewriteNNP400")
    .set_body_typed(StorageRewriteNNP400);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
