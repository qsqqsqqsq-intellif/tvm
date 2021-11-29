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
#include "../../../../arith/ir_mutator_with_analyzer.h"
#include "../../../../tir/schedule/analysis.h"
#include "../../../../tir/schedule/utils.h"
#include "./edgex_primitives.h"
#include "./schedule_utils.h"

namespace tvm {
namespace tir {

using NewBufferRVInfo = std::pair<BufferAxisRVContainer, Array<BufferAxisRV>>;

/*! \brief get represent variables for buffer dimensions */
NewBufferRVInfo CreateBufferAxisRVs(Buffer buffer, ObjectRef alloc_ref) {
  ICHECK(buffer.defined());
  auto node = make_object<BufferAxisRVContainerNode>();
  node->buffer = buffer;
  node->alloc_ref = std::move(alloc_ref);
  node->axes.resize(buffer->shape.size());
  BufferAxisRVContainer container(node);
  Array<BufferAxisRV> results;
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    results.push_back(BufferAxisRV(container, buffer->shape[i]));
    node->axes[i] = results.back().get();
  }
  return {container, results};
}

TVM_REGISTER_NODE_TYPE(BufferAxisRVNode);
TVM_REGISTER_NODE_TYPE(BufferAxisRVContainerNode);

namespace schedule {

using tvm::arith::IRMutatorWithAnalyzer;
using RemapF = std::function<Array<PrimExpr>(const Array<PrimExpr>&, arith::Analyzer*)>;

using LoadRewriteF = std::function<PrimExpr(const BufferLoad&)>;
using StoreRewriteF = std::function<Stmt(const BufferStore&)>;
using RegionRewriteF = std::function<Array<Range>(const Array<Range>&)>;

/*! \brief Record buffer update info */
struct BufferUpdateSpec {
  ObjectRef alloc_ref;
  const BufferNode* origin_buffer;
  Buffer new_buffer;
  RemapF remap_func{nullptr};
  LoadRewriteF load_rewrite_func{nullptr};
  StoreRewriteF store_rewrite_func{nullptr};
  RegionRewriteF region_rewrite_func{nullptr};
  const BlockNode* new_alloc_block{nullptr};
  bool delete_param{false};
};

/*! \brief Update buffer accesses after bufer layout change */
class BufferLayoutUpdater : public IRMutatorWithAnalyzer {
 public:
  BufferLayoutUpdater(arith::Analyzer* analyzer, bool rewrite_block_binding)
      : IRMutatorWithAnalyzer(analyzer), rewrite_block_binding_(rewrite_block_binding) {}

  void AddUpdateSpec(ScheduleState self, const BufferUpdateSpec& spec) {
    ICHECK(spec.alloc_ref.defined());
    ICHECK(spec.origin_buffer);
    ICHECK(spec.new_buffer.defined());
    buffer_map_[spec.origin_buffer->data.get()] = spec;

    // record function mutation info
    if (spec.alloc_ref->IsInstance<GlobalVarNode>()) {
      GlobalVar global_var = Downcast<GlobalVar>(spec.alloc_ref);
      PrimFunc func = Downcast<PrimFunc>(self->mod->Lookup(global_var));
      BlockRealize root_realize = Downcast<BlockRealize>(func->body);
      StmtSRef root_sref = self->stmt2ref.at(root_realize->block.get());
      touched_srefs_[global_var.get()].insert(root_sref);
    } else {
      StmtSRef stmt_sref = Downcast<StmtSRef>(spec.alloc_ref);
      const StmtNode* root_block = GetSRefTreeRoot(stmt_sref)->stmt;
      GlobalVar gv;
      const PrimFuncNode* func = GetRootPrimFunc(self->mod, root_block, &gv);
      ICHECK(func != nullptr && gv.defined());
      touched_srefs_[gv.get()].insert(stmt_sref);

      // record new allocs
      const BlockNode* block = stmt_sref->StmtAs<BlockNode>();
      ICHECK(block);
      if (spec.new_alloc_block && block != spec.new_alloc_block) {
        new_allocs_[spec.new_alloc_block].insert(spec.new_buffer.get());
      }
    }
  }

  void Update(ScheduleState self) {
    for (const auto& p : touched_srefs_) {
      GlobalVar global_var = GetRef<GlobalVar>(p.first);
      PrimFunc func = Downcast<PrimFunc>(self->mod->Lookup(global_var));
      BlockRealize root_realize = Downcast<BlockRealize>(func->body);
      StmtSRef root_sref = self->stmt2ref.at(root_realize->block.get());
      std::vector<StmtSRef> touched_srefs(p.second.begin(), p.second.end());

      // update function header
      ObjectPtr<PrimFuncNode> new_func = make_object<PrimFuncNode>(*func.get());
      new_func->params = Array<Var>();
      new_func->buffer_map = Map<Var, Buffer>();
      for (const Var& param : func->params) {
        auto it = func->buffer_map.find(param);
        if (it == func->buffer_map.end()) {
          new_func->params.push_back(param);
        } else {
          const VarNode* origin_buffer_var = (*it).second->data.get();
          auto spec_it = buffer_map_.find(origin_buffer_var);
          if (spec_it == buffer_map_.end()) {
            new_func->params.push_back(param);
            new_func->buffer_map.Set(param, (*it).second);
          } else {
            const BufferUpdateSpec& spec = spec_it->second;
            if (!spec.delete_param) {
              new_func->params.push_back(param);
              new_func->buffer_map.Set(param, spec.new_buffer);
            }
          }
        }
      }
      self->mod->Update(global_var, PrimFunc(new_func));

      // update function body
      if (touched_srefs.size() >= 2) {
        root_sref = LowestCommonAncestor(touched_srefs, root_sref);
      } else {
        root_sref = touched_srefs[0];
      }

      Stmt updated = this->VisitStmt(std::move(GetRef<Stmt>(root_sref->stmt)));
      self->Replace(root_sref, updated, reuse_dict_);
    }
  }

  /*! \brief Provide block sref reuse information to schedule routines */
  const Map<Block, Block>& reuse_dict() const { return reuse_dict_; }

 private:
  PrimExpr VisitExpr_(const VarNode* var) final {
    auto it = buffer_map_.find(var);
    CHECK(it == buffer_map_.end()) << "Opaque access to buffer " << GetRef<Var>(var)
                                   << " currently is not supported for buffer layout schedule";
    return IRMutatorWithAnalyzer::VisitExpr_(var);
  }

  PrimExpr VisitExpr_(const LoadNode* load) final {
    auto it = buffer_map_.find(load->buffer_var.get());
    CHECK(it == buffer_map_.end()) << "Lower load to buffer " << load->buffer_var
                                   << " currently is not supported for buffer layout schedule";
    return IRMutatorWithAnalyzer::VisitExpr_(load);
  }

  Stmt VisitStmt_(const StoreNode* store) final {
    auto it = buffer_map_.find(store->buffer_var.get());
    CHECK(it == buffer_map_.end()) << "Lower store to buffer " << store->buffer_var
                                   << " currently is not supported for buffer layout schedule";
    return IRMutatorWithAnalyzer::VisitStmt_(store);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto it = buffer_map_.find(op->buffer->data.get());
    if (it != buffer_map_.end()) {
      auto n = make_object<BufferLoadNode>(*op);
      const BufferUpdateSpec& spec = it->second;
      n->buffer = spec.new_buffer;
      if (spec.load_rewrite_func) {
        return spec.load_rewrite_func(std::move(BufferLoad(n)));
      }
      if (spec.remap_func) {
        n->indices = spec.remap_func(op->indices, analyzer_);
      }
      return std::move(BufferLoad(n));
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto it = buffer_map_.find(op->buffer->data.get());
    if (it != buffer_map_.end()) {
      auto n = CopyOnWrite(op);
      const BufferUpdateSpec& spec = it->second;
      n->buffer = spec.new_buffer;
      if (spec.store_rewrite_func) {
        return spec.store_rewrite_func(std::move(BufferStore(n)));
      }
      if (spec.remap_func) {
        n->indices = spec.remap_func(op->indices, analyzer_);
      }
      n->value = VisitExpr(op->value);
      return std::move(BufferStore(n));
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  BufferRegion RewriteRegion(const BufferRegion& region, const BufferUpdateSpec& spec) {
    if (spec.region_rewrite_func) {
      return BufferRegion(spec.new_buffer, spec.region_rewrite_func(region->region));
    }
    Array<PrimExpr> begin_indices;
    Array<PrimExpr> end_indices;
    for (const Range& range : region->region) {
      begin_indices.push_back(range->min);
      if (analyzer_->CanProve(range->extent == 1)) {
        end_indices.push_back(range->min);
      } else {
        end_indices.push_back(range->min + range->extent - 1);
      }
    }
    Array<PrimExpr> new_begin, new_end;
    if (spec.remap_func) {
      new_begin = spec.remap_func(begin_indices, analyzer_);
      new_end = spec.remap_func(end_indices, analyzer_);
    } else {
      new_begin = begin_indices;
      new_end = end_indices;
    }
    Array<Range> new_regions;
    for (size_t i = 0; i < new_begin.size(); ++i) {
      PrimExpr extent = analyzer_->Simplify(new_end[i] - new_begin[i] + 1);
      new_regions.push_back(Range::FromMinExtent(new_begin[i], extent));
    }
    return BufferRegion(spec.new_buffer, new_regions);
  }

  Stmt VisitStmt_(const BlockNode* block) final {
    for (const auto& iter_var : block->iter_vars) {
      analyzer_->Bind(iter_var->var, iter_var->dom);
    }
    auto n = CopyOnWrite(block);
    Array<Buffer> new_alloc_buffers;
    for (size_t i = 0; i < block->alloc_buffers.size(); ++i) {
      const Buffer& buffer = block->alloc_buffers[i];
      auto it = buffer_map_.find(buffer->data.get());
      if (it != buffer_map_.end()) {
        const BufferUpdateSpec& spec = it->second;
        if (!spec.new_alloc_block) {
          new_alloc_buffers.push_back(it->second.new_buffer);
        }
      } else {
        new_alloc_buffers.push_back(buffer);
      }
    }
    auto bit = new_allocs_.find(block);
    if (bit != new_allocs_.end()) {
      for (const BufferNode* buffer : bit->second) {
        new_alloc_buffers.push_back(GetRef<Buffer>(buffer));
      }
    }
    n->alloc_buffers = std::move(new_alloc_buffers);

    for (size_t i = 0; i < n->reads.size(); ++i) {
      Buffer buffer = n->reads[i]->buffer;
      auto it = buffer_map_.find(buffer->data.get());
      if (it != buffer_map_.end()) {
        const BufferUpdateSpec& spec = it->second;
        n->reads.Set(i, RewriteRegion(n->reads[i], spec));
      }
    }
    for (size_t i = 0; i < n->writes.size(); ++i) {
      Buffer buffer = n->writes[i]->buffer;
      auto it = buffer_map_.find(buffer->data.get());
      if (it != buffer_map_.end()) {
        const BufferUpdateSpec& spec = it->second;
        n->writes.Set(i, RewriteRegion(n->writes[i], spec));
      }
    }
    for (size_t i = 0; i < n->match_buffers.size(); ++i) {
      CHECK(!buffer_map_.count(n->match_buffers[i]->buffer->data.get()) &&
            !buffer_map_.count(n->match_buffers[i]->source->buffer->data.get()))
          << "Do not support schedule axes of buffer used by match_buffer at "
          << GetRef<Block>(block);
    }
    Stmt updated = IRMutatorWithAnalyzer::VisitStmt_(n.get());
    ICHECK(updated->IsInstance<BlockNode>());
    // try best to reuse block srefs
    reuse_dict_.Set(GetRef<Block>(block), Downcast<Block>(updated));
    return updated;
  }

  /*! \brief mapping from buffer var to update spec */
  std::unordered_map<const VarNode*, BufferUpdateSpec> buffer_map_;

  /*! \brief mapping from function global var to stmt srefs need update */
  std::unordered_map<const GlobalVarNode*,
                     std::unordered_set<StmtSRef, ObjectPtrHash, ObjectPtrEqual>>
      touched_srefs_;

  /*! \brief mapping from block to new allocation at this block */
  std::unordered_map<const BlockNode*, std::unordered_set<const BufferNode*>> new_allocs_;

  /*! \brief record reused blocks */
  Map<Block, Block> reuse_dict_;

  /*! \brief try rewrite block binding to simplify indices if enabled */
  // TODO(baoxinqi): try simplify block binding when work with loop schedule
  bool rewrite_block_binding_;
};

NewBufferRVInfo GetBlockAccessBufferAxes(ScheduleState self, StmtSRef stmt_sref,
                                         const Buffer& buffer) {
  // find alloc block or in function buffer map
  // do not use `GetBufferDefiningSite` for it currently throw for a function buffer
  Block alloc_block;
  const StmtNode* root_block = GetSRefTreeRoot(stmt_sref)->stmt;
  PostOrderVisit(GetRef<Stmt>(root_block), [&alloc_block, &buffer](const ObjectRef& n) {
    if (const BlockNode* b = n.as<BlockNode>()) {
      for (const Buffer& alloc_buf : b->alloc_buffers) {
        if (alloc_buf.same_as(buffer)) {
          alloc_block = GetRef<Block>(b);
          break;
        }
      }
    }
  });
  if (alloc_block.defined()) {
    auto it = self->stmt2ref.find(alloc_block.get());
    CHECK(it != self->stmt2ref.end()) << "ValueError: fail to find sref for buffer alloc block";
    return CreateBufferAxisRVs(buffer, it->second);
  } else {
    // or else try find in function buffer map
    GlobalVar gv;
    const PrimFuncNode* func = GetRootPrimFunc(self->mod, root_block, &gv);
    CHECK(func) << "ValueError: No function found for current block sref";
    for (const auto& p : func->buffer_map) {
      if (p.second.same_as(buffer)) {
        return CreateBufferAxisRVs(buffer, gv);
      }
    }
  }
  LOG(FATAL) << "ValueError: the schedulable buffer should either be allocated by some block or "
                "defined in function buffer map";
  return {BufferAxisRVContainer(), {}};
}

Array<BufferAxisRV> GetBlockAccessBufferAxes(ScheduleState self, StmtSRef block_sref,
                                             int64_t buffer_idx, bool is_write) {
  const BlockNode* block = block_sref->StmtAs<BlockNode>();
  CHECK(block) << "TypeError: `get_write_buffer_axes` expect block as input";
  int64_t n_buffer = is_write ? block->writes.size() : block->reads.size();
  CHECK(n_buffer > buffer_idx && buffer_idx >= 0) << "ValueError: buffer index out of bound";
  Buffer buffer = is_write ? block->writes[buffer_idx]->buffer : block->reads[buffer_idx]->buffer;
  return GetBlockAccessBufferAxes(self, block_sref, buffer).second;
}

void CheckBufferValidity(ScheduleState self, const BufferAxisRV& buffer_axis) {
  // General checks:
  // (1) the buffer to schedule should not take non-trivial strides
  // (2) the block or function that allocate the buffer is not expired
  // (3) the buffer is not expired, that is, check it is allocated by the alloc_ref obj
  // (4) the buffer axis is not expired, it is still in container's axes and
  //    is not splitted or fused out.
  BufferAxisRVContainer container = buffer_axis->container;
  ICHECK(container.defined());
  Buffer buffer = container->buffer;
  ICHECK(buffer.defined());

  CHECK(!buffer->strides.defined() || buffer->strides.empty())
      << "ValueError: currently can not schedule buffer axes with non-trivial strides " << buffer;
  CHECK(std::find(container->axes.begin(), container->axes.end(), buffer_axis.get()) !=
        container->axes.end())
      << "ValueError: buffer axis is expired for buffer " << buffer;

  if (container->IsAllocatedBuffer()) {
    StmtSRef alloc_ref = Downcast<StmtSRef>(container->alloc_ref);
    CHECK(alloc_ref->stmt != nullptr)
        << "ValueError: buffer alloc block sref is expired for buffer " << buffer;
    const BlockNode* alloc_block = alloc_ref->StmtAs<BlockNode>();
    ICHECK(alloc_block != nullptr);
    CHECK(std::any_of(alloc_block->alloc_buffers.begin(), alloc_block->alloc_buffers.end(),
                      [&buffer](const Buffer& b) { return b.same_as(buffer); }))
        << "ValueError: buffer associated by the axis is expired: " << buffer;
  } else {
    GlobalVar global_var = Downcast<GlobalVar>(container->alloc_ref);
    auto f = self->mod->Lookup(global_var);
    CHECK(f.defined() && f->IsInstance<PrimFuncNode>())
        << "ValueError: the function define the buffer " << buffer
        << " is not found in module: " << global_var;
    PrimFunc alloc_func = Downcast<PrimFunc>(f);
    CHECK(std::any_of(alloc_func->buffer_map.begin(), alloc_func->buffer_map.end(),
                      [&buffer](const auto& p) { return p.second.same_as(buffer); }))
        << "ValueError: buffer associated by the axis is expired: " << buffer;
  }
}

Array<BufferAxisRV> SplitBuffer(ScheduleState self, BufferAxisRV buffer_axis,
                                const Array<Optional<PrimExpr>>& factor_rvs) {
  size_t n = factor_rvs.size();
  CHECK_GE(n, 2U) << "ValueError: `split_buffer` requires at least 2 parts";
  if (n == 2 && factor_rvs[0].defined() && factor_rvs[1].defined()) {  // fast path
    return SplitBuffer(self, buffer_axis, factor_rvs[0].value(), factor_rvs[1].value());
  }
  CheckBufferValidity(self, buffer_axis);

  // find out the None
  PrimExpr outer_volume = 1;  // volume except None axis
  std::vector<PrimExpr> factors;
  factors.reserve(n);
  int p = -1;
  for (size_t i = 0; i < n; ++i) {
    PrimExpr factor = factor_rvs[i].value_or(Integer(-1));
    if (const IntImmNode* imm = factor.as<IntImmNode>()) {
      if (imm->value == -1) {
        CHECK_EQ(p, -1)
            << "ValueError: `split_buffer` requires at most one `None` factor, but gets: "
            << factor_rvs;
        p = i;
        factors.emplace_back(Integer(-1));
        continue;
      }
    }
    outer_volume = outer_volume * factor;
    factors.emplace_back(std::move(factor));
  }

  if (p >= 0) {
    arith::Analyzer analyzer;
    bool dividable = analyzer.CanProve(floormod(buffer_axis->extent, outer_volume) == 0);
    CHECK(p == 0 || dividable)
        << "ValueError: invalid factors for `split_buffer`, the axis extent is "
        << buffer_axis->extent
        << ", but factors are: " << Array<PrimExpr>{factors.begin(), factors.end()};
    factors[p] =
        analyzer.Simplify(floordiv(buffer_axis->extent, outer_volume) + (dividable ? 0 : 1));
  }
  std::vector<PrimExpr> strides(factors.size());
  for (int i = n - 1; i >= 0; --i) {
    strides[i] = i < static_cast<int>(n - 1) ? strides[i + 1] * factors[i + 1] : 1;
  }

  // Split from left to right
  std::vector<BufferAxisRV> results(n, BufferAxisRV{nullptr});
  BufferAxisRV cur_axis = buffer_axis;
  for (size_t i = 0; i < n - 1; ++i) {
    Array<BufferAxisRV> parts = SplitBuffer(self, cur_axis, factors[i], strides[i]);
    ICHECK_EQ(parts.size(), 2U);
    results[i] = parts[0];
    cur_axis = parts[1];
  }
  results[n - 1] = cur_axis;
  return results;
}

/*! \brief Helper function to perform buffer update in ir nodes */
static void DoBufferUpdate(ScheduleState self, BufferAxisRVContainer container,
                           const Array<PrimExpr>& new_shape,
                           const std::vector<const BufferAxisRVNode*>& new_refs,
                           const RemapF& remap_func) {
  Buffer buffer = container->buffer;
  const BufferNode* origin_buffer = buffer.get();
  auto new_buffer = buffer.CopyOnWrite();
  auto n = container.get_mutable();
  new_buffer->shape = new_shape;
  n->buffer = GetRef<Buffer>(new_buffer);
  n->axes = new_refs;

  arith::Analyzer analyzer;
  BufferLayoutUpdater updater(&analyzer, true);
  BufferUpdateSpec spec;
  spec.alloc_ref = container->alloc_ref;
  spec.origin_buffer = origin_buffer;
  spec.new_buffer = GetRef<Buffer>(new_buffer);
  spec.remap_func = remap_func;
  updater.AddUpdateSpec(self, spec);
  updater.Update(self);
}

Array<BufferAxisRV> SplitBuffer(ScheduleState self, BufferAxisRV buffer_axis,
                                const PrimExpr& nparts, const PrimExpr& factor) {
  // Split checks:
  // (1) general buffer validity
  CheckBufferValidity(self, buffer_axis);
  CHECK(nparts.defined() && factor.defined());
  BufferAxisRVContainer container = buffer_axis->container;
  Buffer buffer = container->buffer;
  int change_idx = -1;
  Array<BufferAxisRV> results;
  Array<PrimExpr> new_shape;
  std::vector<const BufferAxisRVNode*> new_refs;
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    if (container->axes[i] == buffer_axis.get()) {
      change_idx = i;
      BufferAxisRV outer(container, nparts);
      BufferAxisRV inner(container, factor);
      new_shape.push_back(nparts);
      new_shape.push_back(factor);
      new_refs.push_back(outer.get());
      new_refs.push_back(inner.get());
      results.push_back(outer);
      results.push_back(inner);
    } else {
      new_shape.push_back(buffer->shape[i]);
      new_refs.push_back(container->axes[i]);
    }
  }
  ICHECK(change_idx >= 0);

  auto remap_func = [change_idx, &factor](const Array<PrimExpr>& indices,
                                          arith::Analyzer* analyzer) {
    Array<PrimExpr> new_indices;
    for (size_t i = 0; i < indices.size(); ++i) {
      if (i == static_cast<size_t>(change_idx)) {
        new_indices.push_back(analyzer->Simplify(floordiv(indices[i], factor)));
        new_indices.push_back(analyzer->Simplify(floormod(indices[i], factor)));
      } else {
        new_indices.push_back(indices[i]);
      }
    }
    return new_indices;
  };
  DoBufferUpdate(self, container, new_shape, new_refs, remap_func);
  return results;
}

BufferAxisRV FuseBuffer(ScheduleState self, BufferAxisRV outer, BufferAxisRV inner) {
  // Fuse checks:
  // (1) general buffer validity
  // (2) the outer and inner axis should belong to same buffer and be adjacent
  CheckBufferValidity(self, outer);
  BufferAxisRVContainer container = outer->container;
  CHECK(container.same_as(inner->container))
      << "ValueError: buffer axes to fuse must belong to same buffer";
  Buffer buffer = container->buffer;

  int change_idx = -1;
  BufferAxisRV result(container, PrimExpr());
  Array<PrimExpr> new_shape;
  std::vector<const BufferAxisRVNode*> new_refs;
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    if (container->axes[i] == outer.get()) {
      change_idx = i;
      i += 1;
      CHECK(i < buffer->shape.size() && container->axes[i] == inner.get())
          << "ValueError: buffer axes to fuse are not adjacent";
      BufferAxisRV fuse(container, outer->extent * inner->extent);
      new_shape.push_back(buffer->shape[i - 1] * buffer->shape[i]);
      new_refs.push_back(fuse.get());
      result = std::move(fuse);
    } else {
      new_shape.push_back(buffer->shape[i]);
      new_refs.push_back(container->axes[i]);
    }
  }
  CHECK(change_idx >= 0) << "ValueError: buffer dim is expired for buffer " << buffer;

  auto remap_func = [change_idx, &inner](const Array<PrimExpr>& indices,
                                         arith::Analyzer* analyzer) {
    Array<PrimExpr> new_indices;
    for (size_t i = 0; i < indices.size(); ++i) {
      if (i == static_cast<size_t>(change_idx)) {
        ICHECK(i < indices.size() - 1);
        i += 1;
        new_indices.push_back(analyzer->Simplify(indices[i - 1] * inner->extent + indices[i]));
      } else {
        new_indices.push_back(indices[i]);
      }
    }
    return new_indices;
  };
  DoBufferUpdate(self, container, new_shape, new_refs, remap_func);
  return result;
}

void ReorderBuffer(ScheduleState self, const Array<BufferAxisRV>& order) {
  // Reorder checks:
  // (1) general buffer validity
  // (2) the order axes should belong to same buffer
  CHECK(!order.empty()) << "ValueError: 'reorder_buffer' expects 'order' to be an non-empty list";
  CheckBufferValidity(self, order[0]);
  BufferAxisRVContainer container = order[0]->container;
  Buffer buffer = container->buffer;
  std::unordered_map<const BufferAxisRVNode*, int64_t> axes_origin_pos;
  for (const BufferAxisRV& buffer_axis : order) {
    CHECK(container.same_as(buffer_axis->container))
        << "ValueError: buffer axes to reorder must belong to same buffer";
    auto it = axes_origin_pos.find(buffer_axis.get());
    CHECK(it == axes_origin_pos.end())
        << "ValueError: 'reorder_buffer' expects an array of unique array, but get duplicate";
    int64_t idx = -1;
    for (size_t k = 0; k < buffer->shape.size(); ++k) {
      if (container->axes[k] == buffer_axis.get()) {
        idx = k;
        break;
      }
    }
    CHECK(idx >= 0) << "ValueError: buffer axes expired for buffer " << buffer;
    axes_origin_pos.insert(it, {buffer_axis.get(), idx});
  }
  size_t pos = 0;
  Array<PrimExpr> new_shape;
  std::vector<const BufferAxisRVNode*> new_refs;
  std::vector<size_t> indice_mapping;
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    auto it = axes_origin_pos.find(container->axes[i]);
    if (it != axes_origin_pos.end()) {
      const BufferAxisRV& transpose_axis = order[pos];
      pos += 1;
      int64_t transpose_pos = axes_origin_pos.at(transpose_axis.get());
      new_shape.push_back(buffer->shape[transpose_pos]);
      new_refs.push_back(container->axes[transpose_pos]);
      indice_mapping.push_back(transpose_pos);
    } else {
      new_shape.push_back(buffer->shape[i]);
      new_refs.push_back(container->axes[i]);
      indice_mapping.push_back(i);
    }
  }
  ICHECK_EQ(pos, order.size());

  auto remap_func = [&indice_mapping](const Array<PrimExpr>& indices, arith::Analyzer* analyzer) {
    ICHECK_EQ(indices.size(), indice_mapping.size());
    Array<PrimExpr> new_indices;
    for (size_t i = 0; i < indices.size(); ++i) {
      new_indices.push_back(indices[indice_mapping[i]]);
    }
    return new_indices;
  };
  DoBufferUpdate(self, container, new_shape, new_refs, remap_func);
}

void StackBuffer(ScheduleState self, BufferAxisRV axis0, BufferAxisRV axis1) {
  // Reorder checks:
  // (1) general buffer validity
  // (2) two buffer must be not same, and compatible except the stacked axis extent.
  // (3) two stack axis must be at the same position in each belonging buffer.
  arith::Analyzer analyzer;
  CheckBufferValidity(self, axis0);
  CheckBufferValidity(self, axis1);
  BufferAxisRVContainer container0 = axis0->container;
  BufferAxisRVContainer container1 = axis1->container;
  Buffer buffer0 = container0->buffer;
  Buffer buffer1 = container1->buffer;
  CHECK_NE(buffer0.get(), buffer1.get())
      << "ValueError: 'stack_buffer' expects two axes from different buffer";
  CHECK_EQ(buffer0->shape.size(), buffer1->shape.size())
      << "ValueError: 'stack_buffer' expects two buffer have same number of dimensions: " << buffer0
      << " " << buffer1;
  CHECK(buffer0->buffer_type == buffer1->buffer_type &&
        buffer0->data_alignment == buffer1->data_alignment && buffer0->dtype == buffer1->dtype &&
        analyzer.CanProve(buffer0->elem_offset == buffer1->elem_offset) &&
        buffer0->offset_factor == buffer1->offset_factor && buffer0.scope() == buffer1.scope());
  size_t ndim = buffer0->shape.size();
  size_t pos0 = std::find(container0->axes.begin(), container0->axes.end(), axis0.get()) -
                container0->axes.begin();
  size_t pos1 = std::find(container1->axes.begin(), container1->axes.end(), axis1.get()) -
                container1->axes.begin();
  CHECK_EQ(pos0, pos1)
      << "ValueError: 'stack_buffer' expects two axes at the same dimension position";
  for (size_t i = 0; i < ndim; ++i) {
    if (i != pos0) {
      CHECK(analyzer.CanProve(buffer0->shape[i] == buffer1->shape[i]))
          << "ValueError: buffer dimension not match at " << i;
    }
  }

  BufferLayoutUpdater updater(&analyzer, true);
  PrimExpr stack_extent = buffer0->shape[pos0] + buffer1->shape[pos0];
  auto n = make_object<BufferNode>(*buffer0.get());
  n->data = buffer0->data.copy_with_suffix("_" + buffer1->name);
  n->shape.Set(pos0, stack_extent);
  n->name = buffer0->name + "_" + buffer1->name;
  Buffer new_buffer = Buffer(n);

  BufferUpdateSpec spec0, spec1;
  spec0.alloc_ref = container0->alloc_ref;
  spec0.origin_buffer = buffer0.get();
  spec0.new_buffer = new_buffer;
  spec1.alloc_ref = container1->alloc_ref;
  spec1.origin_buffer = buffer1.get();
  spec1.new_buffer = new_buffer;
  spec1.remap_func = [pos0, &buffer0](const Array<PrimExpr>& indices, arith::Analyzer* analyzer) {
    Array<PrimExpr> new_indices = indices;
    new_indices.Set(pos0, analyzer->Simplify(indices[pos0] + buffer0->shape[pos0]));
    return new_indices;
  };

  // find allocation point for merged buffer
  if (container0->IsFunctionBuffer() && container1->IsFunctionBuffer()) {
    CHECK(container0->alloc_ref.same_as(container1->alloc_ref))
        << "ValueError: 'stack_buffer' expects two buffer be either both function buffers or "
           "buffers allocated by block from the same function";
    spec1.delete_param = true;

  } else if (container0->IsAllocatedBuffer() && container1->IsAllocatedBuffer()) {
    StmtSRef sref0 = Downcast<StmtSRef>(container0->alloc_ref);
    StmtSRef sref1 = Downcast<StmtSRef>(container1->alloc_ref);
    StmtSRef root_sref = LowestCommonAncestor({sref0, sref1}, GetSRefTreeRoot(sref0));
    const BlockNode* alloc_block;
    while (root_sref.defined()) {
      alloc_block = root_sref->StmtAs<BlockNode>();
      if (alloc_block) break;
      root_sref = GetRef<StmtSRef>(root_sref->parent);
    }
    CHECK(alloc_block)
        << "ValueError: 'stack_buffer' fail to find lca allocation point for merged buffer";
    spec0.new_alloc_block = alloc_block;
    spec1.new_alloc_block = alloc_block;
  } else {
    LOG(FATAL) << "ValueError: 'stack_buffer' expects two buffer be either both function buffers "
                  "or buffers allocated by block from the same function";
  }

  updater.AddUpdateSpec(self, spec0);
  updater.AddUpdateSpec(self, spec1);
  updater.Update(self);

  // update rv refs
  auto container0_mutable = container0.get_mutable();
  container0_mutable->buffer = new_buffer;
  container0_mutable->alloc_ref = spec0.alloc_ref;
  auto container1_mutable = container1.get_mutable();
  container1_mutable->buffer = new_buffer;
  container1_mutable->alloc_ref = spec1.alloc_ref;

  axis0.get_mutable()->extent = stack_extent;
  axis1.get_mutable()->extent = stack_extent;
}

void ReplaceBuffer(ScheduleState self, StmtSRef stmt_sref, Buffer origin_buffer, Buffer new_buffer,
                   PackedFunc load_rewrite_func, PackedFunc store_rewrite_func,
                   PackedFunc region_rewrite_func) {
  BufferAxisRVContainer container = GetBlockAccessBufferAxes(self, stmt_sref, origin_buffer).first;
  arith::Analyzer analyzer;
  BufferLayoutUpdater updater(&analyzer, true);
  BufferUpdateSpec spec;
  spec.alloc_ref = container->alloc_ref;
  spec.origin_buffer = container->buffer.get();
  spec.new_buffer = new_buffer;

  if (load_rewrite_func.body()) {
    spec.load_rewrite_func = [&load_rewrite_func](const BufferLoad& load) {
      return TypedPackedFunc<PrimExpr(const BufferLoad&)>(load_rewrite_func)(load);
    };
  }
  if (store_rewrite_func.body()) {
    spec.store_rewrite_func = [&store_rewrite_func](const BufferStore& store) {
      return TypedPackedFunc<Stmt(const BufferStore&)>(store_rewrite_func)(store);
    };
  }
  if (region_rewrite_func.body()) {
    spec.region_rewrite_func = [&region_rewrite_func](const Array<Range>& region) {
      return TypedPackedFunc<Array<Range>(const Array<Range>&)>(region_rewrite_func)(region);
    };
  }

  updater.AddUpdateSpec(self, spec);
  updater.Update(self);
}

}  // namespace schedule
}  // namespace tir
}  // namespace tvm
