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
#include <tvm/arith/analyzer.h>
#include <tvm/arith/pattern.h>
#include <tvm/tir/op.h>
#include <tvm/tir/schedule/schedule.h>
#include <tvm/tir/stmt_functor.h>

#include "../../../tir/schedule/analysis.h"

namespace tvm {
namespace edgex {
namespace uniform_schedule {

using namespace tir;

/*! \brief Access bound on particular buffer dimension.
 * For a buffer dim of read buffer A, when write B[v0, v1, ..., vk]
 * it would read ranges A[...,
 *    (l0 * v0 + l1 * v1 + ... + lk * vk + lk+1:
 *     u0 * v0 + u1 * v1 + ... + vk * vk + v=uk+1, ...]
 */
class TileRelationNode : public runtime::Object {
 public:
  Array<PrimExpr> lower_bound_coeffs;
  Array<PrimExpr> upper_bound_coeffs;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("lower_bound_coeffs", &lower_bound_coeffs);
    v->Visit("upper_bound_coeffs", &upper_bound_coeffs);
  }

  static constexpr const char* _type_key = "edgex.uniform_schedule.TileRelation";
  TVM_DECLARE_FINAL_OBJECT_INFO(TileRelationNode, Object);
};

class TileRelation : public runtime::ObjectRef {
 public:
  TVM_DLL TileRelation(size_t write_ndim) {
    auto n = make_object<TileRelationNode>();
    n->lower_bound_coeffs = Array<PrimExpr>(write_ndim + 1, 0);
    n->upper_bound_coeffs = Array<PrimExpr>(write_ndim + 1, 0);
    data_ = std::move(n);
  }

  void Merge(const Array<PrimExpr>& lower_bound_coeffs, const Array<PrimExpr>& upper_bound_coeffs) {
    auto n = static_cast<TileRelationNode*>(this->get_mutable());
    ICHECK_EQ(n->lower_bound_coeffs.size(), lower_bound_coeffs.size());
    for (size_t i = 0; i < lower_bound_coeffs.size(); ++i) {
      n->lower_bound_coeffs.Set(i, min(n->lower_bound_coeffs[i], lower_bound_coeffs[i]));
    }
    ICHECK_EQ(n->upper_bound_coeffs.size(), upper_bound_coeffs.size());
    for (size_t i = 0; i < upper_bound_coeffs.size(); ++i) {
      n->upper_bound_coeffs.Set(i, max(n->upper_bound_coeffs[i], upper_bound_coeffs[i]));
    }
  }

  void Reset(const Array<PrimExpr>& lower_bound_coeffs, const Array<PrimExpr>& upper_bound_coeffs) {
    auto n = static_cast<TileRelationNode*>(this->get_mutable());
    n->lower_bound_coeffs = lower_bound_coeffs;
    n->upper_bound_coeffs = upper_bound_coeffs;
  }

  TVM_DEFINE_OBJECT_REF_METHODS(TileRelation, ObjectRef, TileRelationNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TileRelationNode);
};

TVM_REGISTER_NODE_TYPE(TileRelationNode);

Map<Buffer, Array<TileRelation>> GetTileRelations(const BlockRealize& block_realize,
                                                  const Array<For>& loops) {
  arith::Analyzer analyzer;
  Block block = block_realize->block;
  // block var -> iter binding
  Map<Var, PrimExpr> bindings;
  for (size_t i = 0; i < block_realize->iter_values.size(); ++i) {
    bindings.Set(block->iter_vars[i]->var, block_realize->iter_values[i]);
  }
  // loop var -> loop domain
  Map<Var, arith::IntSet> dom_map;
  for (const For& loop : loops) {
    dom_map.Set(loop->loop_var, arith::IntSet::FromMinExtent(loop->min, loop->extent));
  }
  // bind_loop[i] = loop var match the ith axis of write buffer
  Array<Var> bind_loop;

  // Check (1): there is single write buffer
  if (block->writes.size() != 1) {
    LOG(INFO) << "not single write buffer";
    return {};
  }

  // Check (2): the write region cover full buffer and the write region are trivial points
  std::unordered_set<const VarNode*> used_loop_vars;
  const Buffer& write_buffer = block->writes[0]->buffer;
  const auto& write_region = block->writes[0]->region;
  for (size_t i = 0; i < write_buffer->shape.size(); ++i) {
    const Range& range = write_region[i];
    if (!is_one(range->extent)) {
      // not a single point
      LOG(INFO) << "not single point";
      return {};
    }
    PrimExpr point = analyzer.Simplify(Substitute(range->min, bindings));
    if (!point->IsInstance<VarNode>()) {
      // indice is not loop var
      LOG(INFO) << "not loop var";
      return {};
    }
    Var loop_var = Downcast<Var>(point);
    auto it = dom_map.find(loop_var);
    if (it == dom_map.end()) {
      // indice is not loop var
      LOG(INFO) << "not loop var";
      return {};
    }
    if (!is_zero((*it).second.min()) ||
        !analyzer.CanProveEqual((*it).second.max() + 1, write_buffer->shape[i])) {
      // do not cover the buffer shape
      LOG(INFO) << "not cover" << i << " " << (*it).second << " " << write_buffer->shape[i];
      return {};
    }
    if (used_loop_vars.count(loop_var.get())) {
      return {};
    }
    used_loop_vars.insert(loop_var.get());
    bind_loop.push_back(loop_var);
  }

  // Assume write B[v0, v1, ..., vk], we want to know how large is the coresponding read regions.
  size_t write_ndim = write_buffer->shape.size();
  Array<Var> dummy_vars;
  for (size_t i = 0; i < write_ndim; ++i) {
    Var dummy("k" + std::to_string(i), DataType::Int(32));
    dummy_vars.push_back(dummy);
    if (!analyzer.CanProveEqual(dom_map[bind_loop[i]].max(), 0)) {
      dom_map.Set(bind_loop[i], arith::IntSet::SinglePoint(dummy));
    }
  }
  Map<Buffer, Array<TileRelation>> results;

  for (size_t i = 0; i < block->reads.size(); ++i) {
    const BufferRegion& read = block->reads[i];
    const Buffer& read_buffer = read->buffer;
    // check reduce buffer
    if (read_buffer == write_buffer) {
      if (!StructuralEqual()(read, block->writes[0])) {
        // not a simple reduce
        return {};
      }
    }

    const auto& region = read->region;
    size_t read_ndim = region.size();

    // init relations for the buffer
    bool need_merge = true;
    if (!results.count(read_buffer)) {
      Array<TileRelation> dim_rels;
      for (size_t j = 0; j < read_ndim; ++j) {
        dim_rels.push_back(TileRelation(write_ndim));
      }
      results.Set(read_buffer, dim_rels);
      need_merge = false;
    }

    // update the relations
    auto it = results.find(read_buffer);
    for (size_t j = 0; j < read_ndim; ++j) {
      PrimExpr access_min = Substitute(region[j]->min, bindings);
      access_min = analyzer.Simplify(arith::EvalSet(access_min, dom_map).min());
      auto min_coeffs = arith::DetectLinearEquation(access_min, dummy_vars);

      PrimExpr access_max = Substitute(region[j]->min + region[j]->extent - 1, bindings);
      access_max = analyzer.Simplify(arith::EvalSet(access_max, dom_map).max());
      auto max_coeffs = arith::DetectLinearEquation(access_max, dummy_vars);

      if (std::all_of(min_coeffs.begin(), min_coeffs.end(),
                      [](const PrimExpr& e) { return is_const_int(e); }) &&
          std::all_of(max_coeffs.begin(), max_coeffs.end(),
                      [](const PrimExpr& e) { return is_const_int(e); })) {
        TileRelation rel = Downcast<TileRelation>((*it).second.GetArrayNode()->operator[](j));
        if (need_merge) {
          rel.Merge(min_coeffs, max_coeffs);
        } else {
          rel.Reset(min_coeffs, max_coeffs);
        }
      }
    }
  }
  return results;
}

Map<Buffer, Array<TileRelation>> GetTileRelations(Schedule schedule, const StmtSRef& block_sref) {
  BlockRealize block_realize = GetBlockRealize(schedule->state(), block_sref);
  Array<For> loops;
  const StmtSRefNode* root = GetSRefTreeRoot(block_sref).get();
  for (auto p = block_sref->parent; p != root; p = p->parent) {
    const ForNode* loop = p->StmtAs<ForNode>();
    if (loop == nullptr) {
      // should not be a nested block
      return {};
    }
    loops.push_back(GetRef<For>(loop));
  }
  return GetTileRelations(block_realize, loops);
}

Array<TileRelation> GetRootTileRelations(const Buffer& buffer) {
  Array<TileRelation> results;
  size_t ndim = buffer->shape.size();
  for (size_t i = 0; i < ndim; ++i) {
    auto n = make_object<TileRelationNode>();
    n->lower_bound_coeffs = Array<PrimExpr>(ndim + 1, 0);
    n->lower_bound_coeffs.Set(i, 1);
    n->upper_bound_coeffs = Array<PrimExpr>(ndim + 1, 0);
    n->upper_bound_coeffs.Set(i, 1);
    results.push_back(TileRelation(n));
  }
  return results;
}

Array<TileRelation> GetIrrelevantTileRelations(const Buffer& buffer) {
  Array<TileRelation> results;
  size_t ndim = buffer->shape.size();
  for (size_t i = 0; i < ndim; ++i) {
    auto n = make_object<TileRelationNode>();
    n->lower_bound_coeffs = Array<PrimExpr>(ndim + 1, 0);
    n->lower_bound_coeffs.Set(ndim, buffer->shape[i]);
    n->upper_bound_coeffs = Array<PrimExpr>(ndim + 1, 0);
    n->upper_bound_coeffs.Set(ndim, buffer->shape[i]);
    results.push_back(TileRelation(n));
  }
  return results;
}

Array<PrimExpr> EstimateTileSizes(const Buffer& buffer, const Array<TileRelation>& relations,
                                  const Array<PrimExpr>& placeholders) {
  Array<PrimExpr> results;
  arith::Analyzer analyzer;
  for (size_t i = 0; i < relations.size(); ++i) {
    const TileRelation& rel = relations[i];
    ICHECK_EQ(rel->lower_bound_coeffs.size(), placeholders.size() + 1);
    ICHECK_EQ(rel->upper_bound_coeffs.size(), placeholders.size() + 1);
    bool fallback_to_fullregion = false;
    PrimExpr tile_size = 0;
    for (size_t j = 0; j < placeholders.size(); ++j) {
      if (analyzer.CanProveEqual(rel->lower_bound_coeffs[j], rel->upper_bound_coeffs[j])) {
        // (u0 * (offset + (tile_size0 - 1)) - l0 * offset) => l0 * (tile_size0 - 1), when u0 == l0
        tile_size += rel->lower_bound_coeffs[j] * placeholders[j] - rel->lower_bound_coeffs[j];
      } else {
        fallback_to_fullregion = true;
        break;
      }
    }
    if (fallback_to_fullregion) {
      results.push_back(buffer->shape[i]);
    } else {
      tile_size += rel->upper_bound_coeffs.back() - rel->lower_bound_coeffs.back() + 1;
      results.push_back(analyzer.Simplify(tile_size));
    }
  }
  return results;
}

Array<TileRelation> ComposeTileRelations(const Array<TileRelation>& r1,
                                         const Array<TileRelation>& r2) {
  ICHECK(!r1.empty() && !r2.empty());
  size_t read_ndim = r1.size();
  size_t write_ndim = r2[0]->lower_bound_coeffs.size() - 1;
  size_t K = r2.size();

  Array<TileRelation> results;
  for (size_t i = 0; i < read_ndim; ++i) {
    TileRelation rel(write_ndim);
    auto n = rel.CopyOnWrite();

    ICHECK_EQ(r1[i]->lower_bound_coeffs.size(), K + 1);
    for (size_t j = 0; j < write_ndim + 1; ++j) {
      PrimExpr sum = 0;
      for (size_t k = 0; k < K; ++k) {
        sum += r1[i]->lower_bound_coeffs[k] * r2[k]->lower_bound_coeffs[j];
      }
      if (j == write_ndim) {
        sum += r1[i]->lower_bound_coeffs[K];
      }
      n->lower_bound_coeffs.Set(j, sum);
    }

    ICHECK_EQ(r1[i]->upper_bound_coeffs.size(), K + 1);
    for (size_t j = 0; j < write_ndim + 1; ++j) {
      PrimExpr sum = 0;
      for (size_t k = 0; k < K; ++k) {
        sum += r1[i]->upper_bound_coeffs[k] * r2[k]->upper_bound_coeffs[j];
      }
      if (j == write_ndim) {
        sum += r1[i]->upper_bound_coeffs[K];
      }
      n->upper_bound_coeffs.Set(j, sum);
    }

    results.push_back(TileRelation(rel));
  }
  return results;
}

TVM_REGISTER_GLOBAL("edgex.uniform_schedule.GetTileRelations")
    .set_body_typed([](Schedule schedule, const StmtSRef& block_sref) {
      return GetTileRelations(schedule, block_sref);
    });

TVM_REGISTER_GLOBAL("edgex.uniform_schedule.GetTileRelationsByRV")
    .set_body_typed([](Schedule schedule, const BlockRV& block_rv) {
      auto state = schedule->state();
      StmtSRef block_sref = schedule->GetSRef(block_rv);
      return GetTileRelations(schedule, block_sref);
    });

TVM_REGISTER_GLOBAL("edgex.uniform_schedule.EstimateTileSizes").set_body_typed(EstimateTileSizes);

TVM_REGISTER_GLOBAL("edgex.uniform_schedule.ComposeTileRelations")
    .set_body_typed(ComposeTileRelations);

TVM_REGISTER_GLOBAL("edgex.uniform_schedule.GetIrrelevantTileRelations")
    .set_body_typed(GetIrrelevantTileRelations);

TVM_REGISTER_GLOBAL("edgex.uniform_schedule.GetRootTileRelations")
    .set_body_typed(GetRootTileRelations);

TVM_REGISTER_GLOBAL("edgex.uniform_schedule.GetExprRepr").set_body_typed([](const PrimExpr& e) {
  std::stringstream ss;
  ss << e;
  return ss.str();
});

}  // namespace uniform_schedule
}  // namespace edgex
}  // namespace tvm
