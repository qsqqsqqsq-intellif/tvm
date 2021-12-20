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
 * \file inject_handshake_intrin.cc
 */
#include <tvm/arith/int_solver.h>
#include <tvm/arith/pattern.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <numeric>
#include <stack>

#include "../../../../tir/schedule/utils.h"
#include "../../arith/iter_transform_detector.h"
#include "../attrs.h"
#include "../edgex_ir_utils.h"
#include "../op/builtin.h"

namespace tvm {
namespace tir {

using tvm::runtime::StorageRank;
using tvm::runtime::StorageScope;
using tvm::tir::edgex::NNPAddArg;

static const int EIDMA_NU_C1DHWC0_MODE = 0;
static const int EIDMA_TRANSPOSE_MODE = 2;

static const int EODMA_TRANSPOSE_MODE = 1;

struct DMAShapeAttributes {
  std::array<size_t, 4U> loop_num;
  std::array<size_t, 4U> loop_src_index;
  std::array<size_t, 3U> loop_src_stride;
  std::array<size_t, 3U> loop_dst_stride;
};

class DMAIntrinRewriter : public StmtExprMutator {
 public:
  friend class DMAIntrinScopeRewriter;

  explicit DMAIntrinRewriter(bool verbose) : verbose_(verbose) {}

  /**
   *! \brief Rewrite entrance.
   */
  Stmt RewriteDMA(const Stmt& body, const std::string& intrin_hint_name);

  void SetExtraAttr(const std::string& key, const std::string& value) {
    extra_dma_attrs_.emplace(key, value);
  }

 private:
  /**
   *! \brief Visit dma scope and return target load/store stmt.
   */
  Stmt CollectDmaScope(const Stmt& stmt);

  /**
   *! \brief Normalize loop variable domain for target stmt.
   */
  Stmt WithNormalizedDomain(const Stmt& stmt);

  /**
   *! \brief helper function to create tvm_access_ptr()
   */
  PrimExpr CreateAccessPtr(const Var& var, const DataType& dtype, const PrimExpr& offset,
                           const PrimExpr& index, const std::string& mask);

  Stmt CreateEidmaLoad(const Var& src_buffer, const DataType& src_dtype, const PrimExpr& src_base,
                       const PrimExpr& src_offset, const Var& dst_buffer, const DataType& dst_dtype,
                       const PrimExpr& dst_base, const PrimExpr& dst_offset);

  Stmt CreateEwdmaLoad(const Var& src_buffer, const DataType& src_dtype, const PrimExpr& src_base,
                       const PrimExpr& src_offset, const Var& dst_buffer, const DataType& dst_dtype,
                       const PrimExpr& dst_base, const PrimExpr& dst_offset);

  Stmt CreateEodmaStore(const Var& src_buffer, const DataType& src_dtype, const PrimExpr& src_base,
                        const PrimExpr& src_offset, const Var& dst_buffer,
                        const DataType& dst_dtype, const PrimExpr& dst_base,
                        const PrimExpr& dst_offset);

  Stmt CreateVidmaLoad(const Var& src_buffer, const DataType& src_dtype, const PrimExpr& src_base,
                       const PrimExpr& src_offset, const Var& dst_buffer, const DataType& dst_dtype,
                       const PrimExpr& dst_base, const PrimExpr& dst_offset);

  Stmt CreateVodmaStore(const Var& src_buffer, const DataType& src_dtype, const PrimExpr& src_base,
                        const PrimExpr& src_offset, const Var& dst_buffer,
                        const DataType& dst_dtype, const PrimExpr& dst_base,
                        const PrimExpr& dst_offset);

  Stmt CreateIdmaLoad(const Var& src_buffer, const DataType& src_dtype, const PrimExpr& src_base,
                      const PrimExpr& src_offset, const Var& dst_buffer, const DataType& dst_dtype,
                      const PrimExpr& dst_base, const PrimExpr& dst_offset, const PrimExpr& cond);

  Stmt CreateWdmaLoad(const Var& src_buffer, const DataType& src_dtype, const PrimExpr& src_base,
                      const PrimExpr& src_offset, const Var& dst_buffer, const DataType& dst_dtype,
                      const PrimExpr& dst_base, const PrimExpr& dst_offset);

  Stmt CreateBdmaLoad(const Var& src_buffer, const DataType& src_dtype, const PrimExpr& src_base,
                      const PrimExpr& src_offset, const Var& dst_buffer, const DataType& dst_dtype,
                      const PrimExpr& dst_base, const PrimExpr& dst_offset);

  Stmt CreateOdmaStore(const Var& src_buffer, const DataType& src_dtype, const PrimExpr& src_base,
                       const PrimExpr& src_offset, const Var& dst_buffer, const DataType& dst_dtype,
                       const PrimExpr& dst_base, const PrimExpr& dst_offset);

  // reset states when go into dma rewrite scope
  void ResetStates(const Map<Var, Range>& dom_map) {
    dom_map_ = dom_map;
    loop_vars_.clear();
    dom_intsets_ = Map<Var, arith::IntSet>();
  }

  // set extra attrs specified by nnp_dma_attrs annotation
  void SetExtraDmaAttrs(CallNode* call) {
    for (const auto& p : extra_dma_attrs_) {
      NNPAddArg(call, p.first, p.second);
    }
  }

  // update dom map for conditional scope
  void UpdateConditionalDomain(const PrimExpr& condition, Map<Var, Range>*);

  Map<Var, Range> dom_map_;
  Array<Var> loop_vars_;
  std::unordered_map<std::string, std::string> extra_dma_attrs_;

  // arithmetic utilities initialized after normalize iter domain for target stmt
  arith::Analyzer dom_analyzer_;
  Map<Var, arith::IntSet> dom_intsets_;
  bool verbose_{false};
};

class DMAIntrinScopeRewriter : public StmtExprMutator {
 public:
  explicit DMAIntrinScopeRewriter(const Map<Var, Buffer>& buffer_map, bool verbose)
      : rewriter_(verbose) {}

 private:
  void ParseDMANameAndAttrs(const PrimExpr& value) {
    rewriter_.extra_dma_attrs_.clear();
    if (const StringImmNode* n = value.as<StringImmNode>()) {
      intrin_hint_name_ = n->value;
    } else if (const CallNode* n = value.as<CallNode>()) {
      std::string op_name = Downcast<Op>(n->op)->name;
      size_t beg = op_name.find_first_of("_");
      size_t end = op_name.find_last_of("_");
      ICHECK(beg != std::string::npos && end > beg);
      intrin_hint_name_ = op_name.substr(beg + 1, end - beg - 1);
      for (const auto& arg : n->args) {
        if (const StringImmNode* string_imm = arg.as<StringImmNode>()) {
          std::string kv = string_imm->value;
          size_t pos = kv.find("=");
          if (pos != std::string::npos) {
            rewriter_.SetExtraAttr(kv.substr(0, pos), kv.substr(pos + 1, kv.size() - pos - 1));
          }
        }
      }
    }
  }

  Stmt VisitStmt_(const AttrStmtNode* attr) final {
    if (in_scope_) {
      return StmtExprMutator::VisitStmt_(attr);
    }
    if (attr->attr_key == attr::nnp_dma_scope) {
      ParseDMANameAndAttrs(attr->value);
      in_scope_ = true;
      Stmt result = VisitStmt(attr->body);
      in_scope_ = false;
      return result;
    }
    return StmtExprMutator::VisitStmt_(attr);
  }

  Stmt VisitStmt_(const ForNode* loop) final {
    if (in_scope_) {
      rewriter_.ResetStates(dom_map_);
      return rewriter_.RewriteDMA(GetRef<Stmt>(loop), intrin_hint_name_);
    }
    dom_map_.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
    Stmt res = StmtExprMutator::VisitStmt_(loop);
    dom_map_.erase(loop->loop_var);
    return res;
  }

  Stmt VisitStmt_(const AllocateNode* alloc) final { return StmtExprMutator::VisitStmt_(alloc); }

  Map<Var, Range> dom_map_;
  DMAIntrinRewriter rewriter_;
  std::string intrin_hint_name_;
  bool in_scope_{false};
};

Stmt DMAIntrinRewriter::WithNormalizedDomain(const Stmt& stmt) {
  Map<Var, PrimExpr> repl_dict;
  for (const auto& p : dom_map_) {
    const Var& v = p.first;
    repl_dict.Set(v, v + p.second->min);
  }
  if (verbose_) {
    std::stringstream ss;
    for (auto& p : repl_dict) {
      ss << p.first << " - " << p.second << ", ";
    }
    LOG(INFO) << "Normalize domain: " << ss.str();
  }
  for (auto& p : repl_dict) {
    const Var& v = p.first;
    Range new_range = Range::FromMinExtent(0, dom_map_.at(v)->extent);
    dom_map_.Set(v, new_range);
    dom_intsets_.Set(v, arith::IntSet::FromRange(new_range));
  }
  dom_analyzer_.Bind(dom_map_, true);
  return Substitute(stmt, repl_dict);
}

void DMAIntrinRewriter::UpdateConditionalDomain(const PrimExpr& condition,
                                                Map<Var, Range>* dom_map) {
  Array<PrimExpr> equations;
  std::function<void(const PrimExpr&)> f_split = [&equations, &f_split](const PrimExpr& e) {
    if (const AndNode* binary = e.as<AndNode>()) {
      f_split(binary->a);
      f_split(binary->b);
    } else if (const CallNode* call = e.as<CallNode>()) {
      if (call->op.same_as(builtin::likely())) {
        f_split(call->args[0]);
      }
    } else {
      equations.push_back(e);
    }
  };
  f_split(condition);
  arith::IntConstraints constraint(loop_vars_, *dom_map, equations);
  auto result = arith::SolveInequalitiesToRange(constraint);
  ICHECK(result->relations.empty()) << "Condition " << condition << " can not be fully solved";
  for (size_t i = 0; i < result->variables.size(); ++i) {
    const Var& var = result->variables[i];
    dom_map->Set(var, result->ranges[var]);
  }
}

/**
 *! \brief Utility to collect lower intrinsic info under dma scope.
 */
Stmt DMAIntrinRewriter::CollectDmaScope(const Stmt& stmt) {
  if (const ForNode* loop = stmt.as<ForNode>()) {
    dom_map_.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
    loop_vars_.push_back(loop->loop_var);
    return CollectDmaScope(loop->body);
  } else if (const IfThenElseNode* cond = stmt.as<IfThenElseNode>()) {
    ICHECK(!cond->else_case.defined()) << "Do not support alternative cond branch";
    UpdateConditionalDomain(cond->condition, &dom_map_);
    return CollectDmaScope(cond->then_case);
  } else {
    // reach target stmt
    return WithNormalizedDomain(stmt);
  }
}

/**
 *! \brief Split address indice into base + offset format, where offset part is irrelavant to input
 *variables.
 */
static std::pair<PrimExpr, PrimExpr> GetIndiceDivision(const PrimExpr& indice,
                                                       const Array<Var>& vars) {
  PrimExpr base = 0;
  PrimExpr offset = 0;
  std::unordered_set<const VarNode*> var_set;
  for (const Var& v : vars) var_set.insert(v.get());

  std::stack<std::pair<PrimExpr, bool>> working_stack;
  working_stack.push({indice, true});
  while (!working_stack.empty()) {
    const auto& p = working_stack.top();
    PrimExpr e = p.first;
    bool sign = p.second;
    working_stack.pop();
    if (const AddNode* add = e.as<AddNode>()) {
      working_stack.push({add->a, sign});
      working_stack.push({add->b, sign});
    } else if (const SubNode* sub = e.as<SubNode>()) {
      working_stack.push({sub->a, sign});
      working_stack.push({sub->b, !sign});
    } else {
      if (UsesVar(e, [&var_set](const VarNode* v) { return var_set.count(v); })) {
        base = sign ? base + e : base - e;
      } else {
        offset = sign ? offset + e : offset - e;
      }
    }
  }
  return std::make_pair(base, offset);
}

/**
 * Solve dma intrinsic shape attributes via linear coeffs if it's simple enough.
 */
static bool InferShapeSimple(const PrimExpr& src_index, const PrimExpr& dst_index,
                             const Map<Var, Range>& dom_map, const Array<Var>& vars,
                             int max_loop_sels, bool verbose, DMAShapeAttributes* result) {
  // extract linear coefficients
  using CoeffArray = std::vector<std::pair<const VarNode*, int64_t>>;
  bool has_zero_stride = false;
  auto f_get_coeffs = [&vars, &has_zero_stride](const PrimExpr& e) -> CoeffArray {
    CoeffArray res;
    arith::Analyzer analyzer;
    Array<PrimExpr> coeffs = arith::DetectLinearEquation(e, vars);
    if (coeffs.size() != vars.size() + 1) {
      return {};
    }
    ICHECK(analyzer.CanProve(coeffs.back() == 0)) << "Offset should be trimmed before infer shape";
    for (size_t i = 0; i < vars.size(); ++i) {
      if (is_const_int(coeffs[i])) {
        int64_t value = Downcast<IntImm>(coeffs[i])->value;
        if (value < 0) return {};
        if (value == 0) has_zero_stride = true;
        res.push_back(std::make_pair(vars[i].get(), value));
      } else {
        return {};
      }
    }
    return res;
  };
  CoeffArray src_coeffs = f_get_coeffs(src_index);
  CoeffArray dst_coeffs = f_get_coeffs(dst_index);
  if (src_coeffs.size() != dst_coeffs.size()) {
    return false;
  }

  // sort coeffs from large to small
  auto cmp = [](const std::pair<const VarNode*, int64_t>& p1,
                const std::pair<const VarNode*, int64_t>& p2) { return p1.second > p2.second; };
  std::sort(src_coeffs.begin(), src_coeffs.end(), cmp);
  std::sort(dst_coeffs.begin(), dst_coeffs.end(), cmp);

  // skip unused var
  if (has_zero_stride) {
    size_t pos = 0;
    for (size_t i = 0; i < src_coeffs.size(); ++i) {
      if (src_coeffs[i].second == 0 && dst_coeffs[i].second == 0) continue;
      src_coeffs[pos] = src_coeffs[i];
      dst_coeffs[pos] = dst_coeffs[i];
      ++pos;
    }
    src_coeffs.resize(pos);
    dst_coeffs.resize(pos);
  }

  // maybe add dummy inner loop to ensure j3 loop stride be 1
  if (src_coeffs.empty() || src_coeffs.back().second != 1 || dst_coeffs.back().second != 1) {
    src_coeffs.push_back(std::make_pair(nullptr, 1));
    dst_coeffs.push_back(std::make_pair(nullptr, 1));
  }

  if (verbose) {
    std::stringstream ss;
    ss << "Try simple dma shape inference " << dst_index << " <- " << src_index << "\n";
    ss << "Source coefficients: ";
    for (size_t i = 0; i < src_coeffs.size(); ++i) {
      ss << src_coeffs[i].second << "*" << GetRef<Var>(src_coeffs[i].first) << ", ";
    }
    ss << "\nDest coefficients: ";
    for (size_t i = 0; i < src_coeffs.size(); ++i) {
      ss << dst_coeffs[i].second << "*" << GetRef<Var>(dst_coeffs[i].first) << ", ";
    }
    LOG(INFO) << ss.str();
  }

  int loop_idx = 3;                           // j3, j2, j1, j0
  std::map<size_t, int64_t> src_strides_map;  // src strides sorted
  std::unordered_map<const VarNode*, size_t> src_var_pos;
  std::array<size_t, 4U> loop_src_num;
  for (size_t i = src_coeffs.size(); i > 0; --i) {
    src_var_pos[src_coeffs[i - 1].first] = i - 1;
  }
  for (size_t i = dst_coeffs.size(); i > 0; --i) {
    const VarNode* var = dst_coeffs[i - 1].first;
    int64_t dst_stride = dst_coeffs[i - 1].second;
    ICHECK(src_var_pos.count(var)) << GetRef<Var>(var);
    size_t src_index = src_var_pos.at(var);
    int64_t src_stride = src_coeffs[src_index].second;
    if (var) {
      IntImm extent = Downcast<IntImm>(dom_map[GetRef<Var>(var)]->extent);
      ICHECK(extent.defined()) << "Domain extent for " << GetRef<Var>(var) << " should be constant";
      result->loop_num[loop_idx] = extent->value;
    } else {
      ICHECK(loop_idx == 3);
      result->loop_num[loop_idx] = 1;
    }
    result->loop_src_index[loop_idx] = src_index;
    if (loop_idx < 3) {
      result->loop_dst_stride[loop_idx] = dst_stride;
    }
    src_strides_map[src_index] = src_stride;
    // get src loop num to calculate the stride_in0
    const VarNode* src_var = src_coeffs[i - 1].first;
    if (src_var) {
      IntImm src_extent = Downcast<IntImm>(dom_map[GetRef<Var>(src_var)]->extent);
      ICHECK(src_extent.defined())
          << "Domain extent for " << GetRef<Var>(src_var) << " should be constant";
      loop_src_num[loop_idx] = src_extent->value;
    } else {
      ICHECK(loop_idx == 3);
      loop_src_num[loop_idx] = 1;
    }
    if (loop_idx == 4 - max_loop_sels && i > 1) {
      // loop sel used out
      return false;
    }
    loop_idx -= 1;
  }
  // fill source strides
  int src_loop_idx = 3;
  for (auto it = src_strides_map.rbegin(); it != src_strides_map.rend(); ++it) {
    if (src_loop_idx < 3) {
      result->loop_src_stride[src_loop_idx] = it->second;
    }
    --src_loop_idx;
  }
  // loop variables used out here
  int loop_src_index_base = 0;
  while (loop_idx >= 0) {
    loop_src_index_base += 1;
    result->loop_num[loop_idx] = 1;
    loop_src_num[loop_idx] = 1;
    result->loop_dst_stride[loop_idx] =
        loop_idx < 2 ? result->loop_dst_stride[loop_idx + 1] * result->loop_num[loop_idx + 1] : 1;
    result->loop_src_stride[loop_idx] =
        loop_idx < 2 ? result->loop_src_stride[loop_idx + 1] * loop_src_num[loop_idx + 1] : 1;
    loop_idx -= 1;
  }
  for (loop_idx = 0; loop_idx < 4; ++loop_idx) {
    if (loop_idx >= loop_src_index_base) {
      result->loop_src_index[loop_idx] += loop_src_index_base;
    } else {
      result->loop_src_index[loop_idx] = loop_idx;
    }
  }
  return true;
}

static bool DoInferShapeDesc(const arith::IterTransformDetector& detector,
                             const std::unordered_map<size_t, int64_t>& src_strides_map,
                             const std::unordered_map<size_t, int64_t>& dst_strides_map,
                             std::vector<int64_t>* p_src_shape, std::vector<int64_t>* p_src_strides,
                             std::vector<int64_t>* p_transpose, std::vector<int64_t>* p_dst_shape,
                             std::vector<int64_t>* p_dst_strides) {
  // compute src dom shape
  const auto& shape_ops = detector.shape_ops;
  if (shape_ops[0].type != arith::RESHAPE) {
    return false;
  }
  std::vector<int64_t>& src_shape = *p_src_shape;
  std::vector<int64_t>& src_strides = *p_src_strides;
  std::vector<int64_t>& transpose = *p_transpose;
  std::vector<int64_t>& dst_shape = *p_dst_shape;
  std::vector<int64_t>& dst_strides = *p_dst_strides;

  // trivial dma
  arith::ShapeOperation start_shape = shape_ops[0];
  if (shape_ops.size() == 1) {
    int64_t volume = std::accumulate(start_shape.values.begin(), start_shape.values.end(), 1,
                                     [](int64_t cur, int64_t n) { return cur * n; });
    start_shape.values[0] = volume;
    for (size_t i = 1; i < start_shape.ndim(); ++i) {
      for (size_t iter_idx : start_shape.iters[i]) {
        start_shape.iters[0].push_back(iter_idx);
      }
    }
    start_shape.values.resize(1);
    start_shape.iters.resize(1);
  }

  size_t ndim = start_shape.values.size();
  std::vector<std::vector<int64_t>> src_splitted_shape(ndim);
  std::vector<std::vector<int64_t>> src_splitted_strides(ndim);
  std::vector<std::vector<int64_t>> dst_splitted_shape(ndim);
  std::vector<std::vector<int64_t>> dst_splitted_strides(ndim);

  for (size_t i = 0; i < ndim; ++i) {
    // shape value < 0 denotes a broadcast dimension
    // set loop extent = 1
    int64_t dim = start_shape.values[i];
    if (dim > 0) {
      if (!detector.InferIterDivision(start_shape.iters[i], src_strides_map, dst_strides_map,
                                      &src_splitted_shape[i], &src_splitted_strides[i])) {
        if (detector.verbose) {
          LOG(ERROR) << "Infer iter division under src strides failed:\n"
                     << detector.FormatOp(start_shape);
        }
        return false;
      }
      if (!detector.InferIterDivision(start_shape.iters[i], dst_strides_map, src_strides_map,
                                      &dst_splitted_shape[i], &dst_splitted_strides[i])) {
        if (detector.verbose) {
          LOG(ERROR) << "Infer iter division under dst strides failed:\n"
                     << detector.FormatOp(start_shape);
        }
        return false;
      }
      if (src_splitted_shape[i] != dst_splitted_shape[i]) {
        LOG(ERROR) << "Confliction between src and dst division, should never happen";
        return false;
      }
    } else {
      // broadcast 1 -> N
      src_splitted_shape[i].push_back(1);
      src_splitted_strides[i].push_back(0);
      dst_splitted_shape[i].push_back(-dim);
      dst_splitted_strides[i].push_back(0);
    }
  }

  // compute dst dom shape and transpose axes
  size_t op_idx = 1;
  std::vector<int64_t> coarse_transpose(ndim);
  if (op_idx >= shape_ops.size() || shape_ops[op_idx].type == arith::RESHAPE) {
    for (size_t i = 0; i < ndim; ++i) {
      coarse_transpose[i] = i;
    }
  } else {
    ICHECK_EQ(shape_ops[op_idx].values.size(), ndim);
    for (size_t i = 0; i < ndim; ++i) {
      coarse_transpose[i] = shape_ops[op_idx].values[i];
    }
    op_idx += 1;
  }
  if (op_idx < shape_ops.size() && shape_ops[op_idx].type == arith::RESHAPE) {
    const auto& cur_shape = shape_ops[op_idx].values;
    if (std::any_of(cur_shape.begin(), cur_shape.end(), [](int64_t x) { return x < 0; })) {
      if (detector.verbose) {
        LOG(ERROR) << "Can not handle broadcast after transpose";
      }
      return false;  // can not handle broadcast after transpose
    }
    op_idx += 1;
  }
  if (op_idx < shape_ops.size()) {
    if (detector.verbose) {
      LOG(ERROR) << "Too complex transform pattern";
    }
    return false;  // too complex transform pattern
  }

  // flatten
  std::vector<int64_t> flatten_dst_strides;
  size_t accum = 0;
  std::vector<std::pair<size_t, size_t>> transpose_range(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    size_t src_size = src_splitted_shape[i].size();
    for (size_t j = 0; j < src_size; ++j) {
      src_shape.push_back(src_splitted_shape[i][j]);
      src_strides.push_back(src_splitted_strides[i][j]);
    }
    transpose_range[i] = std::make_pair(accum, src_size);
    accum += src_size;

    size_t dst_coarse_idx = coarse_transpose[i];
    size_t dst_size = dst_splitted_shape[dst_coarse_idx].size();
    for (size_t j = 0; j < dst_size; ++j) {
      dst_shape.push_back(dst_splitted_shape[dst_coarse_idx][j]);
      dst_strides.push_back(dst_splitted_strides[dst_coarse_idx][j]);
    }
  }
  for (size_t i = 0; i < ndim; ++i) {
    size_t offset = transpose_range[coarse_transpose[i]].first;
    size_t extent = transpose_range[coarse_transpose[i]].second;
    for (size_t j = 0; j < extent; ++j) {
      transpose.push_back(offset + j);
    }
  }
  return true;
}

/**
 * Solve dma intrinsic shape attributes via affine analysis.
 */
static bool InferShapeAffine(const PrimExpr& src_index, const PrimExpr& dst_index,
                             const Map<Var, Range>& dom_map, const Array<Var>& vars,
                             int max_loop_sels, bool verbose, DMAShapeAttributes* result) {
  if (verbose) {
    LOG(INFO) << "Try affine dma shape inference: " << dst_index << " <- " << src_index;
  }
  // infer shape transformations for source index
  arith::IterTransformDetector detector(true, verbose);
  if (!detector.DetectReshapeTranspose(vars, {src_index}, dom_map)) {
    return false;
  }

  // extract iter strides for src
  std::unordered_map<size_t, int64_t> src_strides_map;
  std::vector<std::pair<PrimExpr, int>> parts;
  int offset;
  arith::ExtractSumFactors(src_index, true, &parts, &offset);
  for (const auto& p : parts) {
    int iter_id = detector.FindIterId(p.first);
    if (iter_id < 0) {
      LOG(ERROR) << "Can not find iteration " << p.first;
      return false;
    }
    if (verbose) {
      LOG(INFO) << "Bind source stride #" << iter_id << " " << detector.iter_bindings[iter_id]
                << " = " << p.second;
    }
    src_strides_map[iter_id] = p.second;
  }

  // ensure dst to be linear form and extract iter strides for dst
  std::unordered_map<size_t, int64_t> dst_strides_map;
  Array<PrimExpr> dst_coeffs = arith::DetectLinearEquation(dst_index, vars);
  if (dst_coeffs.size() != vars.size() + 1) {
    LOG(ERROR) << "DMA dest index is not in linear form: " << dst_index;
    return false;
  }
  for (size_t i = 0; i < vars.size(); ++i) {
    int iter_id = detector.FindIterId(vars[i]);
    if (iter_id < 0) {
      iter_id = detector.AddNewIter(vars[i], 1);
    }
    if (const IntImmNode* imm = dst_coeffs[i].as<IntImmNode>()) {
      if (verbose) {
        LOG(INFO) << "Bind dest stride #" << iter_id << ":" << detector.iter_bindings[iter_id]
                  << " = " << imm->value;
      }
      dst_strides_map[iter_id] = imm->value;
    } else {
      LOG(ERROR) << i << "th coeff of DMA dest index is not constant: " << dst_index;
      return false;
    }
  }

  std::vector<int64_t> src_reshape;
  std::vector<int64_t> src_strides;
  std::vector<int64_t> transpose;
  std::vector<int64_t> dst_reshape;
  std::vector<int64_t> dst_strides;
  if (!DoInferShapeDesc(detector, src_strides_map, dst_strides_map, &src_reshape, &src_strides,
                        &transpose, &dst_reshape, &dst_strides))
    return false;

  if (verbose) {
    std::stringstream ss;
    for (auto x : src_reshape) ss << x << ",";
    LOG(INFO) << "Source shape = [" << ss.str() << "]";
    ss.str("");
    for (auto x : src_strides) ss << x << ",";
    LOG(INFO) << "Source strides = [" << ss.str() << "]";
    ss.str("");
    for (auto x : transpose) ss << x << ",";
    LOG(INFO) << "Transpose = [" << ss.str() << "]";
    ss.str("");
    for (auto x : dst_reshape) ss << x << ",";
    LOG(INFO) << "Dest shape = [" << ss.str() << "]";
    ss.str("");
    for (auto x : dst_strides) ss << x << ",";
    LOG(INFO) << "Dest strides = [" << ss.str() << "]";
  }

  // ensure stride=1 at innermost dimension
  ICHECK(!src_strides.empty() && !dst_strides.empty());
  if (src_strides.back() != 1 || dst_strides.back() != 1) {
    src_reshape.push_back(1);
    src_strides.push_back(1);
    transpose.push_back(transpose.size());
    dst_reshape.push_back(1);
    dst_strides.push_back(1);
  }

  // ensure 4D iteration
  size_t ndim = src_reshape.size();
  if (ndim > 4) {
    if (verbose) LOG(ERROR) << "Do not support >4 transformation dims: " << ndim;
    return false;
  }
  size_t base = 4 - ndim;
  for (size_t i = 0; i < base; ++i) {
    result->loop_num[i] = 1;
    result->loop_src_index[i] = i;
    if (i < 3) {
      result->loop_src_stride[i] = src_strides[0] * src_reshape[0];
      result->loop_dst_stride[i] = dst_strides[0] * dst_reshape[0];
    }
  }
  for (size_t i = base; i < 4; ++i) {
    result->loop_num[i] = dst_reshape[i - base];
    result->loop_src_index[i] = transpose[i - base] + base;
    if (i < 3) {
      result->loop_src_stride[i] = src_strides[i - base];
      result->loop_dst_stride[i] = dst_strides[i - base];
    }
  }
  return true;
}

/**
 * Solve dma intrinsic shape attributes.
 */
static bool InferDMAShapeAttrs(const PrimExpr& src_index, const PrimExpr& dst_index,
                               const Map<Var, Range>& dom_map, const Array<Var>& vars,
                               int max_loop_sels, bool verbose, DMAShapeAttributes* result) {
  if (InferShapeSimple(src_index, dst_index, dom_map, vars, max_loop_sels, verbose, result)) {
    return true;
  } else if (InferShapeAffine(src_index, dst_index, dom_map, vars, max_loop_sels, verbose,
                              result)) {
    return true;
  } else if (InferShapeAffine(dst_index, src_index, dom_map, vars, max_loop_sels, verbose,
                              result)) {
    DMAShapeAttributes reverse;
    for (size_t i = 0; i < 4; ++i) {
      size_t dst_idx = result->loop_src_index[i];
      reverse.loop_num[dst_idx] = result->loop_num[i];
      reverse.loop_src_index[dst_idx] = i;
      reverse.loop_src_stride[i] = result->loop_dst_stride[i];
      reverse.loop_dst_stride[dst_idx] = result->loop_src_stride[i];
    }
    std::swap(reverse, *result);
    return true;
  } else {
    return false;
  }
}

Stmt DMAIntrinRewriter::RewriteDMA(const Stmt& body, const std::string& intrin_hint_name) {
  // analyze load/store pattern
  if (verbose_) {
    LOG(INFO) << "Try rewrite dma scope: \n" << body;
  }
  Stmt target_stmt = CollectDmaScope(body);
  const StoreNode* store = target_stmt.as<StoreNode>();
  ICHECK(store) << "Unsupported dma stmt " << target_stmt;
  const LoadNode* load = store->value.as<LoadNode>();
  PrimExpr cond;
  if (load == nullptr) {
    if (intrin_hint_name == "idma") {
      // for idma, the load pattern could be padded
      const CallNode* ifthen = store->value.as<CallNode>();
      ICHECK(ifthen && ifthen->op.same_as(builtin::if_then_else()))
          << "Unsupported idma stmt " << target_stmt;
      cond = ifthen->args[0];
      load = ifthen->args[1].as<LoadNode>();
      auto pad_value = ifthen->args[2].as<IntImmNode>();
      ICHECK(pad_value) << "Idma do not support non-const padding value: " << ifthen->args[2];
    } else if (intrin_hint_name == "odma") {
      // for odma, the load pattern could be fused with post computations
      PostOrderVisit(store->value, [&load, this](const ObjectRef& obj) {
        if (const LoadNode* cur_load = obj.as<LoadNode>()) {
          StorageScope scope = GetStorageScope(cur_load->buffer_var);
          if (scope.rank == StorageRank::kCUBE) {
            load = cur_load;
          }
        }
      });
    }
  }
  ICHECK(load) << "Unsupported dma stmt, fail to fetch load pattern: \n" << target_stmt;

  // get base + offset division for src
  auto p_src = GetIndiceDivision(dom_analyzer_.Simplify(load->index), loop_vars_);
  PrimExpr src_base = dom_analyzer_.Simplify(p_src.first);
  PrimExpr src_offset = dom_analyzer_.Simplify(p_src.second);

  // get base + offset division for dst
  auto p_dst = GetIndiceDivision(dom_analyzer_.Simplify(store->index), loop_vars_);
  PrimExpr dst_base = dom_analyzer_.Simplify(p_dst.first);
  PrimExpr dst_offset = dom_analyzer_.Simplify(p_dst.second);

  // detect dma type
  StorageScope src_scope = GetStorageScope(load->buffer_var);
  StorageScope dst_scope = GetStorageScope(store->buffer_var);
  DataType src_dtype = GetBufferElementType(load->buffer_var);
  DataType dst_dtype = GetBufferElementType(store->buffer_var);

  if (src_scope.rank == StorageRank::kGlobal && dst_scope.rank == StorageRank::kDM) {
    if (intrin_hint_name == "ewdma") {
      return CreateEwdmaLoad(load->buffer_var, src_dtype, src_base, src_offset, store->buffer_var,
                             dst_dtype, dst_base, dst_offset);
    } else {
      return CreateEidmaLoad(load->buffer_var, src_dtype, src_base, src_offset, store->buffer_var,
                             dst_dtype, dst_base, dst_offset);
    }
  } else if (src_scope.rank == StorageRank::kDM && dst_scope.rank == StorageRank::kGlobal) {
    return CreateEodmaStore(load->buffer_var, src_dtype, src_base, src_offset, store->buffer_var,
                            dst_dtype, dst_base, dst_offset);
  } else if (src_scope.rank == StorageRank::kDM && dst_scope.rank == StorageRank::kVM) {
    return CreateVidmaLoad(load->buffer_var, src_dtype, src_base, src_offset, store->buffer_var,
                           dst_dtype, dst_base, dst_offset);
  } else if (src_scope.rank == StorageRank::kVM && dst_scope.rank == StorageRank::kDM) {
    return CreateVodmaStore(load->buffer_var, src_dtype, src_base, src_offset, store->buffer_var,
                            dst_dtype, dst_base, dst_offset);
  } else if (src_scope.rank == StorageRank::kDM && dst_scope.rank == StorageRank::kIOBUF) {
    return CreateIdmaLoad(load->buffer_var, src_dtype, src_base, src_offset, store->buffer_var,
                          dst_dtype, dst_base, dst_offset, cond);
  } else if (src_scope.rank == StorageRank::kDM && dst_scope.rank == StorageRank::kWBUF) {
    return CreateWdmaLoad(load->buffer_var, src_dtype, src_base, src_offset, store->buffer_var,
                          dst_dtype, dst_base, dst_offset);
  } else if (src_scope.rank == StorageRank::kDM && dst_scope.rank == StorageRank::kBBUF) {
    return CreateBdmaLoad(load->buffer_var, src_dtype, src_base, src_offset, store->buffer_var,
                          dst_dtype, dst_base, dst_offset);
  } else if (src_scope.rank == StorageRank::kCUBE && dst_scope.rank == StorageRank::kDM) {
    return CreateOdmaStore(load->buffer_var, src_dtype, src_base, src_offset, store->buffer_var,
                           dst_dtype, dst_base, dst_offset);
  } else {
    LOG(FATAL) << "No DMA intrinsic supported from " << src_scope.to_string() << " to "
               << dst_scope.to_string();
  }
  return target_stmt;
}

PrimExpr DMAIntrinRewriter::CreateAccessPtr(const Var& var, const DataType& dtype,
                                            const PrimExpr& offset, const PrimExpr& index,
                                            const std::string& mask) {
  arith::IntSet intset = arith::EvalSet(index, dom_intsets_);
  PrimExpr begin = dom_analyzer_.Simplify(offset + intset.min());
  PrimExpr extent = dom_analyzer_.Simplify(intset.max() - intset.min() + 1);
  Array<PrimExpr> args = {tir::TypeAnnotation(dtype), var, begin, extent, StringImm(mask)};
  return std::move(Call(DataType::Handle(), builtin::tvm_access_ptr(), args));
}

static int GetDmaDatatypeCode(const DataType& dtype) {
  if (dtype == DataType::Int(8)) {
    return 0;
  } else if (dtype == DataType::UInt(8)) {
    return 1;
  } else if (dtype == DataType::Float(16)) {
    return 2;
  } else if (dtype == DataType::Float(32)) {
    return 3;
  } else if (dtype == DataType::Int(32)) {
    return 4;
  } else {
    LOG(FATAL) << "Unsupported datatype " << dtype;
    return -1;
  }
}

Stmt DMAIntrinRewriter::CreateEidmaLoad(const Var& src_buffer, const DataType& src_dtype,
                                        const PrimExpr& src_base, const PrimExpr& src_offset,
                                        const Var& dst_buffer, const DataType& dst_dtype,
                                        const PrimExpr& dst_base, const PrimExpr& dst_offset) {
  ICHECK_EQ(src_dtype, dst_dtype) << "Source and dst datatype do not match: " << src_dtype << ", "
                                  << dst_dtype;
  size_t dtype_bytes = src_dtype.bytes();
  PrimExpr src_access = CreateAccessPtr(src_buffer, src_dtype, src_offset, src_base, "r");
  PrimExpr dst_access = CreateAccessPtr(dst_buffer, dst_dtype, dst_offset, dst_base, "w");
  Call intrin = Call(DataType::Void(), edgex::builtin::nnp_eidma_load(),
                     {StringImm(DLDataType2String(src_dtype)), dst_access, src_access});

  DMAShapeAttributes shape_attrs;
  bool status =
      InferDMAShapeAttrs(src_base, dst_base, dom_map_, loop_vars_, 4, verbose_, &shape_attrs);
  ICHECK(status) << "Fail to infer eidma shape attributes for " << src_base << " <- " << dst_base;

  auto intrin_ptr = const_cast<CallNode*>(intrin.get());
  SetExtraDmaAttrs(intrin_ptr);
  NNPAddArg(intrin_ptr, "ei_start_addr_in_en", 1);
  NNPAddArg(intrin_ptr, "ei_start_addr_out_en", 1);
  NNPAddArg(intrin_ptr, "ei_first_state_en", 1);
  NNPAddArg(intrin_ptr, "ei_state_num", 1);
  NNPAddArg(intrin_ptr, "ei_dtype", GetDmaDatatypeCode(src_dtype));
  NNPAddArg(intrin_ptr, "ei_mode", EIDMA_TRANSPOSE_MODE);
  NNPAddArg(intrin_ptr, "ei_j0_loop_num", shape_attrs.loop_num[0]);
  NNPAddArg(intrin_ptr, "ei_j1_loop_num", shape_attrs.loop_num[1]);
  NNPAddArg(intrin_ptr, "ei_j2_loop_num", shape_attrs.loop_num[2]);
  NNPAddArg(intrin_ptr, "ei_j3_loop_num", shape_attrs.loop_num[3]);
  NNPAddArg(intrin_ptr, "ei_j0_loop_sel", 3 - shape_attrs.loop_src_index[0]);
  NNPAddArg(intrin_ptr, "ei_j1_loop_sel", 3 - shape_attrs.loop_src_index[1]);
  NNPAddArg(intrin_ptr, "ei_j2_loop_sel", 3 - shape_attrs.loop_src_index[2]);
  NNPAddArg(intrin_ptr, "ei_j3_loop_sel", 3 - shape_attrs.loop_src_index[3]);
  NNPAddArg(intrin_ptr, "ei_j0_stridein", shape_attrs.loop_src_stride[0] * dtype_bytes);
  NNPAddArg(intrin_ptr, "ei_j1_stridein", shape_attrs.loop_src_stride[1] * dtype_bytes);
  NNPAddArg(intrin_ptr, "ei_j2_stridein", shape_attrs.loop_src_stride[2] * dtype_bytes);
  NNPAddArg(intrin_ptr, "ei_j0_strideout", shape_attrs.loop_dst_stride[0] * dtype_bytes);
  NNPAddArg(intrin_ptr, "ei_j1_strideout", shape_attrs.loop_dst_stride[1] * dtype_bytes);
  NNPAddArg(intrin_ptr, "ei_j2_strideout", shape_attrs.loop_dst_stride[2] * dtype_bytes);
  return Evaluate(std::move(intrin));
}

Stmt DMAIntrinRewriter::CreateEwdmaLoad(const Var& src_buffer, const DataType& src_dtype,
                                        const PrimExpr& src_base, const PrimExpr& src_offset,
                                        const Var& dst_buffer, const DataType& dst_dtype,
                                        const PrimExpr& dst_base, const PrimExpr& dst_offset) {
  ICHECK_EQ(src_dtype, dst_dtype) << "Source and dst datatype do not match: " << src_dtype << ", "
                                  << dst_dtype;
  size_t dtype_bytes = src_dtype.bytes();
  PrimExpr src_access = CreateAccessPtr(src_buffer, src_dtype, src_offset, src_base, "r");
  PrimExpr dst_access = CreateAccessPtr(dst_buffer, dst_dtype, dst_offset, dst_base, "w");
  Call intrin = Call(DataType::Void(), edgex::builtin::nnp_ewdma_load(),
                     {StringImm(DLDataType2String(src_dtype)), dst_access, src_access});

  DMAShapeAttributes shape_attrs;
  bool status =
      InferDMAShapeAttrs(src_base, dst_base, dom_map_, loop_vars_, 3, verbose_, &shape_attrs);
  ICHECK(status) << "Fail to infer ewdma shape attributes for " << src_base << " <- " << dst_base;
  CHECK(shape_attrs.loop_num[0] == 1 && shape_attrs.loop_src_index[0] == 0 &&
        shape_attrs.loop_src_index[1] == 1 && shape_attrs.loop_src_index[2] == 2 &&
        shape_attrs.loop_src_index[3] == 3 &&
        shape_attrs.loop_src_stride[0] == shape_attrs.loop_src_stride[1] &&
        shape_attrs.loop_dst_stride[0] == shape_attrs.loop_dst_stride[1])
      << "Ewdma can not support complex dma pattern";

  auto intrin_ptr = const_cast<CallNode*>(intrin.get());
  SetExtraDmaAttrs(intrin_ptr);
  NNPAddArg(intrin_ptr, "ew_start_addr_in_en", 1);
  NNPAddArg(intrin_ptr, "ew_start_addr_out_en", 1);
  NNPAddArg(intrin_ptr, "ew_first_state_en", 1);
  NNPAddArg(intrin_ptr, "ew_state_num", 1);
  NNPAddArg(intrin_ptr, "ew_dtype", GetDmaDatatypeCode(src_dtype));
  NNPAddArg(intrin_ptr, "ew_mode", 0);
  NNPAddArg(intrin_ptr, "ew_j1_loop_num", shape_attrs.loop_num[1]);
  NNPAddArg(intrin_ptr, "ew_j2_loop_num", shape_attrs.loop_num[2]);
  NNPAddArg(intrin_ptr, "ew_j3_loop_num", shape_attrs.loop_num[3]);
  NNPAddArg(intrin_ptr, "ew_j1_stridein", shape_attrs.loop_src_stride[1] * dtype_bytes);
  NNPAddArg(intrin_ptr, "ew_j2_stridein", shape_attrs.loop_src_stride[2] * dtype_bytes);
  NNPAddArg(intrin_ptr, "ew_j1_strideout", shape_attrs.loop_dst_stride[1] * dtype_bytes);
  NNPAddArg(intrin_ptr, "ew_j2_strideout", shape_attrs.loop_dst_stride[2] * dtype_bytes);
  return Evaluate(std::move(intrin));
}

Stmt DMAIntrinRewriter::CreateEodmaStore(const Var& src_buffer, const DataType& src_dtype,
                                         const PrimExpr& src_base, const PrimExpr& src_offset,
                                         const Var& dst_buffer, const DataType& dst_dtype,
                                         const PrimExpr& dst_base, const PrimExpr& dst_offset) {
  ICHECK_EQ(src_dtype, dst_dtype) << "Source and dst datatype do not match: " << src_dtype << ", "
                                  << dst_dtype;
  size_t dtype_bytes = src_dtype.bytes();
  PrimExpr src_access = CreateAccessPtr(src_buffer, src_dtype, src_offset, src_base, "r");
  PrimExpr dst_access = CreateAccessPtr(dst_buffer, dst_dtype, dst_offset, dst_base, "w");
  Call intrin = Call(DataType::Void(), edgex::builtin::nnp_eodma_store(),
                     {StringImm(DLDataType2String(src_dtype)), dst_access, src_access});

  DMAShapeAttributes shape_attrs;
  bool status =
      InferDMAShapeAttrs(src_base, dst_base, dom_map_, loop_vars_, 4, verbose_, &shape_attrs);
  ICHECK(status) << "Fail to infer eodma shape attributes for " << src_base << " <- " << dst_base;

  auto intrin_ptr = const_cast<CallNode*>(intrin.get());
  SetExtraDmaAttrs(intrin_ptr);
  NNPAddArg(intrin_ptr, "eo_start_addr_in_en", 1);
  NNPAddArg(intrin_ptr, "eo_start_addr_out_en", 1);
  NNPAddArg(intrin_ptr, "eo_first_state_en", 1);
  NNPAddArg(intrin_ptr, "eo_state_num", 3);
  NNPAddArg(intrin_ptr, "eo_dtype", GetDmaDatatypeCode(src_dtype));
  NNPAddArg(intrin_ptr, "eo_mode", EODMA_TRANSPOSE_MODE);
  NNPAddArg(intrin_ptr, "eo_j0_loop_num", shape_attrs.loop_num[0]);
  NNPAddArg(intrin_ptr, "eo_j1_loop_num", shape_attrs.loop_num[1]);
  NNPAddArg(intrin_ptr, "eo_j2_loop_num", shape_attrs.loop_num[2]);
  NNPAddArg(intrin_ptr, "eo_j3_loop_num", shape_attrs.loop_num[3]);
  NNPAddArg(intrin_ptr, "eo_j0_loop_sel", 3 - shape_attrs.loop_src_index[0]);
  NNPAddArg(intrin_ptr, "eo_j1_loop_sel", 3 - shape_attrs.loop_src_index[1]);
  NNPAddArg(intrin_ptr, "eo_j2_loop_sel", 3 - shape_attrs.loop_src_index[2]);
  NNPAddArg(intrin_ptr, "eo_j3_loop_sel", 3 - shape_attrs.loop_src_index[3]);
  NNPAddArg(intrin_ptr, "eo_stride_in_j0", shape_attrs.loop_src_stride[0] * dtype_bytes);
  NNPAddArg(intrin_ptr, "eo_stride_in_j1", shape_attrs.loop_src_stride[1] * dtype_bytes);
  NNPAddArg(intrin_ptr, "eo_stride_in_j2", shape_attrs.loop_src_stride[2] * dtype_bytes);
  NNPAddArg(intrin_ptr, "eo_j0_strideout", shape_attrs.loop_dst_stride[0] * dtype_bytes);
  NNPAddArg(intrin_ptr, "eo_j1_strideout", shape_attrs.loop_dst_stride[1] * dtype_bytes);
  NNPAddArg(intrin_ptr, "eo_j2_strideout", shape_attrs.loop_dst_stride[2] * dtype_bytes);
  return Evaluate(std::move(intrin));
}

Stmt DMAIntrinRewriter::CreateVidmaLoad(const Var& src_buffer, const DataType& src_dtype,
                                        const PrimExpr& src_base, const PrimExpr& src_offset,
                                        const Var& dst_buffer, const DataType& dst_dtype,
                                        const PrimExpr& dst_base, const PrimExpr& dst_offset) {
  ICHECK_EQ(src_dtype, dst_dtype) << "Source and dst datatype do not match: " << src_dtype << ", "
                                  << dst_dtype;
  size_t dtype_bytes = src_dtype.bytes();
  PrimExpr src_access = CreateAccessPtr(src_buffer, src_dtype, src_offset, src_base, "r");
  PrimExpr dst_access = CreateAccessPtr(dst_buffer, dst_dtype, dst_offset, dst_base, "w");
  Call intrin = Call(DataType::Void(), edgex::builtin::nnp_vidma_load(),
                     {StringImm(DLDataType2String(src_dtype)), dst_access, src_access});

  DMAShapeAttributes shape_attrs;
  bool status =
      InferDMAShapeAttrs(src_base, dst_base, dom_map_, loop_vars_, 4, verbose_, &shape_attrs);
  ICHECK(status) << "Fail to infer vidma shape attributes for " << src_base << " <- " << dst_base;

  auto intrin_ptr = const_cast<CallNode*>(intrin.get());
  SetExtraDmaAttrs(intrin_ptr);
  NNPAddArg(intrin_ptr, "start_addr_in_en_vidma", 1);
  NNPAddArg(intrin_ptr, "start_addr_out_en_vidma", 1);
  NNPAddArg(intrin_ptr, "cb_buf_vm_vidma", 1);
  NNPAddArg(intrin_ptr, "cb_buf_dm_vidma", 1);
  NNPAddArg(intrin_ptr, "crop_en_vidma", 1);
  NNPAddArg(intrin_ptr, "dtype_vidma", GetDmaDatatypeCode(src_dtype));
  NNPAddArg(intrin_ptr, "j0_loop_num_vidma", shape_attrs.loop_num[0]);
  NNPAddArg(intrin_ptr, "j1_loop_num_vidma", shape_attrs.loop_num[1]);
  NNPAddArg(intrin_ptr, "j2_loop_num_vidma", shape_attrs.loop_num[2]);
  NNPAddArg(intrin_ptr, "j3_loop_num_vidma", shape_attrs.loop_num[3]);
  NNPAddArg(intrin_ptr, "j0_loop_sel_vidma", 3 - shape_attrs.loop_src_index[0]);
  NNPAddArg(intrin_ptr, "j1_loop_sel_vidma", 3 - shape_attrs.loop_src_index[1]);
  NNPAddArg(intrin_ptr, "j2_loop_sel_vidma", 3 - shape_attrs.loop_src_index[2]);
  NNPAddArg(intrin_ptr, "j3_loop_sel_vidma", 3 - shape_attrs.loop_src_index[3]);
  NNPAddArg(intrin_ptr, "j0_stridein_vidma", shape_attrs.loop_src_stride[0] * dtype_bytes);
  NNPAddArg(intrin_ptr, "j1_stridein_vidma", shape_attrs.loop_src_stride[1] * dtype_bytes);
  NNPAddArg(intrin_ptr, "j2_stridein_vidma", shape_attrs.loop_src_stride[2] * dtype_bytes);
  NNPAddArg(intrin_ptr, "j0_strideout_vidma", shape_attrs.loop_dst_stride[0] * dtype_bytes);
  NNPAddArg(intrin_ptr, "j1_strideout_vidma", shape_attrs.loop_dst_stride[1] * dtype_bytes);
  NNPAddArg(intrin_ptr, "j2_strideout_vidma", shape_attrs.loop_dst_stride[2] * dtype_bytes);
  return Evaluate(std::move(intrin));
}

Stmt DMAIntrinRewriter::CreateVodmaStore(const Var& src_buffer, const DataType& src_dtype,
                                         const PrimExpr& src_base, const PrimExpr& src_offset,
                                         const Var& dst_buffer, const DataType& dst_dtype,
                                         const PrimExpr& dst_base, const PrimExpr& dst_offset) {
  ICHECK_EQ(src_dtype, dst_dtype) << "Source and dst datatype do not match: " << src_dtype << ", "
                                  << dst_dtype;
  size_t dtype_bytes = src_dtype.bytes();
  PrimExpr src_access = CreateAccessPtr(src_buffer, src_dtype, src_offset, src_base, "r");
  PrimExpr dst_access = CreateAccessPtr(dst_buffer, dst_dtype, dst_offset, dst_base, "w");
  Call intrin = Call(DataType::Void(), edgex::builtin::nnp_vodma_store(),
                     {StringImm(DLDataType2String(src_dtype)), dst_access, src_access});

  DMAShapeAttributes shape_attrs;
  bool status =
      InferDMAShapeAttrs(src_base, dst_base, dom_map_, loop_vars_, 4, verbose_, &shape_attrs);
  ICHECK(status) << "Fail to infer vodma shape attributes for " << src_base << " <- " << dst_base;

  auto intrin_ptr = const_cast<CallNode*>(intrin.get());
  SetExtraDmaAttrs(intrin_ptr);
  NNPAddArg(intrin_ptr, "start_addr_in_en_vodma", 1);
  NNPAddArg(intrin_ptr, "start_addr_out_en_vodma", 1);
  NNPAddArg(intrin_ptr, "cb_buf_vm_vodma", 1);
  NNPAddArg(intrin_ptr, "cb_buf_dm_vodma", 1);
  NNPAddArg(intrin_ptr, "crop_en_vodma", 1);
  NNPAddArg(intrin_ptr, "dtype_vodma", GetDmaDatatypeCode(src_dtype));
  NNPAddArg(intrin_ptr, "j0_loop_num_vodma", shape_attrs.loop_num[0]);
  NNPAddArg(intrin_ptr, "j1_loop_num_vodma", shape_attrs.loop_num[1]);
  NNPAddArg(intrin_ptr, "j2_loop_num_vodma", shape_attrs.loop_num[2]);
  NNPAddArg(intrin_ptr, "j3_loop_num_vodma", shape_attrs.loop_num[3]);
  NNPAddArg(intrin_ptr, "j0_loop_sel_vodma", 3 - shape_attrs.loop_src_index[0]);
  NNPAddArg(intrin_ptr, "j1_loop_sel_vodma", 3 - shape_attrs.loop_src_index[1]);
  NNPAddArg(intrin_ptr, "j2_loop_sel_vodma", 3 - shape_attrs.loop_src_index[2]);
  NNPAddArg(intrin_ptr, "j3_loop_sel_vodma", 3 - shape_attrs.loop_src_index[3]);
  NNPAddArg(intrin_ptr, "j0_stridein_vodma", shape_attrs.loop_src_stride[0] * dtype_bytes);
  NNPAddArg(intrin_ptr, "j1_stridein_vodma", shape_attrs.loop_src_stride[1] * dtype_bytes);
  NNPAddArg(intrin_ptr, "j2_stridein_vodma", shape_attrs.loop_src_stride[2] * dtype_bytes);
  NNPAddArg(intrin_ptr, "j0_strideout_vodma", shape_attrs.loop_dst_stride[0] * dtype_bytes);
  NNPAddArg(intrin_ptr, "j1_strideout_vodma", shape_attrs.loop_dst_stride[1] * dtype_bytes);
  NNPAddArg(intrin_ptr, "j2_strideout_vodma", shape_attrs.loop_dst_stride[2] * dtype_bytes);
  return Evaluate(std::move(intrin));
}

Stmt DMAIntrinRewriter::CreateIdmaLoad(const Var& src_buffer, const DataType& src_dtype,
                                       const PrimExpr& src_base, const PrimExpr& src_offset,
                                       const Var& dst_buffer, const DataType& dst_dtype,
                                       const PrimExpr& dst_base, const PrimExpr& dst_offset,
                                       const PrimExpr& cond) {
  ICHECK_EQ(src_dtype, dst_dtype) << "Source and dst datatype do not match: " << src_dtype << ", "
                                  << dst_dtype;
  PrimExpr src_access;
  if (cond.defined()) {  // padding exists
    Map<Var, Range> src_dom_map(dom_map_.begin(), dom_map_.end());
    UpdateConditionalDomain(cond, &src_dom_map);
    arith::Analyzer src_dom_analyzer;
    Map<Var, arith::IntSet> src_dom_intsets;
    for (const auto& p : src_dom_map) {
      src_dom_intsets.Set(p.first, arith::IntSet::FromRange(p.second));
    }
    src_dom_analyzer.Bind(src_dom_map);
    arith::IntSet intset = arith::EvalSet(src_base, src_dom_intsets);
    PrimExpr extent = src_dom_analyzer.Simplify(intset.max() - intset.min() + 1);
    PrimExpr begin = src_dom_analyzer.Simplify(src_offset + intset.min());
    Array<PrimExpr> args = {tir::TypeAnnotation(src_dtype), src_buffer, begin, extent,
                            StringImm("r")};
    src_access = std::move(Call(DataType::Handle(), builtin::tvm_access_ptr(), args));
  } else {
    src_access = CreateAccessPtr(src_buffer, src_dtype, src_offset, src_base, "r");
  }
  PrimExpr dst_access = CreateAccessPtr(dst_buffer, dst_dtype, dst_offset, dst_base, "w");
  Call intrin = Call(DataType::Void(), edgex::builtin::nnp_idma_load(),
                     {StringImm(DLDataType2String(src_dtype)), dst_access, src_access});
  auto intrin_ptr = const_cast<CallNode*>(intrin.get());
  SetExtraDmaAttrs(intrin_ptr);
  return Evaluate(std::move(intrin));
}

Stmt DMAIntrinRewriter::CreateWdmaLoad(const Var& src_buffer, const DataType& src_dtype,
                                       const PrimExpr& src_base, const PrimExpr& src_offset,
                                       const Var& dst_buffer, const DataType& dst_dtype,
                                       const PrimExpr& dst_base, const PrimExpr& dst_offset) {
  ICHECK_EQ(src_dtype, dst_dtype) << "Source and dst datatype do not match: " << src_dtype << ", "
                                  << dst_dtype;
  PrimExpr src_access = CreateAccessPtr(src_buffer, src_dtype, src_offset, src_base, "r");
  PrimExpr dst_access = CreateAccessPtr(dst_buffer, dst_dtype, dst_offset, dst_base, "w");
  Call intrin = Call(DataType::Void(), edgex::builtin::nnp_wdma_load(),
                     {StringImm(DLDataType2String(src_dtype)), dst_access, src_access});
  auto intrin_ptr = const_cast<CallNode*>(intrin.get());
  SetExtraDmaAttrs(intrin_ptr);
  return Evaluate(std::move(intrin));
}

Stmt DMAIntrinRewriter::CreateBdmaLoad(const Var& src_buffer, const DataType& src_dtype,
                                       const PrimExpr& src_base, const PrimExpr& src_offset,
                                       const Var& dst_buffer, const DataType& dst_dtype,
                                       const PrimExpr& dst_base, const PrimExpr& dst_offset) {
  ICHECK_EQ(src_dtype, dst_dtype) << "Source and dst datatype do not match: " << src_dtype << ", "
                                  << dst_dtype;
  PrimExpr src_access = CreateAccessPtr(src_buffer, src_dtype, src_offset, src_base, "r");
  PrimExpr dst_access = CreateAccessPtr(dst_buffer, dst_dtype, dst_offset, dst_base, "w");
  Call intrin = Call(DataType::Void(), edgex::builtin::nnp_bdma_load(),
                     {StringImm(DLDataType2String(src_dtype)), dst_access, src_access});
  auto intrin_ptr = const_cast<CallNode*>(intrin.get());
  SetExtraDmaAttrs(intrin_ptr);
  return Evaluate(std::move(intrin));
}

Stmt DMAIntrinRewriter::CreateOdmaStore(const Var& src_buffer, const DataType& src_dtype,
                                        const PrimExpr& src_base, const PrimExpr& src_offset,
                                        const Var& dst_buffer, const DataType& dst_dtype,
                                        const PrimExpr& dst_base, const PrimExpr& dst_offset) {
  PrimExpr src_access = CreateAccessPtr(src_buffer, src_dtype, src_offset, src_base, "r");
  PrimExpr dst_access = CreateAccessPtr(dst_buffer, dst_dtype, dst_offset, dst_base, "w");
  Call intrin = Call(DataType::Void(), edgex::builtin::nnp_odma_store(),
                     {StringImm(DLDataType2String(src_dtype)), dst_access, src_access});
  auto intrin_ptr = const_cast<CallNode*>(intrin.get());
  SetExtraDmaAttrs(intrin_ptr);
  return Evaluate(std::move(intrin));
}

Stmt InjectDmaIntrin(Stmt stmt, bool verbose) {
  return DMAIntrinScopeRewriter({}, verbose)(std::move(stmt));
}

Stmt InjectDmaIntrin(PrimFunc f, bool verbose) {
  DMAIntrinScopeRewriter rewriter(f->buffer_map, verbose);
  return rewriter(std::move(f->body));
}

namespace transform {

Pass InjectDmaIntrin() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    bool verbose = ctx->GetConfig<Bool>("tir.edgex.InjectDmaIntrin.verbose", Bool(false)).value();
    auto* n = f.CopyOnWrite();
    n->body = InjectDmaIntrin(f, verbose);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.edgex.InjectDmaIntrin", {});
}

TVM_REGISTER_PASS_CONFIG_OPTION("tir.edgex.InjectDmaIntrin.verbose", Bool);
TVM_REGISTER_GLOBAL("tir.edgex.transform.InjectDmaIntrin").set_body_typed(InjectDmaIntrin);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
