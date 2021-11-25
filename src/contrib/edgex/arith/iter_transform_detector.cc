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
 * \file iter_transform_detector.h
 */
#include "./iter_transform_detector.h"

#include <tvm/arith/analyzer.h>

#include <numeric>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace arith {

bool IterOperation::Depends(const IterOperation& other) const {
  if (type == FUSE) {
    if (other.type == FUSE) {
      return fuse_src().first == other.fuse_dst() || fuse_src().second == other.fuse_dst();
    } else {
      return fuse_src().first == other.split_dst().first ||
             fuse_src().second == other.split_dst().first ||
             fuse_src().first == other.split_dst().second ||
             fuse_src().second == other.split_dst().second;
    }
  } else {
    if (other.type == FUSE) {
      return split_src() == other.fuse_dst();
    } else {
      return split_src() == other.split_dst().first || split_src() == other.split_dst().second;
    }
  }
}

/**
 *! \brief Extract factors of summation a*x1 + b*x2 +... + k
 */
void ExtractSumFactors(const PrimExpr& e, bool sign, std::vector<std::pair<PrimExpr, int>>* factors,
                       int* constant) {
  const auto get_signed = [sign](const PrimExpr& e) {
    arith::Analyzer analyzer;
    return sign ? e : analyzer.Simplify(tvm::neg(e));
  };
  if (const AddNode* add = e.as<AddNode>()) {
    ExtractSumFactors(add->a, sign, factors, constant);
    ExtractSumFactors(add->b, sign, factors, constant);
  } else if (const SubNode* sub = e.as<SubNode>()) {
    ExtractSumFactors(sub->a, sign, factors, constant);
    ExtractSumFactors(sub->b, !sign, factors, constant);
  } else if (const MulNode* mul = e.as<MulNode>()) {
    if (mul->a->IsInstance<IntImmNode>()) {
      factors->push_back({get_signed(mul->b), Downcast<IntImm>(mul->a)->value});
    } else if (mul->b->IsInstance<IntImmNode>()) {
      factors->push_back({get_signed(mul->a), Downcast<IntImm>(mul->b)->value});
    } else {
      factors->push_back({get_signed(e), 1});
    }
  } else if (e->IsInstance<IntImmNode>()) {
    *constant = *constant + (sign ? 1 : -1) * Downcast<IntImm>(e)->value;
  } else {
    factors->push_back({get_signed(e), 1});
  }
}

size_t IterTransformDetector::AddNewIter(const PrimExpr& binding, int64_t extent) {
  size_t idx = iter_bindings.size();
  iter_bindings.push_back(binding);
  iter_extents.push_back(extent);
  iter_producer_mapping.push_back(-1);
  iter_consumer_mapping.push_back(-1);
  return idx;
}

void IterTransformDetector::AddIterOperation(const IterOperation& op) {
  int64_t op_idx = static_cast<int64_t>(op_seq.size());
  op_seq.push_back(op);
  if (op.type == FUSE) {
    std::pair<size_t, size_t> p = op.fuse_src();
    iter_producer_mapping[op.fuse_dst()] = op_idx;
    iter_consumer_mapping[p.first] = op_idx;
    iter_consumer_mapping[p.second] = op_idx;
  } else {
    std::pair<size_t, size_t> p = op.split_dst();
    iter_consumer_mapping[op.split_src()] = op_idx;
    iter_producer_mapping[p.first] = op_idx;
    iter_producer_mapping[p.second] = op_idx;
  }
}

int64_t IterTransformDetector::FindIterId(const PrimExpr& e) const {
  StructuralEqual structural_equal;
  for (size_t i = 0; i < iter_bindings.size(); ++i) {
    const PrimExpr& bind = iter_bindings[i];
    if (structural_equal(e, bind)) return i;
  }
  return -1;
}

void IterTransformDetector::UpdateIterOperation(size_t op_idx, const IterOperation& op) {
  op_seq[op_idx] = op;
  if (op.type == FUSE) {
    std::pair<size_t, size_t> p = op.fuse_src();
    iter_producer_mapping[op.fuse_dst()] = op_idx;
    iter_consumer_mapping[p.first] = op_idx;
    iter_consumer_mapping[p.second] = op_idx;
  } else {
    std::pair<size_t, size_t> p = op.split_dst();
    iter_consumer_mapping[op.split_src()] = op_idx;
    iter_producer_mapping[p.first] = op_idx;
    iter_producer_mapping[p.second] = op_idx;
  }
}

void IterTransformDetector::UpdateIterInfo(size_t idx, const PrimExpr& binding, int64_t extent) {
  iter_bindings[idx] = binding;
  iter_extents[idx] = extent;
}

std::string IterTransformDetector::FormatOp(const IterOperation& op) const {
  std::stringstream ss;
  if (op.type == FUSE) {
    size_t dst = op.fuse_dst();
    auto p = op.fuse_src();
    ss << "Fuse ({#" << p.first << ", " << iter_bindings[p.first] << ", " << iter_extents[p.first]
       << "}, "
       << "{#" << p.second << ", " << iter_bindings[p.second] << ", " << iter_extents[p.second]
       << "}) -> "
       << "{#" << dst << ", " << iter_bindings[dst] << ", " << iter_extents[dst] << "}";
  } else {
    size_t src = op.split_src();
    auto p = op.split_dst();
    ss << "Split {#" << src << ", " << iter_bindings[src] << ", " << iter_extents[src] << "} -> ("
       << "{#" << p.first << ", " << iter_bindings[p.first] << ", " << iter_extents[p.first]
       << "}, "
       << "{#" << p.second << ", " << iter_bindings[p.second] << ", " << iter_extents[p.second]
       << "})";
  }
  return ss.str();
}

std::string IterTransformDetector::FormatOp(const ShapeOperation& op) const {
  std::stringstream ss;
  if (op.type == RESHAPE) {
    ss << "Reshape [";
  } else {
    ss << "Transpose [";
  }
  for (size_t i = 0; i < op.values.size(); ++i) {
    ss << op.values[i];
    if (i < op.values.size() - 1) {
      ss << ", ";
    }
  }
  ss << "]";
  if (verbose) {
    ss << " [";
    for (size_t i = 0; i < op.iters.size(); ++i) {
      auto& ids = op.iters[i];
      ss << "[";
      for (size_t j = 0; j < ids.size(); ++j) {
        ss << "{#" << ids[j] << ", " << iter_bindings[ids[j]] << ", " << iter_extents[ids[j]]
           << "}";
        if (j < ids.size() - 1) {
          ss << ", ";
        }
      }
      ss << "]";
      if (i < op.iters.size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
  }
  return ss.str();
}

void IterTransformDetector::ShowOps() const {
  std::stringstream ss;
  for (auto it = op_seq.rbegin(); it != op_seq.rend(); ++it) {
    ss << FormatOp(*it) << "\n";
  }
  LOG(INFO) << "Current iteration ops info:\n" << ss.str();
}

void IterTransformDetector::ShowShapeOps() const {
  std::stringstream ss;
  for (const ShapeOperation& op : shape_ops) {
    ss << FormatOp(op) << "\n";
  }
  LOG(INFO) << "Current shape ops info:\n" << ss.str();
}

/**
 *! \brief When detect split/fuse sequence, we split all candidate iterations
 * indexed by id into three categories: fuse candidate, split candidate, root.
 */
struct DetectorWorkingSet {
  /*! \brief candidates maybe produced by fusing */
  std::unordered_set<size_t> fuse_set;
  /*! \brief candidates maybe produced by spliting */
  std::unordered_set<size_t> split_set;
  /*! \brief root iterations, either bind to variable or constant */
  std::vector<size_t> root_iters;

  bool HasCandidates() const { return !fuse_set.empty() || !split_set.empty(); }

  /*! \brief Add candidate iteration by the binding type */
  bool Add(size_t iter_idx, const PrimExpr& e, bool is_fuse) {
    if (e->IsInstance<VarNode>() || e->IsInstance<IntImmNode>()) {
      root_iters.push_back(iter_idx);
    } else if (e->IsInstance<FloorDivNode>() || e->IsInstance<FloorModNode>()) {
      split_set.insert(iter_idx);
    } else if (!is_fuse) {
      fuse_set.insert(iter_idx);
    } else {
      LOG(ERROR) << "Illegal iteration component, expect variable or floordiv/floormod: " << e;
      return false;
    }
    return true;
  }
};

/**
 *! \brief Analyze iteration fusing operations.
 */
static bool ProcessFuseSet(IterTransformDetector* self, DetectorWorkingSet* working_set) {
  while (!working_set->fuse_set.empty()) {
    size_t cur_idx = *working_set->fuse_set.begin();
    working_set->fuse_set.erase(cur_idx);
    PrimExpr e = self->GetBinding(cur_idx);
    // get sum components, and sort in descend order by factors
    std::vector<std::pair<PrimExpr, int>> factors;
    int constant = 0;
    ExtractSumFactors(e, true, &factors, &constant);
    std::sort(factors.begin(), factors.end(),
              [](const auto& p1, const auto& p2) { return p1.second < p2.second; });
    if (constant != 0) {
      LOG(ERROR) << "Non-trivial constant offset for " << e;
      return false;
    }
    if (factors.empty()) {
      ICHECK(e->IsInstance<IntImmNode>()) << e;
      working_set->root_iters.push_back(cur_idx);
      continue;
    }
    while (!factors.empty()) {
      int cur_factor = factors.back().second;
      if (cur_factor <= 0) {
        LOG(ERROR) << "Illegal factor " << cur_factor << " at binding " << e;
        return false;
      }
      if (factors.size() == 1) {
        if (cur_factor > 1) {  // ... + x * factor + 0 * 1: inner is single point
          PrimExpr inner = 0;
          PrimExpr outer = factors[0].first;
          size_t inner_idx = self->AddNewIter(inner, cur_factor);
          size_t outer_idx = self->AddNewIter(outer, -1);
          self->AddIterOperation(IterOperation::Fuse(outer_idx, inner_idx, cur_idx));
          if (!working_set->Add(inner_idx, inner, true)) return false;
          if (!working_set->Add(outer_idx, outer, true)) return false;
        } else {
          if (!working_set->Add(cur_idx, factors[0].first, true)) return false;
        }
        break;
      }

      PrimExpr outer = factors.back().first;
      size_t outer_idx = self->AddNewIter(outer, -1);
      if (!working_set->Add(outer_idx, outer, true)) return false;

      PrimExpr inner = std::accumulate(
          factors.rbegin() + 1, factors.rend(), make_const(DataType::Int(32), 0),
          [](PrimExpr cur, const auto& fac) { return cur + fac.first * fac.second; });
      size_t inner_idx = self->AddNewIter(inner, cur_factor);
      self->AddIterOperation(IterOperation::Fuse(outer_idx, inner_idx, cur_idx));
      cur_idx = inner_idx;
      factors.pop_back();

      /*if (factors.back().second != 1) {
        // ... + x * factor + 0 * 1: inner is single point
        factors.push_back({IntImm(DataType::Int(32), 0), 1});
        continue;
      }
      // ... + x * factor + y
      int64_t cur_factor = factors[factors.size() - 2].second;
      if (cur_factor <= 0) {
        LOG(ERROR) << "Illegal factor " << cur_factor << " at binding " << e;
        return false;
      }
      // get inner iteration
      PrimExpr inner = factors[factors.size() - 1].first;
      size_t inner_idx = self->AddNewIter(inner, cur_factor);
      if (!working_set->Add(inner_idx, inner, true)) return false;
      // get outer iteration
      PrimExpr outer = IntImm(DataType::Int(32), 0);
      for (size_t j = factors.size() - 1; j > 0; --j) {
        int64_t next_factor = factors[j - 1].second;
        if (next_factor % cur_factor != 0) {
          LOG(ERROR) << "Illegal factor " << next_factor << " for " << factors[j - 1].first
                     << " at binding " << e << ", expect times of " << cur_factor;
          return false;
        }
        factors[j - 1].second = next_factor / cur_factor;
        outer = factors[j - 1].first * factors[j - 1].second + outer;
      }
      size_t outer_idx = self->AddNewIter(outer, -1);
      self->AddIterOperation(IterOperation::Fuse(outer_idx, inner_idx, cur_idx));
      cur_idx = outer_idx;  // cur_binding = (... // factor + x)
      factors.pop_back();*/
    }
    // single component
    // CHECK_EQ(factors[0].second, 1);
    // if (!working_set->Add(cur_idx, self->GetBinding(cur_idx), true)) return false;
  }
  return true;
}

/**
 *! \brief Analyze iteration split operations when no matching floordiv/floormod pair can be found.
 *  we add dummy iterations for missing floordiv or floormod.
 */
bool ProcessUnmatchedSplitSet(IterTransformDetector* self, DetectorWorkingSet* working_set) {
  auto& split_set = working_set->split_set;
  for (auto it = split_set.begin(); it != split_set.end(); ++it) {
    const PrimExpr& e = self->GetBinding(*it);
    PrimExpr to_split;
    int64_t factor;
    if (const FloorDivNode* div = e.as<FloorDivNode>()) {
      to_split = div->a;
      factor = div->b.as<IntImmNode>()->value;
    } else if (const FloorModNode* mod = e.as<FloorModNode>()) {
      to_split = mod->a;
      factor = mod->b.as<IntImmNode>()->value;
    } else {
      LOG(ERROR) << "Illegal split expression " << e;
      return false;
    }
    size_t split_idx = self->AddNewIter(to_split, -1);
    size_t unused_idx = self->AddNewIter(IntImm(DataType::Int(32), 0), 0);
    if (!working_set->Add(split_idx, to_split, false)) return false;
    size_t outer_idx, inner_idx;
    if (e->IsInstance<FloorDivNode>()) {
      outer_idx = *it;
      inner_idx = unused_idx;
    } else {
      outer_idx = unused_idx;
      inner_idx = *it;
    }
    if (self->GetExtent(inner_idx) > 0 && self->GetExtent(inner_idx) != factor) {
      LOG(ERROR) << "Split for " << e << " inner extent should be " << factor << ", but requires "
                 << self->GetExtent(inner_idx);
      return false;
    }
    self->iter_extents[inner_idx] = factor;
    self->AddIterOperation(IterOperation::Split(split_idx, outer_idx, inner_idx));
  }
  split_set.clear();
  return true;
}

/**
 *! \brief Analyze iteration split operations.
 */
static bool ProcessSplitSet(IterTransformDetector* self, DetectorWorkingSet* working_set) {
  auto& split_set = working_set->split_set;
  while (!split_set.empty()) {
    bool found_pair = false;
    for (auto it = split_set.begin(); it != split_set.end(); ++it) {
      PrimExpr e = self->GetBinding(*it);
      PrimExpr companion;  // eg, for (x * 10 + y) // 8 the companion to find is (x * 10 + y) % 8
      PrimExpr to_split;   // eg, for (x * 10 + y) // 8 it is x * 10 + y
      int64_t factor;
      if (const FloorDivNode* div = e.as<FloorDivNode>()) {
        to_split = div->a;
        if (const IntImmNode* imm = div->b.as<IntImmNode>()) {
          factor = imm->value;
        } else {
          LOG(ERROR) << "Floordiv is illegal: " << e;
          return false;
        }
        companion = FloorMod(div->a, div->b);
      } else if (const FloorModNode* mod = e.as<FloorModNode>()) {
        to_split = mod->a;
        if (const IntImmNode* imm = mod->b.as<IntImmNode>()) {
          factor = imm->value;
        } else {
          LOG(ERROR) << "Floormod is illegal: " << e;
          return false;
        }
        companion = FloorDiv(mod->a, mod->b);
      } else {
        LOG(ERROR) << "Illegal split expression " << e;
        return false;
      }
      // e is either x // n or x % n, we try find its companion iteration in split candidates
      std::unordered_set<size_t>::iterator it2 = it;
      ++it2;
      it2 = std::find_if(it2, split_set.end(), [self, &companion](size_t k) {
        return StructuralEqual()(self->GetBinding(k), companion);
      });
      if (it2 != split_set.end()) {
        // both x // n and x % n in split candidates, remove them from working set
        // and add x into working set.
        size_t outer_idx, inner_idx;
        size_t split_idx = self->AddNewIter(to_split, -1);
        if (!working_set->Add(split_idx, to_split, false)) return false;
        if (e->IsInstance<FloorDivNode>()) {
          outer_idx = *it;
          inner_idx = *it2;
        } else {
          outer_idx = *it2;
          inner_idx = *it;
        }
        if (self->GetExtent(inner_idx) > 0 && self->GetExtent(inner_idx) < factor) {
          LOG(ERROR) << "Split for " << e << " inner extent should be " << factor
                     << ", but requires " << self->GetExtent(inner_idx);
          return false;
        }
        self->iter_extents[inner_idx] = factor;
        self->AddIterOperation(IterOperation::Split(split_idx, outer_idx, inner_idx));
        split_set.erase(outer_idx);
        split_set.erase(inner_idx);
        found_pair = true;
        break;
      }
    }
    if (!found_pair) {
      // till here no floor/mod pair are found
      if (!ProcessUnmatchedSplitSet(self, working_set)) return false;
    }
  }
  return true;
}

/**
 *! \brief Check there is no input variable bind by multiple root iterations. Thus all input
 * variables are used at most once in iteration operations.
 * \param root_iters  root iteration ids, each bind to a variable or constant.
 * \param bindings  iteration expr bindings indexed by id.
 * \param input_extents  input variable iteration extents dict.
 */
static bool CheckDuplicate(const std::vector<size_t>& root_iters,
                           const std::vector<PrimExpr>& bindings,
                           const std::unordered_map<const VarNode*, int64_t>& input_extents) {
  std::set<const VarNode*> used_vars;
  for (size_t root_idx : root_iters) {
    const PrimExpr& e = bindings[root_idx];
    if (const VarNode* v = e.as<VarNode>()) {
      if (!input_extents.count(v)) {
        LOG(ERROR) << "Unknown variable " << e;
        return false;
      } else if (used_vars.count(v)) {
        LOG(ERROR) << "Duplicate variable " << e << ", maybe get fused or splitted multiple times";
        return false;
      } else {
        used_vars.insert(v);
      }
    } else if (!e->IsInstance<IntImmNode>()) {
      LOG(ERROR) << "Unknown root iteration binding: " << e;
      return false;
    }
  }
  return true;
}

bool IterTransformDetector::InferExtents(const std::vector<size_t>& root_iters) {
  // fill extents for root iterations
  for (size_t root_idx : root_iters) {
    const PrimExpr& e = iter_bindings[root_idx];
    if (const VarNode* v = e.as<VarNode>()) {
      ICHECK(input_extents.count(v));
      if (iter_extents[root_idx] > 0 && iter_extents[root_idx] != input_extents[v]) {
        if (iter_extents[root_idx] < input_extents[v]) {
          LOG(ERROR) << "Variable " << e << " extent confliction, domain extent is "
                     << input_extents[v] << ", but requires at least " << iter_extents[root_idx];
          return false;
        } else if (respect_input_dom) {
          iter_extents[root_idx] = input_extents[v];
        }
      } else {
        iter_extents[root_idx] = input_extents[v];
      }
    } else {
      // dummy indice extent should get bind already.
      ICHECK_GE(iter_extents[root_idx], 0);
    }
  }
  // visit ops in reverse order, ensure op's src extents are inferred already
  for (auto it = op_seq.rbegin(); it != op_seq.rend(); ++it) {
    const IterOperation& op = *it;
    if (op.type == FUSE) {
      std::pair<size_t, size_t> p = op.fuse_src();
      size_t outer_idx = p.first;
      size_t inner_idx = p.second;
      size_t fuse_idx = op.fuse_dst();
      ICHECK_GT(iter_extents[inner_idx], 0);
      ICHECK_GT(iter_extents[outer_idx], 0);
      int64_t fuse_extent = iter_extents[inner_idx] * iter_extents[outer_idx];
      if (iter_extents[fuse_idx] > 0 && iter_extents[fuse_idx] != fuse_extent) {
        if (!respect_input_dom || iter_extents[fuse_idx] < fuse_extent) {
          LOG(ERROR) << FormatOp(op) << " extent confliction, "
                     << "fuse extent is " << fuse_extent << ", but requires "
                     << iter_extents[fuse_idx];
          return false;
        }
      }
      iter_extents[fuse_idx] = fuse_extent;
    } else {
      std::pair<size_t, size_t> p = op.split_dst();
      size_t outer_idx = p.first;
      size_t inner_idx = p.second;
      size_t split_idx = op.split_src();
      ICHECK_GT(iter_extents[split_idx], 0);
      ICHECK_GT(iter_extents[inner_idx], 0);
      if (iter_extents[split_idx] % iter_extents[inner_idx] != 0) {
        LOG(ERROR) << FormatOp(op) << " extent confliction, "
                   << "inner extent " << iter_extents[inner_idx] << " is not a divident of "
                   << iter_extents[split_idx];
        return false;
      } else {
        int64_t outer_extent = iter_extents[split_idx] / iter_extents[inner_idx];
        if (iter_extents[outer_idx] > 0 && iter_extents[outer_idx] != outer_extent) {
          LOG(ERROR) << FormatOp(op) << " extent confliction, "
                     << "outer extent be " << outer_extent << ", but requires "
                     << iter_extents[outer_idx];
          return false;
        }
        iter_extents[outer_idx] = outer_extent;
      }
    }
  }
  return true;
}

bool IterTransformDetector::DetectFuseSplit(const Array<Var>& X, const Array<PrimExpr>& Y,
                                            const Map<Var, Range>& dom_map) {
  // Bind input iteration vars, then try simplify binding expressions
  arith::Analyzer analyzer;
  for (const Var& v : X) {
    CHECK(dom_map.count(v));
    IntImm const_extent = Downcast<IntImm>(dom_map[v]->extent);
    if (!const_extent.defined()) {
      LOG(ERROR) << "Input domain extents should be constant at " << v << ":" << dom_map[v];
      return false;
    }
    analyzer.Bind(v, dom_map[v]);
    input_extents[v.get()] = const_extent->value;
  }
  std::vector<PrimExpr> simplified_bindings;
  for (const PrimExpr& e : Y) {
    if (e->IsInstance<VarNode>()) {
      simplified_bindings.push_back(e);  // prevent extent=1 itervar optimized out
    } else {
      simplified_bindings.push_back(analyzer.Simplify(e));
    }
  }

  // Visit iterations backwards from output iteration bindings to input,
  // - first process fuses, until no more candidates are result of a fuse operation.
  // - then process splits, until no more candidates are result of a split operation.
  // iterate on above steps until only root iterations left in working set.
  // the detection should ensure that every iteration:
  // (1) either a root iteration or produced by single split/fuse op
  // (2) can only be source iteration of at most one split/fuse op.
  DetectorWorkingSet working_set;
  for (size_t i = 0; i < Y.size(); ++i) {
    size_t out_iter_idx = AddNewIter(simplified_bindings[i], -1);
    working_set.fuse_set.insert(out_iter_idx);
  }
  while (working_set.HasCandidates()) {
    if (!ProcessFuseSet(this, &working_set)) return false;
    if (!ProcessSplitSet(this, &working_set)) return false;
  }

  // Detect conflictions caused by duplicate variable usage or unmatching iteration extents.
  if (!CheckDuplicate(working_set.root_iters, iter_bindings, input_extents)) return false;
  if (!InferExtents(working_set.root_iters)) return false;
  if (verbose) {
    ShowOps();
  }
  return true;
}

void IterTransformDetector::ReorderFuseSplit() {
  bool updated = true;
  while (updated) {
    updated = false;
    for (size_t i = 0; i < op_seq.size(); ++i) {  // reverse order visit
      // find fuse -> split pair [A, B] -> [AB=CD] -> [C, D]
      // try optimize to split -> fuse
      const IterOperation& split = op_seq[i];
      if (split.type != SPLIT) continue;
      int64_t producer_op_idx = iter_producer_mapping[split.split_src()];
      if (producer_op_idx < 0) continue;
      const IterOperation& fuse = op_seq[producer_op_idx];
      if (fuse.type != FUSE) continue;
      ICHECK_EQ(fuse.fuse_dst(), split.split_src());
      int fuse_outer_ext = iter_extents[fuse.fuse_src().first];
      int fuse_inner_ext = iter_extents[fuse.fuse_src().second];
      int split_outer_ext = iter_extents[split.split_dst().first];
      int split_inner_ext = iter_extents[split.split_dst().second];
      ICHECK_EQ(fuse_outer_ext * fuse_inner_ext, split_outer_ext * split_inner_ext);
      if (fuse_outer_ext % split_outer_ext == 0) {
        // "[XY, Z] -> [XYZ] -> [X, YZ]" == "[XY, Z] -> [[X, Y], Z] -> [X, YZ]"
        int factor = fuse_outer_ext / split_outer_ext;
        size_t new_split_src_idx = fuse.fuse_src().first;
        size_t new_fuse_inner_idx = fuse.fuse_src().second;
        size_t new_split_outer_idx = split.split_dst().first;
        size_t new_split_inner_idx = fuse.fuse_dst();
        size_t new_fuse_dst_idx = split.split_dst().second;
        UpdateIterInfo(new_split_outer_idx, floordiv(iter_bindings[new_split_src_idx], factor),
                       split_outer_ext);
        UpdateIterInfo(new_split_inner_idx, floormod(iter_bindings[new_split_src_idx], factor),
                       factor);
        UpdateIterInfo(new_fuse_dst_idx,
                       iter_bindings[new_split_inner_idx] * split_inner_ext +
                           iter_bindings[new_fuse_inner_idx],
                       split_inner_ext);
        IterOperation new_split =
            IterOperation::Split(new_split_src_idx, new_split_outer_idx, new_split_inner_idx);
        IterOperation new_fuse =
            IterOperation::Fuse(new_split_inner_idx, new_fuse_inner_idx, new_fuse_dst_idx);
        UpdateIterOperation(producer_op_idx, new_split);
        UpdateIterOperation(i, new_fuse);
        updated = true;
        break;
      } else if (fuse_inner_ext % split_inner_ext == 0) {
        // "[X, YZ] -> [XYZ] -> [XY, Z]" == "[X, YZ] -> [X, [Y, Z]] -> [XY, Z]"
        int factor = split_inner_ext;
        size_t new_split_src_idx = fuse.fuse_src().second;
        size_t new_fuse_outer_idx = fuse.fuse_src().first;
        size_t new_split_outer_idx = fuse.fuse_dst();
        size_t new_split_inner_idx = split.split_dst().second;
        size_t new_fuse_dst_idx = split.split_dst().first;
        UpdateIterInfo(new_split_outer_idx, floordiv(iter_bindings[new_split_src_idx], factor),
                       fuse_inner_ext / split_inner_ext);
        UpdateIterInfo(new_split_inner_idx, floormod(iter_bindings[new_split_src_idx], factor),
                       factor);
        UpdateIterInfo(new_fuse_dst_idx,
                       iter_bindings[new_fuse_outer_idx] * (fuse_inner_ext / split_inner_ext) +
                           iter_bindings[new_split_outer_idx],
                       split_outer_ext);
        IterOperation new_split =
            IterOperation::Split(new_split_src_idx, new_split_outer_idx, new_split_inner_idx);
        IterOperation new_fuse =
            IterOperation::Fuse(new_fuse_outer_idx, new_split_outer_idx, new_fuse_dst_idx);
        UpdateIterOperation(producer_op_idx, new_split);
        UpdateIterOperation(i, new_fuse);
        updated = true;
        break;
      }
    }
  }
}

/**
 *! \brief Recursive visit fuse operations with current iter as the final fused result,
 * return iterations that are fused into current iter. The outer iter is at left.
 */
static void DoVisitFuse(IterTransformDetector* self, size_t iter_idx,
                        std::vector<size_t>* input_iters) {
  int64_t op_idx = self->iter_producer_mapping[iter_idx];
  if (op_idx < 0 || self->op_seq[op_idx].type == SPLIT) {
    input_iters->push_back(iter_idx);
    return;
  }
  const IterOperation& fuse = self->op_seq[op_idx];
  ICHECK_EQ(fuse.fuse_dst(), iter_idx);
  DoVisitFuse(self, fuse.fuse_src().first, input_iters);
  DoVisitFuse(self, fuse.fuse_src().second, input_iters);
}

using InputIterInfo = std::pair<size_t, std::list<size_t>>;

/**
 *! \brief Recursive visit split operations with current iter as the final split result,
 * return iterations that are splited. For each returned iter, we also record the whole set
 * of iters under it's split tree. The outer iter is at left.
 */
static void DoVisitSplit(IterTransformDetector* self, size_t iter_idx,
                         std::vector<InputIterInfo>* input_iters_info,
                         std::list<size_t>* cur_leaves, std::unordered_set<size_t>* unvisited) {
  int64_t op_idx = self->iter_producer_mapping[iter_idx];
  if (op_idx < 0 || self->op_seq[op_idx].type == FUSE) {
    input_iters_info->push_back({iter_idx, *cur_leaves});
    return;
  }
  const IterOperation& split = self->op_seq[op_idx];
  size_t outer_idx = split.split_dst().first;
  size_t inner_idx = split.split_dst().second;
  if (outer_idx == iter_idx) {
    cur_leaves->push_back(inner_idx);
  } else if (inner_idx == iter_idx) {
    cur_leaves->push_front(outer_idx);
  } else {
    LOG(FATAL) << "Should never happen";
  }
  unvisited->erase(outer_idx);
  unvisited->erase(inner_idx);
  DoVisitSplit(self, split.split_src(), input_iters_info, cur_leaves, unvisited);
}

/**
 *! \brief Given a sequence of input iters, and a sequence of output iters which are transformed
 *from input iters by split ops. Infer equavalent reshape/transpose operations.
 */
static void InferReshapeTransposeOps(const std::vector<size_t>& out_iters,
                                     const std::vector<InputIterInfo>& input_iters_info,
                                     const std::vector<int64_t>& extents,
                                     std::vector<ShapeOperation>* shape_ops) {
  // iter_idx -> index in `out_iters`
  std::unordered_map<size_t, size_t> out_iter_ordering;
  for (size_t i = 0; i < out_iters.size(); ++i) {
    out_iter_ordering[out_iters[i]] = i;
  }

  // a division to out iters, iters in each part will be merged by reshape op
  // and refer to an axis in transpose op then.
  // each item record {begin, end, transpose_tgt_idx, volume} triple.
  std::vector<std::tuple<int, int, int, int>> out_iter_divisions;

  // expect to create ops out_reshape -> transpose -> input_reshape
  std::vector<int64_t> out_reshape, input_reshape;

  // store iter ids for each reshape group
  std::vector<std::vector<size_t>> out_reshape_iters, input_reshape_iters;

  // iterate all output iterations with respect to the order specified
  // by `input_iters`, which are split tree roots for `out_iters`
  int origin_begin = -1;
  int origin_end = -1;
  int volume = 1;
  for (size_t i = 0; i < input_iters_info.size(); ++i) {
    // infer input reshape
    const auto& p = input_iters_info[i];
    size_t input_iter = p.first;
    input_reshape.push_back(extents[input_iter]);
    input_reshape_iters.push_back({input_iter});

    // infer output reshape
    const auto& input_iter_cover = p.second;
    for (size_t iter_idx : input_iter_cover) {
      bool has_range = origin_begin >= 0;
      auto it = out_iter_ordering.find(iter_idx);
      if (it == out_iter_ordering.end()) {
        // the splitted iter is not in out iters, it should be a broadcast dimension
        if (has_range) {
          out_iter_divisions.push_back(
              {origin_begin, origin_end, out_iter_divisions.size(), volume});
        }
        out_iter_divisions.push_back(
            {origin_begin, origin_end, out_iter_divisions.size(), -extents[iter_idx]});
        origin_begin = -1;
        origin_end = -1;
        volume = 1;
      } else if (has_range && origin_end == static_cast<int>(it->second)) {
        // can merge consequtive iterations in out iters
        origin_end += 1;
        volume *= extents[iter_idx];
      } else {
        // trivial case, process a single iteration in out iters
        if (has_range) {
          out_iter_divisions.push_back(
              {origin_begin, origin_end, out_iter_divisions.size(), volume});
        }
        origin_begin = it->second;
        origin_end = origin_begin + 1;
        volume = extents[iter_idx];
      }
    }
  }
  if (origin_begin >= 0) {  // tail part
    out_iter_divisions.push_back({origin_begin, origin_end, out_iter_divisions.size(), volume});
  }

  // inference transpose and reshape from out iter divisions.
  std::vector<int64_t> transpose(out_iter_divisions.size());
  std::vector<std::vector<size_t>> transpose_iters(out_iter_divisions.size());
  std::stable_sort(
      out_iter_divisions.begin(), out_iter_divisions.end(),
      [](const auto& p1, const auto& p2) { return std::get<0>(p1) < std::get<0>(p2); });
  for (size_t i = 0; i < out_iter_divisions.size(); ++i) {
    int begin = std::get<0>(out_iter_divisions[i]);
    int end = std::get<1>(out_iter_divisions[i]);
    int transpose_idx = std::get<2>(out_iter_divisions[i]);
    int volume = std::get<3>(out_iter_divisions[i]);
    std::vector<size_t> merged_iters(out_iters.begin() + begin, out_iters.begin() + end);
    transpose[transpose_idx] = i;
    transpose_iters[transpose_idx] = merged_iters;
    out_reshape.emplace_back(volume);
    out_reshape_iters.emplace_back(merged_iters);
  }
  shape_ops->push_back(ShapeOperation(RESHAPE, out_reshape, out_reshape_iters));
  shape_ops->push_back(ShapeOperation(TRANSPOSE, transpose, transpose_iters));
  shape_ops->push_back(ShapeOperation(RESHAPE, input_reshape, input_reshape_iters));
}

static void ConvertFuseSplitSeq(IterTransformDetector* self, const std::vector<size_t>& out_iters,
                                const std::unordered_map<size_t, size_t>& force_iter_order,
                                std::vector<size_t>* p_input_iters) {
  // visit a fuse sequence, get iters that are fused into `out_iters`
  // they are treated as the results of a split sequence next.
  std::vector<size_t> split_out_iters;
  for (size_t iter_idx : out_iters) {
    DoVisitFuse(self, iter_idx, &split_out_iters);
  }
  std::unordered_set<size_t> split_out_iters_set(split_out_iters.begin(), split_out_iters.end());

  // visit a split sequence, get iters that are splited as `split_out_iters`
  // they are treated as input iters in current convert round.
  std::vector<InputIterInfo> input_iters_info;
  for (size_t iter_idx : split_out_iters) {
    if (split_out_iters_set.count(iter_idx)) {
      std::list<size_t> split_subset;
      split_subset.push_back(iter_idx);
      DoVisitSplit(self, iter_idx, &input_iters_info, &split_subset, &split_out_iters_set);
    }
  }

  // sort input iterations respect to input order
  auto f_get_order = [&force_iter_order](size_t iter_idx) -> size_t {
    auto it = force_iter_order.find(iter_idx);
    return it == force_iter_order.end() ? std::numeric_limits<size_t>::max() : it->second;
  };
  std::sort(input_iters_info.begin(), input_iters_info.end(),
            [&f_get_order](const InputIterInfo& x, const InputIterInfo& y) {
              return f_get_order(x.first) < f_get_order(y.first);
            });

  // infer reshape/transpose operations
  InferReshapeTransposeOps(split_out_iters, input_iters_info, self->iter_extents, &self->shape_ops);

  // output input iteration indexes
  for (const InputIterInfo& p : input_iters_info) {
    p_input_iters->push_back(p.first);
  }
}

/**
 *! \brief Create reshape/transpose from `root_iters` to input iterations.
 * assume root iterations take consistent order with input variables.
 */
static void ConvertRootReshape(const Array<Var>& X,
                               const std::unordered_map<const VarNode*, int64_t>& input_extents,
                               const std::vector<size_t>& root_iters,
                               const std::vector<PrimExpr>& bindings,
                               const std::vector<int64_t>& extents,
                               std::vector<ShapeOperation>* shape_ops) {
  std::unordered_set<size_t> drop_indices;
  std::vector<int64_t> reshape;
  std::vector<std::vector<size_t>> reshape_iters;
  size_t i = 0, j = 0;
  bool is_trivial = true;
  while (i < X.size() && j < root_iters.size()) {
    const Var& v = X[i];
    const PrimExpr& e = bindings[root_iters[j]];
    if (e.same_as(v)) {
      // match var
      ICHECK_GE(extents[root_iters[j]], input_extents.at(v.get()));
      reshape.push_back(extents[root_iters[j]]);
      reshape_iters.push_back({root_iters[j]});
      ++i;
      ++j;
    } else if (e->IsInstance<VarNode>()) {
      // free input var, denotes a broadcast
      if (input_extents.at(v.get()) > 1) {
        reshape.push_back(-input_extents.at(v.get()));
        reshape_iters.push_back({});
        is_trivial = false;
      }
      ++i;
    } else {
      // zero values root iter, denotes a strided access
      const auto imm = e.as<IntImmNode>();
      ICHECK(imm && imm->value == 0);
      drop_indices.insert(reshape.size());
      reshape.push_back(extents[root_iters[j]]);
      reshape_iters.push_back({root_iters[j]});
      is_trivial = false;
      ++j;
    }
  }
  while (i < X.size()) {
    // free input var, denotes a broadcast
    const Var& v = X[i];
    if (input_extents.at(v.get()) > 1) {
      reshape.push_back(-input_extents.at(v.get()));
      reshape_iters.push_back({});
      is_trivial = false;
    }
    ++i;
  }
  while (j < root_iters.size()) {
    // zero values root iter, denotes a strided access
    const PrimExpr& e = bindings[root_iters[j]];
    const auto imm = e.as<IntImmNode>();
    ICHECK(imm && imm->value == 0);
    drop_indices.insert(reshape.size());
    reshape.push_back(extents[root_iters[j]]);
    reshape_iters.push_back({root_iters[j]});
    is_trivial = false;
    ++j;
  }
  if (!is_trivial) {
    shape_ops->push_back(ShapeOperation(RESHAPE, reshape, reshape_iters));
    if (!drop_indices.empty()) {
      std::vector<int64_t> transpose;
      std::vector<std::vector<size_t>> transpose_iters;
      for (i = 0; i < reshape.size(); ++i) {
        if (!drop_indices.count(i)) {
          transpose.push_back(i);
          transpose_iters.push_back(reshape_iters[i]);
        }
      }
      shape_ops->push_back(ShapeOperation(TRANSPOSE, transpose, transpose_iters));
    }
  }
}

void IterTransformDetector::OptimizeReshapeTranspose() {
  if (verbose) {
    LOG(INFO) << "Try optimize shape operations";
    ShowShapeOps();
  }
  std::vector<ShapeOperation> new_ops;
  int cur_reshape = -1;
  size_t i = 0;
  while (i < shape_ops.size()) {
    const ShapeOperation& op = shape_ops[i];
    if (op.type == RESHAPE) {
      if (std::any_of(op.values.begin(), op.values.end(), [](int64_t x) { return x < 0; })) {
        // should not optimize broadcast reshape
        new_ops.push_back(op);
        cur_reshape = -1;
      }
      cur_reshape = i;
    } else {
      bool trivial_transpose = true;
      if (i < 1 || shape_ops[i - 1].ndim() != op.ndim()) {
        trivial_transpose = false;
      }
      for (size_t j = 0; j < op.ndim(); ++j) {
        if (op.values[j] != static_cast<int64_t>(j)) {
          trivial_transpose = false;
          break;
        }
      }
      if (!trivial_transpose) {
        // can not optimize transpose
        if (cur_reshape >= 0) {
          new_ops.push_back(shape_ops[cur_reshape]);
          cur_reshape = -1;
        }
        new_ops.push_back(op);
      }
    }
    ++i;
  }
  if (cur_reshape >= 0) {
    new_ops.push_back(shape_ops[cur_reshape]);
  }
  std::swap(this->shape_ops, new_ops);
}

bool IterTransformDetector::DetectReshapeTranspose(const Array<Var>& X, const Array<PrimExpr>& Y,
                                                   const Map<Var, Range>& dom_map) {
  // detect fuse/split ops, prefer split before fuse
  if (!DetectFuseSplit(X, Y, dom_map)) return false;
  ReorderFuseSplit();

  // enforce order for input iterations
  std::unordered_map<size_t, size_t> iter_input_order;
  for (size_t i = 0; i < iter_bindings.size(); ++i) {
    const PrimExpr& e = iter_bindings[i];
    auto it = std::find_if(X.begin(), X.end(), [&e](const Var& v) { return e.same_as(v); });
    if (it != X.end()) {
      iter_input_order[i] = it - X.begin();
    }
  }

  // iterative apply `ConvertFuseSplitSeq` starting from output iterations.
  // stop until no non-root iterations left.
  std::vector<size_t> input_iters, out_iters;
  for (size_t i = 0; i < Y.size(); ++i) {
    out_iters.push_back(i);
  }
  std::unordered_set<size_t> root_iters;
  for (size_t i = 0; i < iter_bindings.size(); ++i) {
    if (iter_producer_mapping[i] < 0) root_iters.insert(i);
  }
  bool reach_root = false;
  while (!reach_root) {
    reach_root = true;
    ConvertFuseSplitSeq(this, out_iters, iter_input_order, &input_iters);
    reach_root = std::all_of(input_iters.begin(), input_iters.end(),
                             [&root_iters](size_t iter_idx) { return root_iters.count(iter_idx); });
    std::swap(input_iters, out_iters);
  }

  // process root iterations, where we assume iterations in `out_iters`:
  // - if constant, must be 0 and denotes a strided access
  // - if variable, should be ordered respect to input variable orders
  // - input variable not appear in `out_iters` denotes a broadcasted access
  ConvertRootReshape(X, input_extents, out_iters, iter_bindings, iter_extents, &shape_ops);

  // optimize reshape/transform ops
  OptimizeReshapeTranspose();
  return true;
}

static int64_t DoInferIterStride(const IterTransformDetector* self, size_t iter_idx,
                                 std::vector<int64_t>* strides_map) {
  int64_t cur_val = (*strides_map)[iter_idx];
  if (cur_val >= 0) {
    return cur_val;  // already has result
  } else if (cur_val == -2) {
    return -1;  // already on visit path
  } else {
    (*strides_map)[iter_idx] = -2;
  }
  // visit producers
  int64_t op_idx = self->iter_producer_mapping[iter_idx];
  int64_t stride_from_producer = -1;
  if (op_idx >= 0) {
    const IterOperation& op = self->op_seq[op_idx];
    if (op.type == FUSE) {
      size_t outer_idx = op.fuse_src().first;
      size_t inner_idx = op.fuse_src().second;
      int64_t factor = self->iter_extents[inner_idx];
      int64_t outer_stride = DoInferIterStride(self, outer_idx, strides_map);
      int64_t inner_stride = DoInferIterStride(self, inner_idx, strides_map);
      if (inner_stride >= 0 && (outer_stride < 0 || outer_stride == inner_stride * factor)) {
        // stride of vi = stride of (vo * c + vi), if stride of vo is compatible
        stride_from_producer = inner_stride;
      } else if (outer_stride >= 0 && outer_stride % factor == 0 &&
                 (inner_stride < 0 || outer_stride == inner_stride * factor)) {
        stride_from_producer = outer_stride / factor;
      }
    } else {
      size_t outer_idx = op.split_dst().first;
      size_t inner_idx = op.split_dst().second;
      int64_t factor = self->iter_extents[inner_idx];
      int64_t fuse_stride = DoInferIterStride(self, op.split_src(), strides_map);
      if (iter_idx == outer_idx) {
        int64_t inner_stride = DoInferIterStride(self, inner_idx, strides_map);
        if (inner_stride >= 0 && (fuse_stride < 0 || fuse_stride == inner_stride)) {
          // stride of vo = c * stride of vi, if stride of (vo * c + vi) is compatible
          stride_from_producer = inner_stride * factor;
        } else if (fuse_stride >= 0 && (inner_stride < 0 || fuse_stride == inner_stride)) {
          // stride of vo = c * stride of (vo * c + vi), if stride of vi is compatible
          stride_from_producer = fuse_stride * factor;
        }
      } else {
        int64_t outer_stride = DoInferIterStride(self, outer_idx, strides_map);
        if (outer_stride >= 0 && outer_stride % factor == 0 &&
            (fuse_stride < 0 || fuse_stride * factor == outer_stride)) {
          // stride of vi = stride of vo / c, if stride of (vo * c + vi) is compatible
          stride_from_producer = outer_stride / factor;
        } else if (fuse_stride >= 0 && (outer_stride < 0 || fuse_stride * factor == outer_stride)) {
          // stride of vi = stride of (vo * c + vi), if stride of vo is compatible
          stride_from_producer = fuse_stride;
        }
      }
    }
  }
  // visit consumers
  op_idx = self->iter_consumer_mapping[iter_idx];
  int64_t stride_from_consumer = -1;
  if (op_idx >= 0) {
    const IterOperation& op = self->op_seq[op_idx];
    if (op.type == FUSE) {
      size_t outer_idx = op.fuse_src().first;
      size_t inner_idx = op.fuse_src().second;
      int64_t factor = self->iter_extents[inner_idx];
      int64_t fuse_stride = DoInferIterStride(self, op.fuse_dst(), strides_map);
      if (iter_idx == outer_idx) {
        int64_t inner_stride = DoInferIterStride(self, inner_idx, strides_map);
        if (inner_stride >= 0 && (fuse_stride < 0 || fuse_stride == inner_stride)) {
          // stride of vo = c * stride of vi, if stride of (vo * c + vi) is compatible
          stride_from_consumer = inner_stride * factor;
        } else if (fuse_stride >= 0 && (inner_stride < 0 || fuse_stride == inner_stride)) {
          // stride of vo = c * stride of (vo * c + vi), if stride of vi is compatible
          stride_from_consumer = fuse_stride * factor;
        }
      } else {
        int64_t outer_stride = DoInferIterStride(self, outer_idx, strides_map);
        if (outer_stride >= 0 && (fuse_stride < 0 || fuse_stride * factor == outer_stride)) {
          // stride of vi = stride of vo / c, if stride of (vo * c + vi) is compatible
          stride_from_consumer = outer_stride / factor;
        } else if (fuse_stride >= 0 && (outer_stride < 0 || fuse_stride * factor == outer_stride)) {
          // stride of vi = stride of (vo * c + vi), if stride of vo is compatible
          stride_from_consumer = fuse_stride;
        }
      }
    } else {
      size_t outer_idx = op.split_dst().first;
      size_t inner_idx = op.split_dst().second;
      int64_t factor = self->iter_extents[inner_idx];
      int64_t outer_stride = DoInferIterStride(self, outer_idx, strides_map);
      int64_t inner_stride = DoInferIterStride(self, inner_idx, strides_map);
      if (inner_stride >= 0 && (outer_stride < 0 || outer_stride == inner_stride * factor)) {
        // stride of (vo * c + vi) = stride of vi, if stride of vo is compatible
        stride_from_consumer = inner_stride;
      } else if (outer_stride >= 0 && outer_stride % factor == 0 &&
                 (inner_stride < 0 || outer_stride == inner_stride * factor)) {
        // stride of (vo * c + vi) = stride of vo / c, if stride of vi is compatible
        stride_from_consumer = outer_stride / factor;
      }
    }
  }
  // check compatibility
  int64_t result = -1;
  if (stride_from_producer >= 0 && stride_from_consumer >= 0) {
    if (stride_from_producer == stride_from_consumer) {
      result = stride_from_producer;
    }
  } else if (stride_from_producer >= 0) {
    result = stride_from_producer;
  } else if (stride_from_consumer >= 0) {
    result = stride_from_consumer;
  }
  (*strides_map)[iter_idx] = result;
  return result;
}

int64_t IterTransformDetector::InferIterStride(
    size_t iter_idx, const std::unordered_map<size_t, int64_t>& strides_map) const {
  std::vector<int64_t> stride_cache(iter_bindings.size(), -3);
  for (const auto& p : strides_map) {
    stride_cache[p.first] = p.second;
  }
  return DoInferIterStride(this, iter_idx, &stride_cache);
}

bool IterTransformDetector::InferIterDivision(
    const std::vector<size_t>& fuse_iters, const std::unordered_map<size_t, int64_t>& strides_map0,
    const std::unordered_map<size_t, int64_t>& strides_map1, std::vector<int64_t>* p_shape,
    std::vector<int64_t>* p_strides) const {
  // -1: unknown, -2: visited, -3: unvisit yet
  std::vector<int64_t> stride_cache0(iter_bindings.size(), -3);
  for (const auto& p : strides_map0) {
    stride_cache0[p.first] = p.second;
  }
  std::vector<int64_t> stride_cache1(iter_bindings.size(), -3);
  for (const auto& p : strides_map1) {
    stride_cache1[p.first] = p.second;
  }

  int next_stride0 = -1;
  int next_stride1 = -1;
  std::vector<int64_t>& shape = *p_shape;
  std::vector<int64_t>& strides = *p_strides;
  int64_t cur_volume = -1;
  int64_t cur_stride = -1;
  for (int i = fuse_iters.size() - 1; i >= 0; --i) {
    int64_t extent = iter_extents[fuse_iters[i]];
    int64_t stride0 = DoInferIterStride(this, fuse_iters[i], &stride_cache0);
    int64_t stride1 = DoInferIterStride(this, fuse_iters[i], &stride_cache1);
    if (stride0 < 0 || stride1 < 0) return false;
    if (cur_volume == -1) {
      cur_stride = stride0;
      cur_volume = extent;
    } else if (next_stride0 != stride0 ||
               next_stride1 != stride1) {  // can not fuse due to incompatible stride
      shape.push_back(cur_volume);
      strides.push_back(cur_stride);
      cur_stride = stride0;
      cur_volume = extent;
    } else {
      cur_volume = cur_volume * extent;
    }
    next_stride0 = stride0 * extent;
    next_stride1 = stride1 * extent;
  }
  shape.push_back(cur_volume);
  strides.push_back(cur_stride);
  std::reverse(shape.begin(), shape.end());
  std::reverse(strides.begin(), strides.end());
  return true;
}

TVM_REGISTER_GLOBAL("edgex.arith.DetectFuseSplitSeq")
    .set_body_typed([](const Array<Var>& vars, const Array<PrimExpr>& bindings,
                       const Map<Var, Range>& dom_map, bool respect_input_dom,
                       bool verbose) -> runtime::ADT {
      IterTransformDetector detector(respect_input_dom, verbose);
      Array<runtime::ADT> op_results;
      detector.DetectFuseSplit(vars, bindings, dom_map);
      for (auto it = detector.op_seq.rbegin(); it != detector.op_seq.rend(); ++it) {
        const IterOperation& op = *it;
        runtime::String type = op.type == SPLIT ? "split" : "fuse";
        std::vector<ObjectRef> op_fields;
        if (op.type == SPLIT) {
          op_fields.emplace_back(std::move(runtime::String("split")));
          op_fields.emplace_back(IntImm(DataType::Int(64), op.split_src()));
          op_fields.emplace_back(IntImm(DataType::Int(64), op.split_dst().first));
          op_fields.emplace_back(IntImm(DataType::Int(64), op.split_dst().second));
        } else {
          op_fields.emplace_back(std::move(runtime::String("fuse")));
          op_fields.emplace_back(IntImm(DataType::Int(64), op.fuse_dst()));
          op_fields.emplace_back(IntImm(DataType::Int(64), op.fuse_src().first));
          op_fields.emplace_back(IntImm(DataType::Int(64), op.fuse_src().second));
        }
        op_results.push_back(runtime::ADT::Tuple(op_fields));
      }
      std::vector<ObjectRef> fields;
      fields.emplace_back(std::move(op_results));
      fields.emplace_back(std::move(Array<PrimExpr>(detector.iter_bindings)));
      Array<IntImm> extent_arr;
      for (int64_t x : detector.iter_extents) {
        extent_arr.push_back(IntImm(DataType::Int(64), x));
      }
      fields.emplace_back(std::move(extent_arr));
      return runtime::ADT::Tuple(fields);
    });

TVM_REGISTER_GLOBAL("edgex.arith.DetectReshapeTransposeSeq")
    .set_body_typed([](const Array<Var>& vars, const Array<PrimExpr>& bindings,
                       const Map<Var, Range>& dom_map, bool respect_input_dom,
                       bool verbose) -> Array<runtime::ADT> {
      IterTransformDetector detector(respect_input_dom, verbose);
      Array<runtime::ADT> results;
      if (!detector.DetectReshapeTranspose(vars, bindings, dom_map)) {
        return results;
      }
      for (const ShapeOperation& op : detector.shape_ops) {
        runtime::String type = op.type == RESHAPE ? "reshape" : "transpose";
        Array<IntImm> arr;
        for (int64_t v : op.values) {
          arr.push_back(IntImm(DataType::Int(64), v));
        }
        std::vector<ObjectRef> fields = {std::move(type), std::move(arr)};
        results.push_back(runtime::ADT::Tuple(fields));
      }
      return results;
    });

}  // namespace arith
}  // namespace tvm
