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
#include <tvm/arith/bound.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>
#include <unordered_set>

#include "../../../../arith/interval_set.h"
#include "../../../../runtime/thread_storage_scope.h"
#include "../../../../tir/schedule/analysis.h"
#include "../../../../tir/schedule/transform.h"
#include "../../../../tir/schedule/utils.h"
#include "../../../../tir/transforms/ir_utils.h"
#include "./edgex_primitives.h"
#include "./schedule_utils.h"

namespace tvm {
namespace tir {
namespace schedule {

using arith::DeduceBound;
using arith::Intersect;
using arith::IntSet;

using PartitionKey = std::pair<PrimExpr, bool>;
struct PartitionKeyHash {
  std::size_t operator()(PartitionKey const& k) const noexcept {
    std::size_t h1 = ObjectPtrHash{}(k.first);  // NOLINT(whitespace/braces)
    std::size_t h2 = std::hash<bool>{}(k.second);
    return h1 ^ h2;
  }
};

struct PartitionKeyEqual {
  bool operator()(const PartitionKey& k1, const PartitionKey& k2) const {
    // NOLINTNEXTLINE(whitespace/braces)
    return k1.second == k2.second && ObjectPtrEqual{}(k1.first, k2.first);
  }
};

// Each mapping (cond, cond_value) -> interval represents the fact that
// condition cond is proven to have value cond_value (true or false) in interval.
using Partition = std::unordered_map<PartitionKey, IntSet, PartitionKeyHash, PartitionKeyEqual>;

using ExpressionSet = std::unordered_set<PrimExpr, ObjectPtrHash, ObjectPtrEqual>;

// Populate partitions data structure, i.e., for a specific variable,
// find an interval in which each condition
// (currently, "likely" conditions) has fixed true or false value
class PartitionFinder : public StmtExprVisitor {
 public:
  explicit PartitionFinder(Var current_var,
                           const std::unordered_map<const VarNode*, IntSet>& hint_map,
                           const std::unordered_map<const VarNode*, IntSet>& relax_map)
      : current_var_(current_var), hint_map_(hint_map), relax_map_(relax_map) {
    for (const auto& kv : hint_map) {
      out_vars_.insert(kv.first);
    }
    for (const auto& kv : relax_map) {
      out_vars_.insert(kv.first);
    }
  }

  void VisitStmt_(const ForNode* op) final {
    auto is_out_var = [this](const VarNode* v) { return out_vars_.count(v); };
    if (UsesVar(op->min, is_out_var) || UsesVar(op->extent, is_out_var)) return;

    const VarNode* var = op->loop_var.get();
    hint_map_.insert({var, IntSet::Interval(op->min, op->min + op->extent - 1)});
    relax_map_.insert({var, IntSet::Interval(op->min, op->min + op->extent - 1)});
    StmtExprVisitor::VisitStmt_(op);
    relax_map_.erase(var);
    hint_map_.erase(var);
  }

  void VisitStmt_(const BlockRealizeNode* op) final {
    DeducePartitions(op->predicate);

    auto iter_vars = op->block->iter_vars;
    auto iter_bindings = op->iter_values;
    ICHECK_EQ(iter_vars.size(), iter_bindings.size());
    for (size_t i = 0; i < iter_vars.size(); ++i) {
      block_vmap_.Set(iter_vars[i]->var, iter_bindings[i]);
    }
    StmtExprVisitor::VisitStmt_(op);
    for (size_t i = 0; i < iter_vars.size(); ++i) {
      block_vmap_.erase(iter_vars[i]->var);
    }
  }

  void VisitExpr_(const CallNode* op) final {
    PrimExpr cond;
    if (op->op.same_as(builtin::likely())) {
      cond = op->args[0];
    }
    if (cond.defined()) {
      DeducePartitions(cond);
    } else {
      StmtExprVisitor::VisitExpr_(op);
    }
  }

  Partition partitions;

 private:
  void DeducePartitions(PrimExpr cond) {
    cond = Substitute(cond, block_vmap_);
    if (UsesVar(cond, [this](const VarNode* v) { return v == current_var_.get(); })) {
      // For cond, find out the interval, if exists, in which we can prove that cond is
      // true. Also find the interval, if exists, in which we can prove that cond is
      // false.
      IntSet interval = DeduceBound(current_var_, cond, hint_map_, relax_map_);
      if (!interval.IsNothing()) {
        // cond is true within interval
        partitions[{cond, true}] = interval;
      }
      PrimExpr inverse_cond = InverseCond(cond);
      if (inverse_cond.defined()) {
        IntSet interval = DeduceBound(current_var_, inverse_cond, hint_map_, relax_map_);
        if (!interval.IsNothing()) {
          // cond is false within interval
          partitions[{cond, false}] = interval;
        }
      }
    }
  }

  PrimExpr InverseCond(const PrimExpr& cond) {
    PrimExpr inverse_cond;
    if (const LTNode* op = cond.as<LTNode>()) {
      // a < b -> a >= b
      inverse_cond = GE(op->a, op->b);
    } else if (const GTNode* op = cond.as<GTNode>()) {
      // a > b -> a <= b
      inverse_cond = LE(op->a, op->b);
    } else if (const LENode* op = cond.as<LENode>()) {
      // a <= b -> a > b
      inverse_cond = GT(op->a, op->b);
    } else if (const GENode* op = cond.as<GENode>()) {
      // a >= b -> a < b
      inverse_cond = LT(op->a, op->b);
    } else if (const EQNode* op = cond.as<EQNode>()) {
      // a == b -> a != b
      inverse_cond = NE(op->a, op->b);
      // a != b -> a == b
    } else if (const NENode* op = cond.as<NENode>()) {
      inverse_cond = EQ(op->a, op->b);
    }
    return inverse_cond;
  }

  Var current_var_;
  Map<Var, PrimExpr> block_vmap_;
  std::unordered_set<const VarNode*> out_vars_;
  std::unordered_map<const VarNode*, IntSet> hint_map_;
  std::unordered_map<const VarNode*, IntSet> relax_map_;
};

// Replace the set of conditions given by ps with cond_value (true or false)
class ConditionEliminator : public StmtExprMutator {
 public:
  explicit ConditionEliminator(const ExpressionSet& ps, bool cond_value = true)
      : ps_(ps), cond_value_(cond_value) {}

  PrimExpr VisitExpr(const PrimExpr& e) final {
    if (ps_.find(e) != ps_.end()) {
      return VisitExpr(cond_value_ ? const_true() : const_false());
    }
    return StmtExprMutator::VisitExpr(e);
  }

 private:
  ExpressionSet ps_;
  bool cond_value_;
};

// Try to partition range of iteration variables in order to remove (some)
// likely conditions
class LoopPartitioner : public StmtMutator {
 public:
  void AddCandidate(const ForNode* op) { candidate_itervars_.insert(op->loop_var->name_hint); }

 private:
  /**
   *! \brief Try skip current loop and partition the nested body.
   *  It will be called if TryPartition() failed on current loop.
   */
  Stmt VisitLoopBody(const ForNode* op) {
    // normal loop variable can be put into hint map.
    hint_map_.insert({op->loop_var.get(), IntSet::Interval(op->min, op->min + op->extent - 1)});
    Stmt res = StmtMutator::VisitStmt_(op);
    hint_map_.erase(op->loop_var.get());
    return res;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    auto fs = GetRef<Stmt>(op);
    analyzer_.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent), true);
    if (candidate_itervars_.count(op->loop_var->name_hint)) {
      Stmt s = TryPartition(fs, op->loop_var, op->min, op->min + op->extent - 1, op->body);
      if (s.defined()) return s;
    }
    // normal path when loop partition fails
    return VisitLoopBody(op);
  }

  Stmt TryPartition(const Stmt& stmt, Var var, PrimExpr min, PrimExpr max, Stmt body);

  std::pair<IntSet, ExpressionSet> GetIntervalAndCondset(const Partition& partitions,
                                                         const arith::IntervalSet& for_interval,
                                                         bool cond_value);

  /*! \brief candidate for iter vars by name. */
  // TODO(bxq): do not depend on name hint, track new nodes.
  std::unordered_set<std::string> candidate_itervars_;

  std::unordered_map<const VarNode*, IntSet> hint_map_;
  std::unordered_map<const VarNode*, IntSet> relax_map_;
  arith::Analyzer analyzer_;
};

// Returns an interval (in the first component) in which all the conditions
// given in the second component provably have value given by cond_value
std::pair<IntSet, ExpressionSet> LoopPartitioner::GetIntervalAndCondset(
    const Partition& partitions, const arith::IntervalSet& for_interval, bool cond_value) {
  Array<IntSet> sets;
  ExpressionSet cond_set;

  for (const auto& kv : partitions) {
    if (kv.first.second == cond_value) {
      arith::IntervalSet interval = Downcast<arith::IntervalSet>(kv.second);
      arith::IntervalSet intersection = arith::Intersect(&analyzer_, interval, for_interval);
      if (!intersection->IsEmpty()) {
        if ((intersection->HasLowerBound() &&
             analyzer_.CanProve(intersection.min() > for_interval.min())) ||
            (intersection->HasUpperBound() &&
             analyzer_.CanProve(intersection.max() < for_interval.max()))) {
          // we can get more tight interval
          sets.push_back(intersection);
          cond_set.insert(kv.first.first);
        } else {
          // default
          sets.push_back(kv.second);
          cond_set.insert(kv.first.first);
        }
      }
    }
  }
  IntSet interval = sets.empty() ? IntSet::Nothing() : Intersect(sets);
  return std::make_pair(interval, cond_set);
}

/**
 *! \brief Rewrite target loop into partitioned loop. We only keep the block init
 *  in first partitioned loop if the reduce axis is been partitioned.
 */
class LoopBodyRewriter : public StmtExprMutator {
 public:
  explicit LoopBodyRewriter(arith::Analyzer* analyzer) : analyzer_(analyzer) {}

  Stmt Rewrite(const Stmt& op, const Var& loop_var, const PrimExpr& min, const PrimExpr& extent) {
    loop_var_ = loop_var;
    new_loop_var_ = Var(loop_var_->name_hint, loop_var_->dtype, loop_var_->span);
    new_extent_ = extent;
    offset_ = min;
    return VisitStmt(op);
  }

 private:
  PrimExpr VisitExpr_(const VarNode* v) {
    if (v == loop_var_.get()) {
      return new_loop_var_ + offset_;
    } else {
      return StmtExprMutator::VisitExpr_(v);
    }
  }

  bool IsReduceSplit(const BlockRealizeNode* block_realize) const {
    for (size_t i = 0; i < block_realize->iter_values.size(); ++i) {
      if (block_realize->block->iter_vars[i]->iter_type == kCommReduce &&
          UsesVar(block_realize->iter_values[i],
                  [this](const VarNode* v) { return v == loop_var_.get(); })) {
        return true;
      }
    }
    return false;
  }

  Stmt VisitStmt_(const BlockRealizeNode* block_realize) {
    const BlockNode* origin_block = block_realize->block.get();
    bool is_reduce_split = origin_block->init.defined() && IsReduceSplit(block_realize);
    if (is_reduce_split) {
      if (touched_blocks_.count(origin_block)) {
        Stmt updated = StmtExprMutator::VisitStmt_(block_realize);
        const BlockRealizeNode* new_realize = updated.as<BlockRealizeNode>();
        ICHECK(new_realize);
        auto n = CopyOnWrite(new_realize);
        n->block.CopyOnWrite()->init = nullptr;
        return BlockRealize(n);
      } else {
        touched_blocks_.insert(origin_block);
      }
    }
    return StmtExprMutator::VisitStmt_(block_realize);
  }

  Stmt VisitStmt_(const ForNode* op) {
    if (op->loop_var.get() != loop_var_.get()) {
      return StmtExprMutator::VisitStmt_(op);
    }
    ICHECK(op->kind != ForKind::kThreadBinding);
    auto n = CopyOnWrite(op);
    n->min = make_const(op->min.dtype(), 0);
    n->extent = new_extent_;
    n->loop_var = new_loop_var_;
    n->body = VisitStmt(op->body);
    analyzer_->Bind(n->loop_var, Range::FromMinExtent(n->min, n->extent));
    return std::move(For(n));
  }

  Var loop_var_;
  Var new_loop_var_;
  PrimExpr new_extent_;
  PrimExpr offset_;
  std::unordered_set<const BlockNode*> touched_blocks_;
  arith::Analyzer* analyzer_;
};

/*
 * Tries to recursively partition the range of the variable (given by var) of
 * the for loop (given by node and stmt) into a
 * number of disjoint ranges such that in some ranges one or more predicates
 * in the loopnest are provably true or false in each range. For example, given the
 * following loop to partition:
 * for (i = 0; i < 4; i++)
 *    for (j = 0; j < 10; j++)
 *        if (likely(i*10 + j < 36))
 *            A[10*i+j] = B[10*i+j]
 *
 * We first partition range of i, i.e., [0,3] into subranges [0,2] and [3,3] because the
 * likely condition is always true for the first subrange but not always true for the
 * second subrange. Therefore, we'll have
 * for (i = 0; i < 3; i++)
 *    for (j = 0; j < 10; j++)
 *        if (likely(1))
 *           A[10*i+j] = B[10*i+j]
 * for (i = 0; i < 1; i++)
 *    for (j = 0; j < 10; j++)
 *        if (likely((i+3)*10 + j < 36))
 *            A[10*(i+3)+j] = B[10*(i+3)+j]
 * Which is simplified as:
 * for (i = 0; i < 3; i++)
 *    for (j = 0; j < 10; j++)
 *        A[10*i+j] = B[10*i+j]
 * for (j = 0; j < 10; j++) // loopnest 1
 *    if (likely(j < 6))
 *            A[30+j] = B[30+j]
 * Now, we recursively partition j in loopnest 1 into subranges [0,5] and [6,9] where the
 * condition is true for the first subrange and now always true for the second subrange.
 * for (j = 0; j < 6; j++)
 *    if (likely(1))
 *         A[30+j] = B[30+j]
 * for (j = 0; j < 4; j++) // loop 2
 *    if (likely(j < 0))
 *        A[36+j] = B[36+j]
 * Finally we recursively partition loop 2 above into subrange [0,3] where the
 * condition is false and empty interval where the condition is not false,
 * therefore we generate
 * for (j = 0; j < 4; j++)
 *    if (likely(0))
 *        A[36+j] = B[36+j]
 * which will eventually be simplified to empty code. And because only one loop was generated
 * from loop 2 we stop recursing.
 */
Stmt LoopPartitioner::TryPartition(const Stmt& stmt, Var var, PrimExpr min, PrimExpr max,
                                   Stmt body) {
  using arith::IntervalSet;

  // include hint of var.
  hint_map_.insert({var.get(), IntSet::Interval(min, max)});
  PartitionFinder finder(var, hint_map_, relax_map_);
  finder(body);
  hint_map_.erase(var.get());

  if (finder.partitions.empty()) return Stmt();

  arith::IntervalSet for_interval(min, max);
  bool cond_value;
  IntSet middle_interval;
  ExpressionSet cond_set;
  // find an interval in which all conditions on var are true
  std::tie(middle_interval, cond_set) =
      GetIntervalAndCondset(finder.partitions, for_interval, true);
  if (middle_interval.IsNothing()) {
    // if such interval doesn't exist, find an interval in which all
    // conditions on var are false
    std::tie(middle_interval, cond_set) =
        GetIntervalAndCondset(finder.partitions, for_interval, false);
    if (middle_interval.IsNothing()) {
      // we couldn't find an interval in which the conditions are provably true or false
      // Therefore, we can't partition the loop based on those conds
      return Stmt();
    }
    cond_value = false;
  } else {
    cond_value = true;
  }

  IntervalSet middle_interval_i = Downcast<IntervalSet>(middle_interval);

  // if we get a trivial interval match full range of loop, just eliminate conditions and continue.
  if ((!middle_interval_i->HasLowerBound() ||
       analyzer_.CanProve(middle_interval_i.min() <= for_interval.min())) &&
      (!middle_interval_i->HasUpperBound() ||
       analyzer_.CanProve(middle_interval_i.max() >= for_interval.max()))) {
    Stmt simplified = ConditionEliminator(cond_set, cond_value)(stmt);
    const ForNode* simplified_loop = simplified.as<ForNode>();
    ICHECK(simplified_loop);
    return VisitLoopBody(simplified_loop);
  }

  // middle_interval is the subrange of the loop variable range for which a
  // set of conditions are true (or false resp.)
  // The part of the loop variable range that is before (after resp.) that
  // subrange is prefixed with pre- (post- resp.)

  // Calculating pre-subrange and generating code for it.
  // pre-subrange = [min, body_begin)
  PrimExpr body_begin;
  Stmt pre_stmt;
  PrimExpr pre_extent;
  bool pre_stmt_recurse = true;
  bool gen_pre = false;
  if (middle_interval_i->HasLowerBound()) {
    body_begin = analyzer_.Simplify(middle_interval.min());
    if (!analyzer_.CanProve(body_begin == min)) {
      PrimExpr offset = analyzer_.Simplify(body_begin - min);
      PrimExpr cond = offset >= 0;
      bool cond_is_true = analyzer_.CanProve(cond);
      bool cond_is_false = analyzer_.CanProve(!cond);
      if (!cond_is_true) {
        if (!cond_is_false) {
          LOG(WARNING) << "Cannot prove: " << cond << ", when generating the pre doubt loop";
        }
        body_begin = analyzer_.Simplify(Max(body_begin, min));
        // stop recursing on this interval if we can't prove it has non-negative length
        pre_stmt_recurse = false;
      }
      if (!cond_is_false) {
        gen_pre = true;
        pre_extent = offset;
      }
    }
  } else {
    body_begin = min;
  }

  // Calculating post-subrange and generating code for it.
  // post-subrange = [post_doubt_begin, max+1)
  PrimExpr post_doubt_begin;
  PrimExpr post_extent;
  Stmt post_stmt;
  bool post_stmt_recurse = true;
  bool gen_post = false;
  if (middle_interval_i->HasUpperBound()) {
    post_doubt_begin = analyzer_.Simplify(middle_interval.max() + 1);
    if (!analyzer_.CanProve(middle_interval.max() == max)) {
      // require the extent to be non-negative
      PrimExpr offset = analyzer_.Simplify(max - post_doubt_begin + 1);
      PrimExpr cond = offset >= 0;
      bool cond_is_true = analyzer_.CanProve(cond);
      bool cond_is_false = analyzer_.CanProve(!cond);
      if (!cond_is_true) {
        if (!cond_is_false) {
          LOG(WARNING) << "Cannot prove: " << cond << ", when generating the post doubt loop";
        }
        post_doubt_begin = analyzer_.Simplify(Min(post_doubt_begin, max + 1));
        // stop recursing on this interval if we can't prove it has non-negative length
        post_stmt_recurse = false;
      }
      if (!cond_is_false) {
        gen_post = true;
        post_extent = offset;
      }
    }
  } else {
    post_doubt_begin = max + 1;
  }

  // Generating code for middle subrange
  Stmt mid_stmt;
  bool gen_mid = false;
  if (!analyzer_.CanProve(body_begin >= post_doubt_begin)) {
    // [body_begin, post_doubt_begin)
    gen_mid = true;
  }

  LoopBodyRewriter rewriter(&analyzer_);
  if (gen_pre) {
    pre_stmt = rewriter.Rewrite(stmt, var, min, pre_extent);
  }
  if (gen_mid) {
    Stmt simplified_stmt = ConditionEliminator(cond_set, cond_value)(stmt);
    mid_stmt = rewriter.Rewrite(simplified_stmt, var, body_begin, post_doubt_begin - body_begin);
  }
  if (gen_post) {
    post_stmt = rewriter.Rewrite(stmt, var, post_doubt_begin, post_extent);
  }

  // Recurse for each non-empty subrange only if there are at least
  // two non-empty subranges
  if (gen_mid) {
    if (pre_stmt.defined() || post_stmt.defined()) {
      mid_stmt = VisitStmt(mid_stmt);
      if (pre_stmt.defined() && pre_stmt_recurse) {
        pre_stmt = VisitStmt(pre_stmt);
      }
      if (post_stmt.defined() && post_stmt_recurse) {
        post_stmt = VisitStmt(post_stmt);
      }
    }
  }

  Stmt res = SeqStmt::Flatten(pre_stmt, mid_stmt, post_stmt);
  res = ConvertSSA(res);
  return res;
}

/*
 *! \brief Refined partitioned loops.
 * (1) Update block itervar's dom and read/write region.
 * (2) Rewrite constant branches and likely tags.
 */
class RefinePartitioned : public StmtExprMutator {
 public:
  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::likely())) {
      ICHECK_EQ(op->args.size(), 1);
      PrimExpr cond = StmtExprMutator::VisitExpr(op->args[0]);
      cond = analyzer_.Simplify(cond);
      if (cond->IsInstance<IntImmNode>()) {
        return cond;
      } else {
        return StmtExprMutator::VisitExpr_(op);
      }
    } else if (op->op.same_as(builtin::if_then_else())) {
      ICHECK_EQ(op->args.size(), 3);
      PrimExpr cond = StmtExprMutator::VisitExpr(op->args[0]);
      PrimExpr simplified_cond = analyzer_.Simplify(cond);
      if (auto ptr = simplified_cond.as<IntImmNode>()) {
        if (ptr->value == 0) {
          return StmtExprMutator::VisitExpr(op->args[2]);
        } else {
          return StmtExprMutator::VisitExpr(op->args[1]);
        }
      } else {
        return std::move(Call(op->dtype, builtin::if_then_else(),
                              {simplified_cond, StmtExprMutator::VisitExpr(op->args[1]),
                               StmtExprMutator::VisitExpr(op->args[2])},
                              op->span));
      }
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    auto iter_vars = op->block->iter_vars;
    auto iter_bindings = op->iter_values;
    ICHECK_EQ(iter_vars.size(), iter_bindings.size());
    for (size_t i = 0; i < iter_vars.size(); ++i) {
      const auto& v = iter_vars[i]->var;
      auto iter_int_set = analyzer_.int_set(iter_bindings[i], dom_map_);
      Range iter_range = Range(iter_int_set.min(), iter_int_set.max() + 1);
      analyzer_.Bind(v, iter_range);
      dom_map_.Set(v, iter_int_set);
    }
    Stmt result;
    if (op->predicate.defined()) {
      PrimExpr cond = StmtExprMutator::VisitExpr(op->predicate);
      cond = analyzer_.Simplify(cond);
      auto n = CopyOnWrite(op);
      n->predicate = cond;
      result = std::move(BlockRealize(n));
    }
    result = StmtExprMutator::VisitStmt_(op);
    for (size_t i = 0; i < iter_vars.size(); ++i) {
      dom_map_.erase(iter_vars[i]->var);
    }
    return result;
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    // optimize body first
    auto n = CopyOnWrite(StmtExprMutator::VisitStmt_(op).as<BlockNode>());
    // update buffer accesses after partition
    Map<Var, Buffer> buffer_map;
    for (auto access : n->reads) {
      buffer_map.Set(access->buffer->data, access->buffer);
    }
    for (auto access : n->writes) {
      buffer_map.Set(access->buffer->data, access->buffer);
    }
    auto accesses = GetBlockAccessRegion(Block(n), buffer_map);
    n->reads = accesses[0];
    n->writes = accesses[1];

    // update iter doms
    for (size_t i = 0; i < n->iter_vars.size(); ++i) {
      auto iter_var = n->iter_vars[i];
      auto dtype = iter_var->var.dtype();
      auto bound = analyzer_.const_int_bound(iter_var->var);
      // if iter binding depends on free variables,
      // we can not get valid constant bound, skip to use old dom.
      // TODO(bxq): we only need to update dom for those depend
      // on target iter var which get partitioned.
      if (bound->min_value < 0) {
        continue;
      }
      auto new_dom = Range(IntImm(dtype, bound->min_value), IntImm(dtype, bound->max_value + 1));
      n->iter_vars.Set(i, IterVar(new_dom, iter_var->var, iter_var->iter_type, iter_var->thread_tag,
                                  iter_var->span));
    }
    return Block(n);
  }

  Stmt VisitStmt_(const ForNode* op) final {
    Range range = Range::FromMinExtent(op->min, op->extent);
    // assume ssa thus binded var should never be used out of scope
    analyzer_.Bind(op->loop_var, range);
    dom_map_.Set(op->loop_var, IntSet::FromRange(range));
    Stmt result = StmtExprMutator::VisitStmt_(op);
    dom_map_.erase(op->loop_var);
    return result;
  }

 private:
  // TODO(bxq): verify analyzer and dom_map usage.
  arith::Analyzer analyzer_;
  Map<Var, IntSet> dom_map_;
};

void LoopPartition(ScheduleState self, const Array<StmtSRef>& loop_srefs, bool lazy) {
  ICHECK(!loop_srefs.empty()) << "Empty input loops";
  if (lazy) {
    for (const StmtSRef& loop_sref : loop_srefs) {
      const ForNode* loop_node = loop_sref->StmtAs<ForNode>();
      CHECK(loop_node) << "TypeError: 'loop partition' expects a loop, but get type: "
                       << loop_sref->stmt->GetTypeKey();
      For new_loop =
          WithAnnotation(loop_node, attr::pragma_loop_partition_hint, IntImm(DataType::Int(32), 1));
      ReplaceStmt(self, loop_sref, new_loop, {});
    }
    return;
  }
  StmtSRef root_sref;
  if (loop_srefs.size() >= 2) {
    std::vector<StmtSRef> nodes(loop_srefs.begin(), loop_srefs.end());
    root_sref = LowestCommonAncestor(nodes, GetSRefTreeRoot(loop_srefs[0]));
  } else {
    root_sref = loop_srefs[0];
  }

  // configure candidate partition axes
  LoopPartitioner partitioner;
  for (const StmtSRef& loop_sref : loop_srefs) {
    const ForNode* loop_node = loop_sref->StmtAs<ForNode>();
    CHECK(loop_node) << "TypeError: 'loop partition' expects a loop, but get type: "
                     << loop_sref->stmt->GetTypeKey();
    partitioner.AddCandidate(loop_node);
  }
  // do partition
  Stmt partitioned_loops = partitioner(std::move(GetRef<Stmt>(root_sref->stmt)));
  partitioned_loops = RefinePartitioned()(std::move(partitioned_loops));
  ReplaceStmt(self, root_sref, partitioned_loops, {});
}

}  // namespace schedule
}  // namespace tir
}  // namespace tvm
