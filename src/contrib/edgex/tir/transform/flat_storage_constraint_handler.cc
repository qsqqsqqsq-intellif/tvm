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
 * \file flat_storage_constraint_handler.cc
 * \brief Handle the storage address or memory size constraits about flattened buffers.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <numeric>

#include "../../../../arith/int_operator.h"
#include "../../../../arith/ir_mutator_with_analyzer.h"
#include "../edgex_ir_utils.h"
#include "../op/builtin.h"
#include "./edgex_transform.h"

namespace tvm {
namespace tir {

/*! \brief record buffer requirements. */
struct BufferReq {
  int minimal_size{0};
  int align_bytes{0};
};

using edgex::GetValueByKey;
using edgex::IsNNPDMAIntrinsic;
using edgex::NNPGetDmaDst;
using edgex::NNPGetDmaSrc;

/*! \brief subclass to rewrite buffer allocations from requirements */
class BufferConstraintAdaptor : public StmtExprMutator {
 public:
  explicit BufferConstraintAdaptor(
      const std::unordered_map<const VarNode*, BufferReq>& requirements)
      : requirements_(requirements) {}

 private:
  Stmt VisitStmt_(const AllocateNode* alloc) final {
    if (alloc->extents.size() != 1) {
      return StmtExprMutator::VisitStmt_(alloc);
    }
    const BufferReq& requirement = requirements_.at(alloc->buffer_var.get());
    CHECK_EQ(alloc->extents.size(), 1U) << "Expect flat buffer allocation";
    auto n = CopyOnWrite(alloc);
    if (requirement.minimal_size > 0) {
      int elem_bytes = alloc->dtype.bytes() * alloc->dtype.lanes();
      int require_bytes = Align(requirement.minimal_size, elem_bytes) / elem_bytes;
      n->extents.Set(0, max(n->extents[0], require_bytes));
    }
    if (requirement.align_bytes > 0) {
      AttrStmt attr =
          AttrStmt(alloc->buffer_var, attr::storage_alignment, requirement.align_bytes, n->body);
      n->body = attr;
    }
    return std::move(Allocate(n));
  }

  Stmt VisitStmt_(const AttrStmtNode* attr) final {
    if (attr->attr_key == attr::storage_alignment) {
      return VisitStmt(attr->body);
    }
    return StmtExprMutator::VisitStmt_(attr);
  }

  const std::unordered_map<const VarNode*, BufferReq>& requirements_;
};

/*!
 * \brief collect requirements to buffer from memory accesses, and rewrite memory access info
 * simultaneously `tir.tvm_access_ptr(dtype, buffer, offset, extent, mask)` => denotes a buffer
 * access [lhs, rhs) = [buffer_base_address + offset, buffer_base_address + offset + extent) to
 * ensure both lhs and rhs take specified alignment requirement, should ensure (1) offset % align ==
 * 0, access offset should never be modified (2) buffer_base_address % algin == 0, lead to align
 * requirement to specific buffer (3) extent % align == 0, or else extent should be modified
 *
 * each buffer may take multiple accesses, so requirements should be aggregated.
 * eg, a 32B alignment access and a 48B alignment access to same buffer should induce a 96B LCM
 * alignment.
 */
class FlatMemoryAccessRewritter : public arith::IRMutatorWithAnalyzer {
 public:
  explicit FlatMemoryAccessRewritter(arith::Analyzer* analyzer)
      : arith::IRMutatorWithAnalyzer(analyzer) {}

  Stmt Rewrite(const Stmt& root) {
    Stmt updated = VisitStmt(root);
    return BufferConstraintAdaptor(requirements_)(updated);
  }

  const std::unordered_map<const VarNode*, BufferReq>& GetRequirements() const {
    return requirements_;
  }

 private:
  Stmt VisitStmt_(const AllocateNode* alloc) final {
    if (alloc->extents.size() == 1) {
      requirements_[alloc->buffer_var.get()] = BufferReq();
    }
    return arith::IRMutatorWithAnalyzer::VisitStmt_(alloc);
  }

  Stmt VisitStmt_(const AttrStmtNode* attr) final {
    if (attr->attr_key == attr::storage_alignment) {
      const VarNode* buffer_var = attr->node.as<VarNode>();
      BufferReq& cur = requirements_[buffer_var];
      CHECK(is_const_int(attr->value));
      int align_bytes = attr->value.as<IntImmNode>()->value;
      cur.align_bytes = cur.align_bytes == 0
                            ? align_bytes
                            : arith::LeastCommonMultiple(cur.align_bytes, align_bytes);
    }
    return arith::IRMutatorWithAnalyzer::VisitStmt_(attr);
  }

  PrimExpr VisitExpr_(const CallNode* call) final {
    const RelayExpr& op = call->op;
    if (!IsNNPDMAIntrinsic(op)) {
      return arith::IRMutatorWithAnalyzer::VisitExpr_(call);
    }
    const size_t dst_idx = 1;
    const size_t src_idx = 2;
    auto n = make_object<CallNode>(*call);
    auto src_access = NNPGetDmaSrc(call);
    auto dst_access = NNPGetDmaDst(call);
    if (op.same_as(edgex::builtin::nnp_bdma_load())) {
      // xxx_st_addr1/2 must align by 16B [3:0]=0
      // xxx_end_addr1/2 must align by 16B [3:0]=f
      n->args.Set(src_idx, RewriteAccessPtr(src_access, 16));
    } else if (op.same_as(edgex::builtin::nnp_idma_load())) {
      int ci_w = GetValueByKey(call, "ci_w_idma");
      CHECK(ci_w != -1) << "Get ci_w_idma value failed.";
      // int8: 16ci*1bytes; fp16: 8ci*2bytes
      int align = ci_w * 16;
      n->args.Set(src_idx, RewriteAccessPtr(src_access, align));
    } else if (op.same_as(edgex::builtin::nnp_odma_store())) {
      // odma dst dm take 128bytes alignment
      n->args.Set(dst_idx, RewriteAccessPtr(dst_access, 128));
    } else if (op.same_as(edgex::builtin::nnp_wdma_load())) {
      int cube_en = GetValueByKey(call, "cube_enable_wdma");
      CHECK(cube_en != -1) << "Get cube_enable_wdma value failed.";
      int rotate_en = GetValueByKey(call, "rotate_en_wdma");
      CHECK(rotate_en == 0) << "Not support rotate_en_wdma!=0.";
      // int8: 16ci*1bytes; fp16: 8ci*2bytes
      int align = 256 * (cube_en + 1);
      n->args.Set(src_idx, RewriteAccessPtr(src_access, align));
    }
    return std::move(Call(n));
  }

  PrimExpr RewriteAccessPtr(const CallNode* access_call, int align_bytes) {
    BufferReq requirement;
    requirement.align_bytes = align_bytes;

    DataType dtype = access_call->args[0].dtype();
    int elem_bytes = dtype.bits() * dtype.lanes() / 8;
    CHECK_EQ(elem_bytes * 8, dtype.bits() * dtype.lanes());
    CHECK(align_bytes % elem_bytes == 0)
        << "Illegal align bytes " << align_bytes << " for datatype " << dtype;

    PrimExpr begin = access_call->args[2] * elem_bytes;
    CHECK(analyzer_->CanProve(floormod(begin, align_bytes) == 0))
        << "Access offset do not align to " << align_bytes << " at " << GetRef<Call>(access_call);

    auto n = make_object<CallNode>(*access_call);
    PrimExpr extent = access_call->args[3] * elem_bytes;
    if (!analyzer_->CanProve(floormod(extent, align_bytes) == 0)) {
      extent = analyzer_->Simplify(floordiv(Align(extent, align_bytes), elem_bytes));
      n->args.Set(3, extent);
      arith::ConstIntBound bound = analyzer_->const_int_bound(extent);
      if (bound->max_value != arith::ConstIntBound::kPosInf && bound->max_value > 0) {
        requirement.minimal_size = bound->max_value * elem_bytes;
      } else {
        LOG(WARNING) << "Skip infer upper bound of extent " << extent;
      }
    }

    const VarNode* buffer_var = access_call->args[1].as<VarNode>();
    auto it = requirements_.find(buffer_var);
    CHECK(it != requirements_.end()) << "No allocation found for " << GetRef<Var>(buffer_var);
    BufferReq& cur = it->second;
    if (requirement.align_bytes > 0) {
      cur.align_bytes = cur.align_bytes == 0
                            ? requirement.align_bytes
                            : arith::LeastCommonMultiple(cur.align_bytes, requirement.align_bytes);
    }
    if (requirement.minimal_size > 0) {
      cur.minimal_size = std::max(cur.minimal_size, requirement.minimal_size);
    }
    return std::move(Call(n));
  }

  std::unordered_map<const VarNode*, BufferReq> requirements_;
};

namespace transform {

Pass FlatStorageConstraintHandler() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    arith::Analyzer analyzer;
    FlatMemoryAccessRewritter rewritter(&analyzer);
    n->body = rewritter.Rewrite(std::move(f->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.edgex.FlatStorageConstraintHandler", {});
}

TVM_REGISTER_GLOBAL("tir.edgex.transform.FlatStorageConstraintHandler")
    .set_body_typed(FlatStorageConstraintHandler);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
