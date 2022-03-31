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
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>
#include <unordered_set>

#include "../attrs.h"
#include "../edgex_ir_utils.h"
#include "../op/builtin.h"

// buckets num for each nlfc sub-table
#define NLFC_BUCKET_NUM 128

// size in bytes of one k sub-table or b sub-table
#define NLFC_TABLE_PART_BYTES (NLFC_BUCKET_NUM * 4)

namespace tvm {
namespace tir {
namespace edgex {

using tvm::runtime::NDArray;

class NlfcPreScheduleConverter : public StmtExprMutator {
 public:
  PrimFunc Rewrite(PrimFunc f) {
    auto* n = f.CopyOnWrite();
    n->body = VisitStmt(std::move(f->body));
    if (!buffer_map_.empty()) {
      Array<Var> nlfc_params;
      Array<NDArray> nlfc_arrs;
      Array<Var> origin_params = f->params;
      n->params.clear();
      for (const auto& kv : buffer_map_) {
        Var var(kv.first->name + "_param", PrimType(DataType::Handle()));
        n->params.push_back(var);
        n->buffer_map.Set(var, kv.first);
        nlfc_params.push_back(var);
        nlfc_arrs.push_back(kv.second);
      }
      for (const Var& v : origin_params) n->params.push_back(v);
      f = WithAttr(f, attr::kNlfcTableParams, nlfc_params);
      f = WithAttr(f, attr::kNlfcTableData, nlfc_arrs);
    }
    return f;
  }

 private:
  Stmt VisitStmt_(const BlockNode* op) final {
    std::unordered_set<const BufferNode*> cur;
    std::swap(cur, touched_nlfc_buffers_);
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    if (!touched_nlfc_buffers_.empty()) {
      auto n = CopyOnWrite(block.get());
      for (const BufferNode* buf : touched_nlfc_buffers_) {
        n->reads.push_back(BufferRegion::FullRegion(GetRef<Buffer>(buf)));
      }
      block = Block(n);
    }
    std::swap(cur, touched_nlfc_buffers_);
    return std::move(block);
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    auto dtype = op->dtype;
    if (dtype != DataType::Float(32) && dtype != DataType::Float(16)) {
      return StmtExprMutator::VisitExpr_(op);
    }
    if (Op::HasAttrMap(attr::kFEdgexGetNlfcOp)) {
      static auto attr_map = Op::GetAttrMap<FEdgexGetNlfcOp>(attr::kFEdgexGetNlfcOp);
      auto f_nlfc_op = attr_map.get(op->op, nullptr);
      if (f_nlfc_op != nullptr && op->op->IsInstance<OpNode>()) {
        // add nlfc table argument to floating point arithmetic call
        Op nlfc_op = f_nlfc_op(Downcast<Op>(op->op));
        ICHECK(Op::HasAttrMap(attr::kNlfcOpInfo));
        static auto info_map = Op::GetAttrMap<NlfcOpInfo>(attr::kNlfcOpInfo);
        ICHECK(info_map.count(nlfc_op)) << attr::kNlfcOpInfo << " not registered for " << nlfc_op;
        NlfcOpInfo nlfc_info = info_map[nlfc_op];
        ICHECK(nlfc_info.defined());
        Array<PrimExpr> new_args(op->args);
        Array<Buffer> nlfc_buffers =
            GetNlfcBuffers(nlfc_op->name, nlfc_info->table_keys, op->dtype);
        for (const Buffer& buffer : nlfc_buffers) {
          new_args.push_back(buffer->data);
          touched_nlfc_buffers_.insert(buffer.get());
        }
        auto new_call = Call(op->dtype, nlfc_op, new_args, op->span);
        return std::move(new_call);
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Array<Buffer> GetNlfcBuffers(const std::string& nlfc_op_name, const Array<String>& keys,
                               const DLDataType& dtype) {
    auto it = buffer_cache_.find(nlfc_op_name);
    if (it != buffer_cache_.end()) {
      return it->second;
    }
    Array<Buffer> nlfc_buffers;
    auto* func = tvm::runtime::Registry::Get("tvm.edgex.get_nlfc_table");
    ICHECK(func) << "Cannot find tvm.edgex.get_nlfc_table";
    for (const String& key : keys) {
      NDArray table = (*func)(nlfc_op_name, key, runtime::DLDataType2String(dtype));
      ICHECK(table.defined()) << "Can not find ndarray of nlfc table " << nlfc_op_name;
      ICHECK(table.DataType() == DataType::Int(8)) << "NLFC table should be reinterpret to int8";
      ICHECK_EQ(table.Shape().size(), 1) << "NLFC table should be flattened";
      ICHECK(table.Shape()[0] % NLFC_TABLE_PART_BYTES == 0 &&
             table.Shape()[0] / NLFC_TABLE_PART_BYTES >= 2)
          << "NLFC table size should be multiple of " << NLFC_TABLE_PART_BYTES;
      Buffer buffer = decl_buffer(
          /*shape=*/Array<PrimExpr>({make_const(DataType::Int(32), table.Shape()[0])}),
          /*dtype=*/DataType::Int(8),
          /*name=*/nlfc_op_name + "_" + key,
          /*storage_scope=*/"global");
      nlfc_buffers.push_back(buffer);
      buffer_map_[buffer] = table;
    }
    buffer_cache_.insert(it, {nlfc_op_name, nlfc_buffers});
    return nlfc_buffers;
  }

  /*! \brief map nlfc_op_name to nlfc buffer array */
  std::unordered_map<std::string, Array<Buffer>> buffer_cache_;

  /*! \brief current touched nlfc buffers under block */
  std::unordered_set<const BufferNode*> touched_nlfc_buffers_;

  /*! \brief decl buffer mapping to actual nlfc array */
  std::unordered_map<Buffer, NDArray, ObjectPtrHash, ObjectPtrEqual> buffer_map_;
};

class NlfcRewritter : public StmtExprMutator {
 public:
  PrimFunc Rewrite(PrimFunc f) {
    for (const auto& kv : f->buffer_map) {
      const Buffer& buffer = kv.second;
      buffer_info_[buffer->data] =
          std::make_pair(Downcast<IntImm>(buffer->shape[0])->value, buffer->dtype);
    }
    Stmt update_body = VisitStmt(f->body);
    if (nlfc_buffer_var_.defined()) {
      update_body = Allocate(nlfc_buffer_var_, DataType::Int(8),
                             {make_const(DataType::Int(32), NLFC_TABLE_PART_BYTES * 2)},
                             make_const(DataType::Bool(), 1), update_body);
    }
    f.CopyOnWrite()->body = update_body;
    return std::move(f);
  }

 private:
  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->op->IsInstance<OpNode>() && Op::HasAttrMap(attr::kNlfcOpInfo)) {
      Op nlfc_op = Downcast<Op>(op->op);
      static auto keys_map = Op::GetAttrMap<Array<String>>(attr::kNlfcOpInfo);
      if (keys_map.count(nlfc_op)) {
        return RewriteNlfcOp(op, nlfc_op);
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr RewriteNlfcOp(const CallNode* call, const Op& nlfc_op) {
    std::string inst_name = "nlfc";
    NlfcOpInfo nlfc_info;
    if (Op::HasAttrMap(attr::kNlfcOpInfo)) {
      static auto attr_map = Op::GetAttrMap<NlfcOpInfo>(attr::kNlfcOpInfo);
      nlfc_info = attr_map.get(nlfc_op, NlfcOpInfo());
    }
    PrimExpr nlfc_table_arg = call->args[1];
    if (const BroadcastNode* broadcast = nlfc_table_arg.as<BroadcastNode>()) {
      // recover vectorized table arg
      nlfc_table_arg = broadcast->value;
    }

    ICHECK(call->args.size() == 2 && nlfc_table_arg.dtype().is_handle() &&
           nlfc_table_arg->IsInstance<VarNode>())
        << "Only support nlfc_op(value, table_var): " << GetRef<Call>(call);
    Var nlfc_buffer_var = Downcast<Var>(nlfc_table_arg);
    auto it = buffer_info_.find(nlfc_buffer_var);
    ICHECK(it != buffer_info_.end()) << "Can not find buffer of " << nlfc_table_arg;
    auto size_and_dtype = it->second;

    // determine iter-mode
    int64_t total_bytes = size_and_dtype.first;
    ICHECK(total_bytes % NLFC_TABLE_PART_BYTES == 0 && total_bytes / NLFC_TABLE_PART_BYTES >= 2 &&
           size_and_dtype.second == DataType::Int(8));
    size_t iter_num = total_bytes / NLFC_TABLE_PART_BYTES - 1;
    ICHECK_GE(iter_num, 1);
    if (iter_num == 1) {
      // non-iter mode, the layout of buffer should be b,k
      // vidma for iter-non
      cur_nlfc_stmts_.push_back(SeqStmt({Evaluate(CreateNlfcVidma(nlfc_buffer_var, 0)), Stmt()}));

      // iter-non computation
      Var res_non = Var("nlfc_res", call->dtype);
      cur_nlfc_stmts_.push_back(
          LetStmt(res_non,
                  CreateNNPInlineAsmVcu("=&{vv},{vv}",
                                        GetInlineAsm(nlfc_info, call->dtype.bits(), "iter_non"), 16,
                                        call->dtype, {}, {call->args[0]}, {nlfc_buffer_var}),
                  Evaluate(0)));
      return res_non;
    } else {
      // iter mode, the layout of buffer should be b0,k,b1...,bn
      LOG(FATAL) << "not implemented";
      return GetRef<Call>(call);
    }
  }

  std::string GetInlineAsm(const NlfcOpInfo& nlfc_info, size_t elem_bits, const char* iter_type) {
    std::stringstream ss;
    ss << "nop.10\n";
    bool need_set_nlfth = nlfc_info->nlf_th_value != NlfcOpInfo::HW_DEFAULT_NLF_TH_VALUE ||
                          nlfc_info->nlf_th_sel != 0 || nlfc_info->nlf_th_mode != 0;
    if (need_set_nlfth) {
      ss << "set nlfth 0x" << std::hex << nlfc_info->nlf_th_value << " " << std::dec
         << nlfc_info->nlf_th_mode << " " << nlfc_info->nlf_th_sel << "\n";
    }
    ss << nlfc_info->inst_name << ".f" << elem_bits << "." << iter_type << " $0 $1\n";
    ss << "nop.20\n";
    if (need_set_nlfth) {
      ss << "set nlfth 0x" << std::hex << NlfcOpInfo::HW_DEFAULT_NLF_TH_VALUE << " 0 0\n";
    }
    return ss.str();
  }

  PrimExpr CreateNlfcVidma(const Var& src_buffer_var, size_t iter) {
    // first iter copy both k,b; other iters copy b only
    int32_t extent = iter == 0 ? 2 * NLFC_TABLE_PART_BYTES : NLFC_TABLE_PART_BYTES;
    int32_t src_offset = iter == 0 ? 0 : NLFC_TABLE_PART_BYTES * (iter + 1);
    int32_t dst_offset = 0;
    DataType dtype = DataType::Int(8);
    Call dm_access(
        DataType::Handle(), tvm::tir::builtin::tvm_access_ptr(),
        {tir::TypeAnnotation(dtype), src_buffer_var, src_offset, extent, StringImm("r")});
    if (!nlfc_buffer_var_.defined()) {
      nlfc_buffer_var_ = Var("nlfc_mem", PointerType(PrimType(DataType::Int(8)), "nlfcmem"));
    }
    Call nlfc_access(
        DataType::Handle(), tvm::tir::builtin::tvm_access_ptr(),
        {tir::TypeAnnotation(dtype), nlfc_buffer_var_, dst_offset, extent, StringImm("w")});
    Call vidma = Call(DataType::Void(), edgex::builtin::nnp_vidma_load_nlfc(),
                      {StringImm(DLDataType2String(dtype)), nlfc_access, dm_access});
    auto n = const_cast<CallNode*>(vidma.get());
    NNPAddArg(n, "dtype_vidma", 0);
    NNPAddArg(n, "start_addr_in_en_vidma", 1);
    NNPAddArg(n, "start_addr_out_en_vidma", 1);
    NNPAddArg(n, "cb_buf_vm_vidma", 1);
    NNPAddArg(n, "cb_buf_dm_vidma", 1);
    NNPAddArg(n, "nlfc_mem_en_vidma", 1);
    NNPAddArg(n, "j0_loop_sel_vidma", 3);
    NNPAddArg(n, "j1_loop_sel_vidma", 2);
    NNPAddArg(n, "j2_loop_sel_vidma", 1);
    NNPAddArg(n, "j3_loop_sel_vidma", 0);
    NNPAddArg(n, "j0_loop_num_vidma", 1);
    NNPAddArg(n, "j1_loop_num_vidma", 2);
    NNPAddArg(n, "j2_loop_num_vidma", 128);
    NNPAddArg(n, "j3_loop_num_vidma", 4);
    NNPAddArg(n, "j0_stridein_vidma", 1024);
    NNPAddArg(n, "j1_stridein_vidma", 512);
    NNPAddArg(n, "j2_stridein_vidma", 4);
    NNPAddArg(n, "j0_strideout_vidma", 1024);
    NNPAddArg(n, "j1_strideout_vidma", 512);
    NNPAddArg(n, "j2_strideout_vidma", 4);
    NNPAddArg(n, "wo_data_size_vm_vidma", 256);
    NNPAddArg(n, "wo_data_size_dm_vidma", 256);
    NNPAddArg(n, "ub_data_size_vm_vidma", 256);
    NNPAddArg(n, "ub_data_size_dm_vidma", 256);
    return std::move(vidma);
  }

  Stmt VisitStmt(const Stmt& stmt) final {
    Stmt res = StmtExprMutator::VisitStmt(stmt);
    if (cur_nlfc_stmts_.empty()) {
      return res;
    }
    // there are nlfc computation bindings
    res = MergeNest(cur_nlfc_stmts_, res);
    cur_nlfc_stmts_.clear();
    return res;
  }

  Stmt VisitStmt_(const AllocateNode* op) {
    buffer_info_[op->buffer_var] = {Downcast<IntImm>(op->extents[0])->value, op->dtype};
    return StmtExprMutator::VisitStmt_(op);
  }

  /*! \brief nest stmts for nlfc result computation */
  std::vector<Stmt> cur_nlfc_stmts_;

  /*! \brief buffer map */
  std::unordered_map<Var, std::pair<int64_t, DataType>, ObjectPtrHash, ObjectPtrEqual> buffer_info_;

  /*! \brief nlfc buffer var */
  Var nlfc_buffer_var_{ObjectPtr<Object>()};
};

namespace transform {

using namespace tvm::tir::transform;

Pass RewriteNlfc() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    NlfcRewritter rewritter;
    return rewritter.Rewrite(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.edgex.RewriteNlfc", {});
}

Pass ConvertFpToNlfc() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    NlfcPreScheduleConverter converter;
    return converter.Rewrite(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.edgex.ConvertFpToNlfc", {});
}

TVM_REGISTER_GLOBAL("tir.edgex.transform.RewriteNlfc").set_body_typed(RewriteNlfc);
TVM_REGISTER_GLOBAL("tir.edgex.transform.ConvertFpToNlfc").set_body_typed(ConvertFpToNlfc);

}  // namespace transform
}  // namespace edgex
}  // namespace tir
}  // namespace tvm
