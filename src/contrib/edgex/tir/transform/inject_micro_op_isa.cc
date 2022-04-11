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
 * \brief Inject isa(instruction domain) of micro op(such as tiling).
 * \file inject_micro_op_isa.cc
 */
#include <tvm/arith/int_solver.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../attrs.h"
#include "../edgex_ir_utils.h"
#include "../op/builtin.h"

namespace tvm {
namespace tir {

using tvm::runtime::StorageRank;
using tvm::runtime::StorageScope;
using tvm::tir::edgex::GetValueByKey;

struct MicrOpConfigInfo {
  /*! \brief matched dma store buffer var. */
  const VarNode* matched_buf_var;
  /*! \brief the micro op configuration analysed according to the dma. */
  std::unordered_map<std::string, int32_t> config_vals;
};

// The white list dma scope, need analyze and inject isa.
static const std::set<std::string> intrin_hint_names{"idma", "odma"};

//  The isa assist to evaluate the injected isa value.
static const std::unordered_map<std::string, std::vector<std::string>> dma_isa_map{
    {"idma", {"num_group_idma"}},
    {"odma", {"psum_out_en_odma", "int_type_odma", "num_group_odma"}}};

static void UpdateConditionalDomain(const PrimExpr& condition, const Array<Var> loop_vars,
                                    Map<Var, Range>* dom_map, Map<Var, Range>* restore_map) {
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
  arith::IntConstraints constraint(loop_vars, *dom_map, equations);
  auto result = arith::SolveInequalitiesToRange(constraint);
  ICHECK(result->relations.empty()) << "Condition " << condition << " can not be fully solved";
  for (size_t i = 0; i < result->variables.size(); ++i) {
    const Var& var = result->variables[i];
    auto it = dom_map->find(var);
    if (it != dom_map->end()) {
      restore_map->Set(var, (*it).second);
    }
    dom_map->Set(var, result->ranges[var]);
  }
}

static int32_t GetParaLines(const DataType& dtype) {
  // idma input: int8/uint8/int16/fp16, odma input: int32/fp32
  if (dtype == DataType::Int(8) || dtype == DataType::UInt(8) || dtype == DataType::Int(32)) {
    return 16;
  } else if (dtype == DataType::Int(16) || dtype == DataType::Float(16) ||
             dtype == DataType::Float(32)) {
    return 8;
  } else {
    LOG(FATAL) << "Unsupported data type: " << dtype;
    return -1;
  }
}

class MicrOpAnalyzer : public StmtExprVisitor {
 public:
  void VisitStmt_(const AttrStmtNode* attr) final {
    if (attr->attr_key == attr::nnp_data_layout) {
      auto node = attr->value.as<StringImmNode>();
      ICHECK(node) << "Only support string type data layout value, but get type: "
                   << node->GetTypeKey();
      data_layout_ = node->value;
    } else if (attr->attr_key == attr::nnp_dma_scope) {
      if (const CallNode* n = attr->value.as<CallNode>()) {
        std::string op_name = Downcast<Op>(n->op)->name;
        size_t beg = op_name.find_first_of("_");
        size_t end = op_name.find_last_of("_");
        ICHECK(beg != std::string::npos && end > beg);
        intrin_hint_name_ = op_name.substr(beg + 1, end - beg - 1);
        if (intrin_hint_names.find(intrin_hint_name_) != intrin_hint_names.end()) {
          inject_isa_ = true;
          in_sp_scope_ = true;
          // the idma and odma should come in pairs
          ICHECK(pre_intrin_hint_name_.empty() || pre_intrin_hint_name_ != intrin_hint_name_)
              << "The previous intrinsic hint name should NE current one.";
          this->SetMapByKeyValue(n, intrin_hint_name_);
        }
      }
    }
    StmtExprVisitor::VisitStmt_(attr);
    if (in_sp_scope_) {
      pre_intrin_hint_name_ = intrin_hint_name_;
      this->ResetVar();
    }
  }

  void VisitStmt_(const ForNode* loop) final {
    if (in_sp_scope_) {
      loop_vars_.push_back(loop->loop_var);
      Range range = Range::FromMinExtent(loop->min, loop->extent);
      dom_map_.Set(loop->loop_var, range);
    }
    StmtExprVisitor::VisitStmt_(loop);
  }

  void VisitStmt_(const BufferStoreNode* store) final {
    if (in_sp_scope_) {
      const BufferLoadNode* load = store->value.as<BufferLoadNode>();
      Map<Var, Range> restore_map;
      if (load == nullptr) {
        if (intrin_hint_name_ == "idma") {
          // for idma, the load pattern could be padded
          const CallNode* ifthen = store->value.as<CallNode>();
          ICHECK(ifthen && ifthen->op.same_as(builtin::if_then_else()))
              << "Unsupported idma stmt " << GetRef<Stmt>(store);
          const PrimExpr& cond = ifthen->args[0];
          load = ifthen->args[1].as<BufferLoadNode>();
          UpdateConditionalDomain(cond, loop_vars_, &dom_map_, &restore_map);
          ICHECK(dom_map_.size() == restore_map.size()) << "Map size not match.";
        } else if (intrin_hint_name_ == "odma") {
          // for odma, the load pattern could be fused with post computations
          PostOrderVisit(store->value, [&load, this](const ObjectRef& obj) {
            if (const BufferLoadNode* cur_load = obj.as<BufferLoadNode>()) {
              StorageScope scope = GetStorageScope(cur_load->buffer->data);
              if (scope.rank == StorageRank::kCUBE) {
                load = cur_load;
              }
            }
          });
        }
      }
      ICHECK(load) << "Unsupported dma stmt, failed to fetch load pattern: \n"
                   << GetRef<Stmt>(store);
      const VarNode* buf_var = store->buffer->data.get();
      ICHECK(dma_config_map.count(buf_var) < 1)
          << "The store buffer is duplicated, store stmt: " << GetRef<Stmt>(store);
      if (intrin_hint_name_ == "idma") {
        pre_buf_var_ = buf_var;
      }
      DataType src_dtype = GetBufferElementType(load->buffer->data);
      this->EvaluateMicrOpConfigVal(restore_map, buf_var, src_dtype);
    }
    StmtExprVisitor::VisitStmt_(store);
  }

  /*! \brief detect whether need inject isa. */
  bool NeedInjectIsa() const { return inject_isa_; }
  /*! \brief dma hint name and config value map. */
  std::unordered_map<const VarNode*, MicrOpConfigInfo> dma_config_map;

 private:
  /*! \brief Collect the call node's isa value specified by the key. */
  void SetMapByKeyValue(const CallNode* call, const std::string& key) {
    auto it = dma_isa_map.find(key);
    ICHECK(it != dma_isa_map.end()) << "Not find key: " << key;
    const std::vector<std::string>& vec = it->second;
    for (auto vec_it = vec.begin(); vec_it != vec.end(); vec_it++) {
      int32_t val = GetValueByKey(call, *vec_it);
      ICHECK(val != -1) << "Not find isa: " << *vec_it;
      assist_isa_val_map_.emplace(*vec_it, val);
    }
  }

  /*! \brief get the dimension size. */
  int32_t EvaluateDimSizeSimple(const std::string& dim_name) {
    // if not find the dim name, will return 1.
    int32_t size{1};
    std::size_t pos{0};
    // need find all the pos of specified dimension name.
    while ((pos = data_layout_.find(dim_name, pos)) != std::string::npos) {
      Var loop_var = loop_vars_[pos];
      IntImm extent = Downcast<IntImm>(dom_map_[loop_var]->extent);
      ICHECK(extent.defined()) << "Domain extent for " << loop_var << " should be constant";
      size *= extent->value;
      pos++;
    }
    return size;
  }

  void GetPadding(const Map<Var, Range>& restore_map, const size_t& pos, int32_t* pad1,
                  int32_t* pad2) {
    ICHECK(pos < loop_vars_.size()) << "The pos should LT loop size.";
    if (restore_map.empty()) {
      *pad1 = 0;
      *pad2 = 0;
    } else {
      Var loop_var = loop_vars_[pos];
      IntImm ori_extent = Downcast<IntImm>(dom_map_[loop_var]->extent);
      ICHECK(ori_extent.defined()) << "Domain extent for " << loop_var << " should be constant";
      IntImm ori_min = Downcast<IntImm>(dom_map_[loop_var]->min);
      ICHECK(ori_min.defined()) << "Domain min for " << loop_var << " should be constant";
      IntImm new_extent = Downcast<IntImm>(restore_map[loop_var]->extent);
      ICHECK(new_extent.defined()) << "Domain extent for " << loop_var << " should be constant";
      IntImm new_min = Downcast<IntImm>(restore_map[loop_var]->min);
      ICHECK(new_min.defined()) << "Domain min for " << loop_var << " should be constant";
      // padding front/top/left
      *pad1 = ori_min->value - new_min->value;
      int32_t ori_max = ori_min->value + ori_extent->value;
      int32_t new_max = new_min->value + new_extent->value;
      // padding back/down/right
      *pad2 = new_max - ori_max;
    }
  }

  int32_t GetOdmaOutputElementBytes(const DataType& dtype) {
    int32_t psum_out_en = assist_isa_val_map_["psum_out_en_odma"];
    int32_t int_type = assist_isa_val_map_["int_type_odma"];
    if (psum_out_en) {
      // odma output fp32/int32
      return 4;
    } else if (dtype == DataType::Float(32) && !psum_out_en) {
      // odma output fp16
      return 2;
    } else if (dtype == DataType::Int(32) && !psum_out_en && int_type) {
      // odma output int16
      return 2;
    } else if (dtype == DataType::Int(32) && !psum_out_en && !int_type) {
      // odma output int8
      return 1;
    } else {
      LOG(FATAL) << "Unsupported data type: " << dtype;
      return -1;
    }
  }

  /*! \brief evaluate the dma configuration. */
  void EvaluateMicrOpConfigVal(const Map<Var, Range>& restore_map, const VarNode* buf_var,
                               const DataType& src_dtype) {
    ICHECK(!dom_map_.empty()) << "The loop var and range should not be empty.";
    ICHECK(dom_map_.size() == data_layout_.length())
        << "The data layout not match the loop dimension."
        << " loop var size: " << dom_map_.size() << " , data layout: " << data_layout_;
    MicrOpConfigInfo cfg;
    // transform to lower
    std::transform(data_layout_.begin(), data_layout_.end(), data_layout_.begin(), ::tolower);
    // evaluate the channel, depth, height, width
    int32_t num_c_group = EvaluateDimSizeSimple("c");  // need double check
    if (data_layout_.find("g") == std::string::npos) {
      int32_t group = assist_isa_val_map_["num_group_" + intrin_hint_name_];
      ICHECK((group != 0) && (num_c_group % group == 0))
          << "invalid group: " << group << ", num channel: " << num_c_group;
      num_c_group /= group;
    }
    int32_t d_sz = EvaluateDimSizeSimple("d");
    int32_t h_sz = EvaluateDimSizeSimple("h");
    int32_t w_sz = EvaluateDimSizeSimple("w");
    if (intrin_hint_name_ == "idma") {
      cfg.config_vals.emplace("ci_d", d_sz);
      cfg.config_vals.emplace("ci_h", h_sz);
      cfg.config_vals.emplace("ci_w", w_sz);
    } else if (intrin_hint_name_ == "odma") {
      cfg.config_vals.emplace("co_d", d_sz);
      cfg.config_vals.emplace("co_h", h_sz);
      cfg.config_vals.emplace("co_w", w_sz);
    }
    if (intrin_hint_name_ == "idma") {
      // padding size on front, top, left, back, down and right.
      std::array<int32_t, 6U> padding{0};
      size_t pos{0};
      if ((pos = data_layout_.find("d")) != std::string::npos) {
        this->GetPadding(restore_map, pos, &padding[0], &padding[3]);
      }
      if ((pos = data_layout_.find("h")) != std::string::npos) {
        this->GetPadding(restore_map, pos, &padding[1], &padding[4]);
      }
      if ((pos = data_layout_.find("w")) != std::string::npos) {
        this->GetPadding(restore_map, pos, &padding[2], &padding[5]);
      }
      cfg.config_vals.emplace("pad_f", padding[0]);
      cfg.config_vals.emplace("pad_bh", padding[3]);
      cfg.config_vals.emplace("pad_t", padding[1]);
      cfg.config_vals.emplace("pad_b", padding[4]);
      cfg.config_vals.emplace("pad_l", padding[2]);
      cfg.config_vals.emplace("pad_r", padding[5]);
    }
    // evaluate offset value, maybe can use stride.
    // TODO(someone): maybe need inject in InjectCalculatedIsa.
    int32_t c_para_lines = GetParaLines(src_dtype);
    int32_t dtype_bytes = src_dtype.bytes();
    if (intrin_hint_name_ == "idma") {
      // ci_row_offset:
      // fp16 >= 8*ci_w*2
      // int8 >= 16*ci_w*1
      int32_t ci_row_offset = c_para_lines * w_sz * dtype_bytes;
      cfg.config_vals.emplace("ci_row_offset", ci_row_offset);
      // ci_dense_offset >= ci_h*ci_row_offset
      int32_t ci_dense_offset = h_sz * ci_row_offset;
      cfg.config_vals.emplace("ci_dense_offset", ci_dense_offset);
      // ci_ch_offset >= ci_d*ci_dense_offset
      int32_t ci_ch_offset = d_sz * ci_dense_offset;
      cfg.config_vals.emplace("ci_ch_offset", ci_ch_offset);
      // need double check the num_c_group.
      // group_offset:
      // fp16 >= ceil(num_ci_group/8)*ci_ch_offset
      // int8 >= ceil(num_ci_group/16)*ci_ch_offset
      int32_t group_offset =
          std::ceil(static_cast<float>(num_c_group) / c_para_lines) * ci_ch_offset;
      cfg.config_vals.emplace("group_offset", group_offset);
    } else if (intrin_hint_name_ == "odma") {
      int32_t out_elem_bytes = GetOdmaOutputElementBytes(src_dtype);
      // co_dense_offset:
      // fp32 + psum_out_en >= 8*co_w*co_h*4
      // int32 + psum_out_en >= 16*co_w*co_h*4
      // fp32 + !psum_out_en >= 8*co_w*co_h*2
      // int32 + !psum_out_en + int_type >= 16*co_w*co_h*2
      // int32 + !psum_out_en + !int_type >= 16*co_w*co_h*1
      int32_t co_dense_offset = c_para_lines * h_sz * w_sz * out_elem_bytes;
      cfg.config_vals.emplace("co_dense_offset", co_dense_offset);
      // co_ch_offset >= co_d * co_dense_offset
      int32_t co_ch_offset = d_sz * co_dense_offset;
      cfg.config_vals.emplace("co_ch_offset", co_ch_offset);
      // co_group_offset:
      // int8/int32: >= ceil(num_co_group/16)*co_ch_offset
      // fp16/fp32: >= ceil(num_co_group/8)*co_ch_offset
      int32_t co_group_offset =
          std::ceil(static_cast<float>(num_c_group) / c_para_lines) * co_ch_offset;
      cfg.config_vals.emplace("co_group_offset", co_group_offset);
    }
    // record the configuration.
    if (intrin_hint_name_ == "odma") {
      ICHECK(pre_buf_var_ && dma_config_map.count(pre_buf_var_))
          << "Not fetched previous idma store buffer var.";
      cfg.matched_buf_var = pre_buf_var_;
      dma_config_map[pre_buf_var_].matched_buf_var = buf_var;
      dma_config_map[pre_buf_var_].config_vals.insert(
          {{"co_d", d_sz}, {"co_h", h_sz}, {"co_w", w_sz}});
    }
    dma_config_map.emplace(buf_var, cfg);
  }

  /*! \brief reset some primary variable. */
  void ResetVar() {
    data_layout_.clear();
    if (intrin_hint_name_ == "odma") {
      pre_buf_var_ = nullptr;
    }
    intrin_hint_name_.clear();
    assist_isa_val_map_.clear();
    dom_map_.clear();
    loop_vars_.clear();
    in_sp_scope_ = false;
  }

  /*! \brief record whether need inject isa. */
  bool inject_isa_{false};
  /*! \brief record op's data layout. */
  std::string data_layout_;
  /*! \brief record current intrinsic hint name. */
  std::string intrin_hint_name_;
  /*! \brief record previous intrinsic hint name. */
  std::string pre_intrin_hint_name_;
  /*! \brief record isa and value. */
  std::unordered_map<std::string, int32_t> assist_isa_val_map_;
  /*! \brief iteration range for loop_vars. */
  Map<Var, Range> dom_map_;
  /*! \brief record loop vars. */
  Array<Var> loop_vars_;
  /*! \brief record whether in specific scope. */
  bool in_sp_scope_{false};
  /*! \brief record the dma(idma) store buffer varnode. */
  const VarNode* pre_buf_var_{nullptr};
};

class MicrOpIsaInjector : public StmtExprMutator {
 public:
  Stmt Injector(Stmt stmt) {
    analyzer_(stmt);
    if (analyzer_.NeedInjectIsa()) {
      return operator()(std::move(stmt));
    } else {
      return stmt;
    }
  }

  Stmt VisitStmt_(const AttrStmtNode* attr) final {
    const CallNode* n = attr->value.as<CallNode>();
    if (attr->attr_key == attr::nnp_dma_scope && n) {
      std::string op_name = Downcast<Op>(n->op)->name;
      size_t beg = op_name.find_first_of("_");
      size_t end = op_name.find_last_of("_");
      ICHECK(beg != std::string::npos && end > beg);
      std::string intrin_hint_name = op_name.substr(beg + 1, end - beg - 1);
      if (intrin_hint_names.find(intrin_hint_name) != intrin_hint_names.end()) {
        const VarNode* buf_var{nullptr};
        PostOrderVisit(attr->body, [&buf_var, this](const ObjectRef& obj) {
          if (const BufferStoreNode* store = obj.as<BufferStoreNode>()) {
            buf_var = store->buffer->data.get();
            ICHECK(buf_var) << "Fetch store buffer var failed, store: " << GetRef<Stmt>(store);
          }
        });
        // start inject the isa
        auto call_ptr = make_object<CallNode>(*n);
        this->InjectIsa(call_ptr.get(), buf_var, intrin_hint_name);
        return AttrStmt(attr->node, attr->attr_key, std::move(Call(call_ptr)), attr->body,
                        attr->span);
      }
    } else if (attr->attr_key == attr::nnp_data_layout) {
      // remove the data layout attr
      return VisitStmt(attr->body);
    }
    return StmtExprMutator::VisitStmt_(attr);
  }

 private:
  // Inject isa helper function.
  void InjectIsa(CallNode* call, const VarNode* buf_var, const std::string& dma_name) {
    ICHECK(analyzer_.dma_config_map.count(buf_var))
        << "Can't find the buffer var in the dma_config_map";
    const MicrOpConfigInfo& cfg = analyzer_.dma_config_map[buf_var];
    for (const auto& p : cfg.config_vals) {
      std::stringstream key;
      key << p.first << "_" << dma_name;
      edgex::NNPAddArg(call, key.str(), p.second);
    }
  }

  /*! \brief The analyzer of micro op. */
  MicrOpAnalyzer analyzer_;
};

Stmt InjectMicrOpIsa(Stmt stmt) { return MicrOpIsaInjector().Injector(std::move(stmt)); }

namespace transform {

Pass InjectMicrOpIsa() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = InjectMicrOpIsa(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.edgex.InjectMicrOpIsa", {});
}

TVM_REGISTER_GLOBAL("tir.edgex.transform.InjectMicrOpIsa").set_body_typed(InjectMicrOpIsa);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
