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
 * \file inject_calculated_isa.cc
 */
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>

#include "../../../../tir/transforms/ir_utils.h"
#include "../attrs.h"
#include "../edgex_ir_utils.h"
#include "../op/builtin.h"
#include "edgex_transform.h"

namespace tvm {
namespace tir {

using tvm::tir::edgex::GetValueByKey;
using tvm::tir::edgex::NNPDataType;

// cube alpha
#define ALPHA 16
// cube beta
#define BETA(data_type) (((data_type) == 0 || (data_type) == 1) ? 16 : 8)
// cube gamma
#define GAMMA 16
// op mode conv, matmul
enum OpMode { CONV, MATMUL };
// para mode tile, co
enum ParaMode { TILE_PARA, CO_PARA };

// The isa key assist to calculate the injected isa's value.
static const std::unordered_map<std::string, std::vector<std::string>> dma_isa_map{
    {"idma",
     {"sparsity_en_idma", "num_ci_group_idma", "op_idma", "wino_en_idma", "para_mode_idma",
      "co_w_idma", "co_h_idma", "co_d_idma", "cube_enable_idma", "B_dim2_idma", "data_type_idma"}},
    {"wdma", {"A_dim1_wdma", "A_dim2_wdma", "k_size_wdma"}},
    {"odma",
     {"extract_2to1_odma", "num_group_odma", "data_type_odma", "psum_out_en_odma", "int_type_odma",
      "co_w_odma", "co_ch_offset_odma"}}};

// The isa no need inject
static const std::set<std::string> isa_inject_backup{"epsilon",     "epsilon_times", "delta",
                                                     "delta_times", "eps_ci_times",  "last_epsilon",
                                                     "last_delta"};

// Collect useful isa's value to calculate injected isa's value.
class AssistedIsaCollector : public StmtExprVisitor {
 public:
  // Collect isa value from call node.
  void VisitExpr_(const CallNode* call) final {
    auto op = call->op;
    if (op.same_as(edgex::builtin::nnp_idma_load())) {
      inject_loop_isa_ = true;
      this->SetMapByKeyValue(call, "idma");
      for (auto it = isa_inject_backup.begin(); it != isa_inject_backup.end(); it++) {
        int32_t val = GetValueByKey(call, *it + "_idma");
        if (val != -1) {
          blacklist_isa_val_map_.emplace(*it, val);
          isa_inject_blacklist_.emplace(*it);
        }
      }
    } else if (op.same_as(edgex::builtin::nnp_wdma_load())) {
      this->SetMapByKeyValue(call, "wdma");
    } else if (op.same_as(edgex::builtin::nnp_odma_store())) {
      this->SetMapByKeyValue(call, "odma");
    }
    StmtExprVisitor::VisitExpr_(call);
  }

  // Collect useful value from attr node.
  void VisitStmt_(const AttrStmtNode* attr) final {
    if (attr->attr_key == attr::nnp_num_co_scope) {
      const IntImmNode* n = attr->value.as<IntImmNode>();
      ICHECK(n != nullptr);
      int32_t num_co = n->value;
      exist_isa_val_map_["num_co"] = num_co;
    }
    StmtExprVisitor::VisitStmt_(attr);
  }

  /*! \brief Record whether need inject loop isa. */
  bool NeedInjectLoopIsa() const { return inject_loop_isa_; }

  /*! \brief The assisted isa key-value map. */
  std::unordered_map<std::string, int32_t> exist_isa_val_map_;
  /*! \brief The blacklist isa key-value map. */
  std::unordered_map<std::string, int32_t> blacklist_isa_val_map_;
  /*! \brief The blacklist isa no need inject. */
  std::set<std::string> isa_inject_blacklist_;

 private:
  // Collect the call node's isa value specified by the key.
  void SetMapByKeyValue(const CallNode* call, const std::string& key) {
    auto it = dma_isa_map.find(key);
    if (it != dma_isa_map.end()) {
      const std::vector<std::string>& vec = it->second;
      for (auto vec_it = vec.begin(); vec_it != vec.end(); vec_it++) {
        int32_t val = GetValueByKey(call, *vec_it);
        if (val != -1) {
          exist_isa_val_map_[*vec_it] = val;
        } else {
          LOG(ERROR) << "Not find isa: " << *vec_it;
          return;
        }
      }
    } else {
      LOG(ERROR) << "Not find key: " << key;
      return;
    }
  }

  /*! \brief Record whether need inject loop isa. */
  bool inject_loop_isa_{false};
};

class CalculatedIsaInjector : public StmtExprMutator {
 public:
  Stmt Injector(Stmt stmt) {
    // collect the assist isa's value.
    collector_(stmt);
    if (collector_.NeedInjectLoopIsa()) {
      // calculate the loop value.
      this->CalculateLoopVal();
      // calculate the odma constraint value.
      this->CalculateOdmaConstraintVal();
      // start inject the calculated isa.
      return operator()(std::move(stmt));
    } else {
      return stmt;
    }
  }

  PrimExpr VisitExpr_(const CallNode* call) final {
    if (edgex::IsNNPIntrinsic(call->op)) {
      auto updated = StmtExprMutator::VisitExpr_(call);
      return this->VisitNNPIntrinsic(updated.as<CallNode>());
    } else {
      return StmtExprMutator::VisitExpr_(call);
    }
  }

  Stmt VisitStmt_(const AttrStmtNode* attr) final {
    if (attr->attr_key == attr::nnp_num_co_scope) {
      return VisitStmt(attr->body);
    } else {
      return StmtExprMutator::VisitStmt_(attr);
    }
  }

 private:
  // Visit NNP intrinsic and inject the calculated isa.
  PrimExpr VisitNNPIntrinsic(const CallNode* call) {
    auto op = call->op;
    auto n = make_object<CallNode>(*call);
    if (op.same_as(edgex::builtin::nnp_bdma_load())) {
      InjectCalculatedIsa({"delta", "zeta", "dense", "epsilon_times", "delta_times", "zeta_times",
                           "dense_times", "last_delta", "last_zeta", "last_dense"},
                          n.get(), "bdma");
    } else if (op.same_as(edgex::builtin::nnp_cube())) {
      InjectCalculatedIsa({"epsilon", "delta", "zeta", "dense", "epsilon_times", "delta_times",
                           "zeta_times", "dense_times", "last_epsilon", "last_delta", "last_zeta",
                           "last_dense", "last_beta_remind"},
                          n.get(), "cube");
    } else if (op.same_as(edgex::builtin::nnp_idma_load())) {
      InjectCalculatedIsa({"epsilon", "delta", "zeta", "dense", "epsilon_times", "delta_times",
                           "zeta_times", "dense_times", "last_epsilon", "last_delta", "last_zeta",
                           "last_dense", "last_zeta_width", "eps_ci_times", "last_eps_ci_times",
                           "ub_ci_num", "wo_ci_num", "wo_d_num", "wo_h_num"},
                          n.get(), "idma");
    } else if (op.same_as(edgex::builtin::nnp_odma_store())) {
      InjectCalculatedIsa({"delta",
                           "zeta",
                           "dense",
                           "delta_times",
                           "zeta_times",
                           "dense_times",
                           "last_delta",
                           "last_zeta",
                           "last_dense",
                           "last_zeta_width",
                           "last_delta_co",
                           "zeta_offset",
                           "wino_zeta_add_en",
                           "init_xbar_wr_byte",
                           "last_xbar_co_times",
                           "last_xbar_pixel_times",
                           "last_xbar_wr_byte",
                           "last_xbar_cube_times",
                           "delta_times_transfer_co",
                           "delta_ch_offset"},
                          n.get(), "odma");
    } else if (op.same_as(edgex::builtin::nnp_wdma_load())) {
      InjectCalculatedIsa(
          {"epsilon", "delta", "zeta", "dense", "epsilon_times", "delta_times", "zeta_times",
           "dense_times", "last_epsilon", "last_delta", "last_zeta", "last_dense"},
          n.get(), "wdma");
    }
    return std::move(Call(n));
  }

  // Inject isa.
  void InjectCalculatedIsa(const std::vector<std::string>& isa_set, CallNode* call,
                           const std::string& dma_name) {
    if (isa_set.empty()) {
      LOG(INFO) << "The inject isa is empty.";
      return;
    }
    for (auto it = isa_set.begin(); it != isa_set.end(); it++) {
      auto map_it = inject_isa_val_map_.find(*it);
      if ((map_it != inject_isa_val_map_.end()) &&
          (collector_.isa_inject_blacklist_.find(*it) == collector_.isa_inject_blacklist_.end())) {
        int32_t val = map_it->second;
        auto to_hex = [](size_t i) -> std::string {
          std::stringstream ss;
          ss << "0x" << std::hex << i;
          return ss.str();
        };
        std::stringstream key;
        key << *it << "_" << dma_name;
        edgex::NNPAddArg(call, key.str(), to_hex(val));
      } else {
        ICHECK(collector_.isa_inject_blacklist_.find(*it) != collector_.isa_inject_blacklist_.end())
            << "Not find key: " << *it;
      }
    }
    return;
  }

  // Calculate epsilon region loop value.
  void CalculateEpsilonRegionLoop(const int32_t& op_mode) {
    // calculate epsilon and epsilon_times.
    int32_t sparsity_en = GetValueFromMap(collector_.exist_isa_val_map_, "sparsity_en_idma");
    int32_t kernel_size = GetValueFromMap(collector_.exist_isa_val_map_, "k_size_wdma");
    int32_t data_type = GetValueFromMap(collector_.exist_isa_val_map_, "data_type_idma");
    int32_t num_ci_group_a_dim2{0};
    int32_t eps_ci_times{1}, epsilon{0}, epsilon_times{0};
    if (op_mode == CONV) {
      int32_t num_ci_group = GetValueFromMap(collector_.exist_isa_val_map_, "num_ci_group_idma");
      int32_t num_ci_group_sparsity{0};
      if (sparsity_en == 1) {
        CHECK((num_ci_group % 2) == 0) << "The num_ci_group should be divided evenly by 2.";
        num_ci_group_sparsity = num_ci_group / 2;
        CHECK_LE(num_ci_group_sparsity, 16535);
      } else {
        num_ci_group_sparsity = num_ci_group;
        CHECK_LE(num_ci_group_sparsity, 32767);
      }
      num_ci_group_a_dim2 = num_ci_group_sparsity;
      // TODO(someone): eps_ci_times default set 1,
      // this can be optimized by winograd enable or not respectively.
      epsilon = eps_ci_times * kernel_size;
      epsilon_times =
          ((num_ci_group_sparsity + BETA(data_type) - 1) / BETA(data_type) + eps_ci_times - 1) /
          eps_ci_times;
    } else {
      // op is matmul.
      int32_t a_dim2 = GetValueFromMap(collector_.exist_isa_val_map_, "A_dim2_wdma");
      int32_t a_dim2_sparsity{0};
      if (sparsity_en == 1) {
        CHECK((a_dim2 % 2) == 0) << "The a_dim2 should be divided evenly by 2.";
        a_dim2_sparsity = a_dim2 / 2;
        CHECK_LT(a_dim2_sparsity, 4096);
      } else {
        a_dim2_sparsity = a_dim2;
        CHECK_LT(a_dim2_sparsity, 8192);
      }
      num_ci_group_a_dim2 = a_dim2_sparsity;
      // TODO(someone): epsilon can be optimized in valid range.
      epsilon = (a_dim2_sparsity + BETA(data_type) - 1) / BETA(data_type);
      eps_ci_times = epsilon;
      epsilon_times =
          ((a_dim2_sparsity + BETA(data_type) - 1) / BETA(data_type) + epsilon - 1) / epsilon;
    }
    auto it = collector_.blacklist_isa_val_map_.find("epsilon");
    if (it != collector_.blacklist_isa_val_map_.end()) {
      epsilon = it->second;
    }
    it = collector_.blacklist_isa_val_map_.find("epsilon_times");
    if (it != collector_.blacklist_isa_val_map_.end()) {
      epsilon_times = it->second;
    }
    it = collector_.blacklist_isa_val_map_.find("eps_ci_times");
    if (it != collector_.blacklist_isa_val_map_.end()) {
      eps_ci_times = it->second;
    }
    inject_isa_val_map_.emplace("epsilon", epsilon);
    inject_isa_val_map_.emplace("epsilon_times", epsilon_times);
    inject_isa_val_map_.emplace("eps_ci_times", eps_ci_times);
    // TODO(someone): double check
    inject_isa_val_map_.emplace("wo_ci_num", epsilon);
    // calculate last_beta_remind.
    int32_t last_beta_remind = num_ci_group_a_dim2 % BETA(data_type);
    inject_isa_val_map_.emplace("last_beta_remind", last_beta_remind);
    // calculate last_epsilon
    // TODO(someone): need double check the ci_k_size if is matmul.
    int32_t last_epsilon{0};
    if (epsilon_times == 1) {
      last_epsilon = epsilon;
    } else {
      int32_t ci_k_size =
          (num_ci_group_a_dim2 + BETA(data_type) - 1) / BETA(data_type) * kernel_size;
      last_epsilon = ((ci_k_size % epsilon) == 0) ? epsilon : (ci_k_size % epsilon);
    }
    it = collector_.blacklist_isa_val_map_.find("last_epsilon");
    if (it != collector_.blacklist_isa_val_map_.end()) {
      last_epsilon = it->second;
    }
    inject_isa_val_map_.emplace("last_epsilon", last_epsilon);
    // calculate ub_ci_num < ((epsilon_times - 1) * epsilon + last_epsilon) * 16/8.
    // TODO(someone): double check.
    int32_t ub_ci_num = 16;
    inject_isa_val_map_.emplace("ub_ci_num", ub_ci_num);
    // calculate last_eps_ci_times
    int32_t last_eps_ci_times{0};
    int32_t winograd_en = GetValueFromMap(collector_.exist_isa_val_map_, "wino_en_idma");
    if (op_mode == CONV && winograd_en == 1) {
      last_eps_ci_times =
          ((((num_ci_group_a_dim2 + BETA(data_type) - 1) / BETA(data_type)) % eps_ci_times) == 0)
              ? eps_ci_times
              : ((num_ci_group_a_dim2 + BETA(data_type) - 1) / BETA(data_type)) % eps_ci_times;
    } else {
      last_eps_ci_times = 1;
    }
    inject_isa_val_map_.emplace("last_eps_ci_times", last_eps_ci_times);
  }

  // Calculate delta region loop value.
  void CalculateDeltaRegionLoop(const int32_t& op_mode, const int32_t& para_mode,
                                const int32_t& pe_num) {
    // calculate delta and delta_times.
    // dense*zeta*delta*(winograd_en?16:1)*ALPHA*GAMMA<=16*256*4
    // assume max zeta is 16, max delta is 4, dense=1.
    int32_t delta{4}, delta_times{0};
    int32_t max_delta{0}, num_co_group_a_dim1{0};
    if (op_mode == CONV) {
      int32_t num_co = GetValueFromMap(collector_.exist_isa_val_map_, "num_co");
      int32_t num_group = GetValueFromMap(collector_.exist_isa_val_map_, "num_group_odma");
      CHECK((num_co % num_group) == 0) << "The num_co should be divided evenly by num_group.";
      int32_t num_co_group = num_co / num_group;
      num_co_group_a_dim1 = num_co_group;
      max_delta = (num_co_group + ALPHA - 1) / ALPHA;
      // TODO(someone): delta can be optimized in valid range.
      if (para_mode == TILE_PARA) {
        int32_t delta_thd = (num_co_group + ALPHA - 1) / ALPHA;
        while (delta > delta_thd) {
          delta--;
        }
        delta_times = ((num_co_group + ALPHA - 1) / ALPHA + delta - 1) / delta;
      } else {
        // is co para
        int32_t delta_thd = ((num_co_group + ALPHA - 1) / ALPHA + pe_num - 1) / pe_num;
        while (delta > delta_thd) {
          delta--;
        }
        delta_times = ((num_co_group + ALPHA - 1) / ALPHA + delta * pe_num - 1) / (delta * pe_num);
      }
    } else {
      // op is matmul.
      int32_t a_dim1 = GetValueFromMap(collector_.exist_isa_val_map_, "A_dim1_wdma");
      num_co_group_a_dim1 = a_dim1;
      max_delta = (a_dim1 + ALPHA - 1) / ALPHA;
      if (para_mode == TILE_PARA) {
        int32_t delta_thd = (a_dim1 + ALPHA - 1) / ALPHA;
        while (delta > delta_thd) {
          delta--;
        }
        delta_times = ((a_dim1 + ALPHA - 1) / ALPHA + delta - 1) / delta;
      } else {
        // is co para
        int32_t delta_thd = ((a_dim1 + ALPHA - 1) / ALPHA + pe_num - 1) / pe_num;
        while (delta > delta_thd) {
          delta--;
        }
        delta_times = ((a_dim1 + ALPHA - 1) / ALPHA + delta * pe_num - 1) / (delta * pe_num);
      }
    }
    auto it = collector_.blacklist_isa_val_map_.find("delta");
    if (it != collector_.blacklist_isa_val_map_.end()) {
      delta = it->second;
    }
    it = collector_.blacklist_isa_val_map_.find("delta_times");
    if (it != collector_.blacklist_isa_val_map_.end()) {
      delta_times = it->second;
    }
    inject_isa_val_map_.emplace("delta", delta);
    inject_isa_val_map_.emplace("delta_times", delta_times);
    // calculate last_delta.
    int32_t last_delta{0};
    if (delta_times == 1) {
      last_delta = delta;
    } else {
      if (para_mode == TILE_PARA) {
        last_delta = ((max_delta % delta) == 0) ? delta : (max_delta % delta);
      } else {
        last_delta = ((max_delta + pe_num - 1) / pe_num) % delta == 0
                         ? delta
                         : ((max_delta + pe_num - 1) / pe_num) % delta;
      }
    }
    it = collector_.blacklist_isa_val_map_.find("last_delta");
    if (it != collector_.blacklist_isa_val_map_.end()) {
      last_delta = it->second;
    }
    inject_isa_val_map_.emplace("last_delta", last_delta);
    // calculate last_delta_co.
    int32_t last_delta_co{0};
    if (para_mode == TILE_PARA) {
      last_delta_co = num_co_group_a_dim1 % ALPHA == 0 ? ALPHA : num_co_group_a_dim1 % ALPHA;
    } else {
      last_delta_co = num_co_group_a_dim1 % (ALPHA * pe_num) == 0
                          ? ALPHA * pe_num
                          : num_co_group_a_dim1 % (ALPHA * pe_num);
    }
    inject_isa_val_map_.emplace("last_delta_co", last_delta_co);
  }

  // Calculate zeta region loop value.
  void CalculateZetaRegionLoop(const int32_t& op_mode, const int32_t& para_mode,
                               const int32_t& pe_num) {
    // calculate zeta and zeta_times.
    // dense*zeta*delta*(winograd_en?16:1)*ALPHA*GAMMA<=16*256*4
    // assume max zeta is 16, max delta is 4, dense=1.
    int32_t zeta{16}, zeta_times{0};
    int32_t max_zeta{0}, co_wxco_h_b_dim2{0};
    if (op_mode == CONV) {
      int32_t co_w = GetValueFromMap(collector_.exist_isa_val_map_, "co_w_idma");
      int32_t co_h = GetValueFromMap(collector_.exist_isa_val_map_, "co_h_idma");
      int32_t extract_2to1_odma =
          GetValueFromMap(collector_.exist_isa_val_map_, "extract_2to1_odma");
      int32_t co_w_x2{0};
      if (extract_2to1_odma == 1) {
        co_w_x2 = co_w * 2;
      } else {
        co_w_x2 = co_w;
      }
      max_zeta = (co_h * co_w_x2 + GAMMA - 1) / GAMMA;
      co_wxco_h_b_dim2 = co_w_x2 * co_h;
      // TODO(someone): zeta can be optimized in valid range.
      if (para_mode == TILE_PARA) {
        int32_t zeta_thd = ((co_w_x2 * co_h + GAMMA - 1) / GAMMA + pe_num - 1) / pe_num;
        while (zeta > zeta_thd) {
          zeta--;
        }
        zeta_times = ((co_w_x2 * co_h + GAMMA - 1) / GAMMA + zeta * pe_num - 1) / (zeta * pe_num);
      } else {
        // is co para
        int32_t zeta_thd = (co_w_x2 * co_h + GAMMA - 1) / GAMMA;
        while (zeta > zeta_thd) {
          zeta--;
        }
        zeta_times = ((co_w_x2 * co_h + GAMMA - 1) / GAMMA + zeta - 1) / zeta;
      }
    } else {
      // op is matmul.
      int32_t b_dim2 = GetValueFromMap(collector_.exist_isa_val_map_, "B_dim2_idma");
      max_zeta = (b_dim2 + GAMMA - 1) / GAMMA;
      co_wxco_h_b_dim2 = b_dim2;
      if (para_mode == TILE_PARA) {
        int32_t zeta_thd = ((b_dim2 + GAMMA - 1) / GAMMA + pe_num - 1) / pe_num;
        while (zeta > zeta_thd) {
          zeta--;
        }
        zeta_times = ((b_dim2 + GAMMA - 1) / GAMMA + zeta * pe_num - 1) / (zeta * pe_num);
      } else {
        // is co para
        int32_t zeta_thd = (b_dim2 + GAMMA - 1) / GAMMA;
        while (zeta > zeta_thd) {
          zeta--;
        }
        zeta_times = ((b_dim2 + GAMMA - 1) / GAMMA + zeta - 1) / zeta;
      }
    }
    inject_isa_val_map_.emplace("zeta", zeta);
    inject_isa_val_map_.emplace("zeta_times", zeta_times);
    // TODO(someone): double check
    inject_isa_val_map_.emplace("wo_h_num", zeta);
    // calculate last zeta.
    int32_t last_zeta{0};
    if (zeta_times == 1) {
      last_zeta = zeta;
    } else {
      if (para_mode == TILE_PARA) {
        last_zeta = (max_zeta + pe_num - 1) / pe_num % zeta == 0
                        ? zeta
                        : (max_zeta + pe_num - 1) / pe_num % zeta;
      } else {
        last_zeta = (max_zeta % zeta) == 0 ? zeta : (max_zeta % zeta);
      }
    }
    inject_isa_val_map_.emplace("last_zeta", last_zeta);
    // calculate last_zeta_width
    int32_t last_zeta_width{0};
    int32_t winograd_en = GetValueFromMap(collector_.exist_isa_val_map_, "wino_en_idma");
    if (para_mode == TILE_PARA) {
      if (winograd_en == 0) {
        last_zeta_width = co_wxco_h_b_dim2 % (GAMMA * pe_num) == 0
                              ? (GAMMA * pe_num)
                              : co_wxco_h_b_dim2 % (GAMMA * pe_num);
      } else {
        last_zeta_width =
            ((co_wxco_h_b_dim2 % (GAMMA * pe_num) == 0 ? (GAMMA * pe_num)
                                                       : co_wxco_h_b_dim2 % (GAMMA * pe_num)) +
             3) /
            4;
      }
    } else {
      if (winograd_en == 0) {
        last_zeta_width = co_wxco_h_b_dim2 % GAMMA == 0 ? GAMMA : co_wxco_h_b_dim2 % GAMMA;
      } else {
        last_zeta_width =
            ((co_wxco_h_b_dim2 % GAMMA == 0 ? GAMMA : co_wxco_h_b_dim2 % GAMMA) + 3) / 4;
      }
    }
    inject_isa_val_map_.emplace("last_zeta_width", last_zeta_width);
  }

  // Calculate dense region loop value.
  void CalculateDenseRegionLoop() {
    // calculate dense and dense_times.
    // dense*zeta*delta*(winograd_en?16:1)*ALPHA*GAMMA<=16*256*4
    // assume max zeta is 16, max delta is 4, dense=1.
    // TODO(someone): dense can be optimized in valid range.
    // dense <= co_d
    int32_t co_d = GetValueFromMap(collector_.exist_isa_val_map_, "co_d_idma");
    int32_t dense = 1;
    int32_t dense_times = (co_d + dense - 1) / dense;
    inject_isa_val_map_.emplace("dense", dense);
    inject_isa_val_map_.emplace("dense_times", dense_times);
    // TODO(someone): double check
    inject_isa_val_map_.emplace("wo_d_num", dense);
    // calculate last_dense.
    int32_t last_dense = (co_d % dense == 0) ? dense : (co_d % dense);
    inject_isa_val_map_.emplace("last_dense", last_dense);
  }

  // Calculate the loop value.
  void CalculateLoopVal() {
    int32_t op_mode = GetValueFromMap(collector_.exist_isa_val_map_, "op_idma");
    // Calculate epsilon region loop value.
    this->CalculateEpsilonRegionLoop(op_mode);
    int32_t para_mode = GetValueFromMap(collector_.exist_isa_val_map_, "para_mode_idma");
    int32_t pe_num = GetValueFromMap(collector_.exist_isa_val_map_, "cube_enable_idma") + 1;
    // Calculate delta region loop value.
    this->CalculateDeltaRegionLoop(op_mode, para_mode, pe_num);
    // Calculate zeta region loop value.
    this->CalculateZetaRegionLoop(op_mode, para_mode, pe_num);
    // Calculate dense region loop value.
    this->CalculateDenseRegionLoop();
  }

  // Calculate the odma constraint value.
  void CalculateOdmaConstraintVal() {
    int32_t data_type = GetValueFromMap(collector_.exist_isa_val_map_, "data_type_odma");
    int32_t delta = GetValueFromMap(inject_isa_val_map_, "delta");
    int32_t para_mode = GetValueFromMap(collector_.exist_isa_val_map_, "para_mode_idma");
    int32_t cube_enable = GetValueFromMap(collector_.exist_isa_val_map_, "cube_enable_idma");
    int32_t delta_times_transfer_co{0};
    // calculate the delta_times_transfer_co.
    if (data_type == NNPDataType::FLOAT32) {
      delta_times_transfer_co = delta * 8;
    } else if (para_mode == TILE_PARA) {
      delta_times_transfer_co = delta * 16;
    } else if (para_mode == CO_PARA && cube_enable == 0) {
      delta_times_transfer_co = delta * 16;
    } else if (para_mode == CO_PARA && cube_enable == 1) {
      delta_times_transfer_co = delta * 32;
    } else if (para_mode == CO_PARA && cube_enable == 2) {
      delta_times_transfer_co = delta * 48;
    }
    inject_isa_val_map_.emplace("delta_times_transfer_co", delta_times_transfer_co);
    // calculate the last_xbar_cube_times.
    int32_t last_zeta_width = GetValueFromMap(inject_isa_val_map_, "last_zeta_width");
    int32_t last_delta_co = GetValueFromMap(inject_isa_val_map_, "last_delta_co");
    int32_t last_xbar_cube_times{0};
    if (para_mode == TILE_PARA && last_zeta_width > 32) {
      last_xbar_cube_times = 2;
    } else if (para_mode == TILE_PARA && last_zeta_width > 16) {
      last_xbar_cube_times = 1;
    } else if (para_mode == CO_PARA && last_delta_co > 32) {
      last_xbar_cube_times = 2;
    } else if (para_mode == CO_PARA && last_delta_co > 16) {
      last_xbar_cube_times = 1;
    }
    inject_isa_val_map_.emplace("last_xbar_cube_times", last_xbar_cube_times);
    // calculate the last_xbar_wr_byte.
    int32_t psum_out_en = GetValueFromMap(collector_.exist_isa_val_map_, "psum_out_en_odma");
    int32_t extract_2to1 = GetValueFromMap(collector_.exist_isa_val_map_, "extract_2to1_odma");
    int32_t last_xbar_wr_byte{0};
    if (psum_out_en == 1 && data_type == NNPDataType::FLOAT32 && extract_2to1 == 1 &&
        (last_zeta_width & 0x3) == 0) {
      last_xbar_wr_byte = 63;
    } else if (psum_out_en == 1 && data_type == NNPDataType::FLOAT32 && extract_2to1 == 1) {
      last_xbar_wr_byte = (last_zeta_width & 0x3) * 16 - 1;
    } else if (psum_out_en == 1 && data_type != NNPDataType::FLOAT32 && extract_2to1 == 1 &&
               (last_zeta_width & 0x1) == 0) {
      last_xbar_wr_byte = 63;
    } else if (psum_out_en == 1 && data_type != NNPDataType::FLOAT32 && extract_2to1 == 1) {
      last_xbar_wr_byte = 31;
    } else if (psum_out_en == 0 && extract_2to1 == 1 && (last_zeta_width & 0x7) == 0) {
      last_xbar_wr_byte = 63;
    } else if (psum_out_en == 0 && extract_2to1 == 1) {
      last_xbar_wr_byte = (last_zeta_width & 0x7) * 8 - 1;
    } else if (psum_out_en == 1 && data_type == NNPDataType::FLOAT32 &&
               (last_zeta_width & 0x3) == 0) {
      last_xbar_wr_byte = 127;
    } else if (psum_out_en == 1 && data_type == NNPDataType::FLOAT32) {
      last_xbar_wr_byte = (last_zeta_width & 0x3) * 32 - 1;
    } else if (psum_out_en == 1 && data_type != NNPDataType::FLOAT32 &&
               (last_zeta_width & 0x1) == 0) {
      last_xbar_wr_byte = 127;
    } else if (psum_out_en == 1 && data_type != NNPDataType::FLOAT32) {
      last_xbar_wr_byte = 63;
    } else if (psum_out_en == 0 && (last_zeta_width & 0x7) == 0) {
      last_xbar_wr_byte = 127;
    } else if (psum_out_en == 0) {
      last_xbar_wr_byte = (last_zeta_width & 0x7) * 16 - 1;
    }
    inject_isa_val_map_.emplace("last_xbar_wr_byte", last_xbar_wr_byte);
    // calculate the last_xbar_pixel_times.
    int32_t last_xbar_pixel_times{0};
    if (psum_out_en == 1 && data_type == NNPDataType::FLOAT32 &&
        ((last_zeta_width & 0xf) > 12 || (last_zeta_width & 0xf) == 0)) {
      last_xbar_pixel_times = 3;
    } else if (psum_out_en == 1 && data_type == NNPDataType::FLOAT32 &&
               (last_zeta_width & 0xf) > 8) {
      last_xbar_pixel_times = 2;
    } else if (psum_out_en == 1 && data_type == NNPDataType::FLOAT32 &&
               (last_zeta_width & 0xf) > 4) {
      last_xbar_pixel_times = 1;
    } else if (psum_out_en == 1 && data_type == NNPDataType::FLOAT32) {
      last_xbar_pixel_times = 0;
    } else if (psum_out_en == 1 && ((last_zeta_width & 0xf) > 14 || (last_zeta_width & 0xf) == 0)) {
      last_xbar_pixel_times = 7;
    } else if (psum_out_en == 1 && (last_zeta_width & 0xf) > 12) {
      last_xbar_pixel_times = 6;
    } else if (psum_out_en == 1 && (last_zeta_width & 0xf) > 10) {
      last_xbar_pixel_times = 5;
    } else if (psum_out_en == 1 && (last_zeta_width & 0xf) > 8) {
      last_xbar_pixel_times = 4;
    } else if (psum_out_en == 1 && (last_zeta_width & 0xf) > 6) {
      last_xbar_pixel_times = 3;
    } else if (psum_out_en == 1 && (last_zeta_width & 0xf) > 4) {
      last_xbar_pixel_times = 2;
    } else if (psum_out_en == 1 && (last_zeta_width & 0xf) > 2) {
      last_xbar_pixel_times = 1;
    } else if (psum_out_en == 1) {
      last_xbar_pixel_times = 0;
    } else if ((last_zeta_width & 0xf) > 8 || (last_zeta_width & 0xf) == 0) {
      last_xbar_pixel_times = 1;
    }
    inject_isa_val_map_.emplace("last_xbar_pixel_times", last_xbar_pixel_times);
    // calculate the last_xbar_co_times.
    int32_t int_type = GetValueFromMap(collector_.exist_isa_val_map_, "int_type_odma");
    int32_t last_xbar_co_times{0};
    if (data_type == NNPDataType::FLOAT32 &&
        ((last_delta_co & 0xf) > 8 || (last_delta_co & 0xf) == 0)) {
      last_xbar_co_times = 1;
    } else if (data_type != NNPDataType::FLOAT32 && psum_out_en == 0 && int_type == 1 &&
               ((last_delta_co & 0xf) > 8 || (last_delta_co & 0xf) == 0)) {
      last_xbar_co_times = 1;
    }
    inject_isa_val_map_.emplace("last_xbar_co_times", last_xbar_co_times);
    // calculate the init_xbar_wr_byte.
    int32_t zeta_times = GetValueFromMap(inject_isa_val_map_, "zeta_times");
    int32_t zeta = GetValueFromMap(inject_isa_val_map_, "zeta");
    int32_t init_xbar_wr_byte{0};
    if (psum_out_en == 1 && data_type == NNPDataType::FLOAT32 && last_zeta_width < 4 &&
        zeta_times == 1 && zeta == 1 && extract_2to1 == 1) {
      init_xbar_wr_byte = (last_zeta_width & 0x3) * 16 - 1;
    } else if (psum_out_en == 1 && last_zeta_width == 1 && zeta_times == 1 && zeta == 1 &&
               extract_2to1 == 1) {
      init_xbar_wr_byte = 31;
    } else if (psum_out_en == 0 && last_zeta_width < 8 && zeta_times == 1 && zeta == 1 &&
               extract_2to1 == 1) {
      init_xbar_wr_byte = (last_zeta_width & 0x7) * 8 - 1;
    } else if (extract_2to1 == 1) {
      init_xbar_wr_byte = 63;
    } else if (psum_out_en == 1 && data_type == NNPDataType::FLOAT32 && last_zeta_width < 4 &&
               zeta_times == 1 && zeta == 1) {
      init_xbar_wr_byte = (last_zeta_width & 0x3) * 32 - 1;
    } else if (psum_out_en == 1 && last_zeta_width == 1 && zeta_times == 1 && zeta == 1) {
      init_xbar_wr_byte = 63;
    } else if (psum_out_en == 0 && last_zeta_width < 8 && zeta_times == 1 && zeta == 1) {
      init_xbar_wr_byte = (last_zeta_width & 0x7) * 16 - 1;
    } else {
      init_xbar_wr_byte = 127;
    }
    inject_isa_val_map_.emplace("init_xbar_wr_byte", init_xbar_wr_byte);
    // get the wino_zeta_add_en.
    int32_t wino_en = GetValueFromMap(collector_.exist_isa_val_map_, "wino_en_idma");
    int32_t co_w = GetValueFromMap(collector_.exist_isa_val_map_, "co_w_odma");
    int32_t wino_zeta_add_en{0};
    if (wino_en == 1 && (para_mode == CO_PARA || (para_mode == TILE_PARA && cube_enable == 0)) &&
        co_w == 8) {
      wino_zeta_add_en = 0;
    } else if (wino_en == 1 && para_mode == TILE_PARA && cube_enable == 1 &&
               (co_w == 8 || co_w == 16 || co_w == 24)) {
      wino_zeta_add_en = 0;
    } else if (wino_en == 1 && para_mode == TILE_PARA && cube_enable == 2 &&
               (co_w == 112 || co_w == 40 || co_w == 32 || co_w == 8 || co_w == 16 || co_w == 24)) {
      wino_zeta_add_en = 0;
    } else if (wino_en == 1) {
      wino_zeta_add_en = 1;
    }
    inject_isa_val_map_.emplace("wino_zeta_add_en", wino_zeta_add_en);
    // calculate the zeta_offset.
    int32_t zeta_offset{0};
    if (wino_en == 1 && (para_mode == CO_PARA || (para_mode == TILE_PARA && cube_enable == 0))) {
      zeta_offset = 32;
    } else if (wino_en == 1 && para_mode == TILE_PARA && cube_enable == 1) {
      zeta_offset = 64;
    } else if (wino_en == 1 && para_mode == TILE_PARA && cube_enable == 2) {
      zeta_offset = 96;
    } else if (psum_out_en == 0 &&
               (para_mode == CO_PARA || (para_mode == TILE_PARA && cube_enable == 0)) &&
               extract_2to1 == 1) {
      zeta_offset = 8;
    } else if (((psum_out_en == 0 && para_mode == TILE_PARA && cube_enable == 1) ||
                (psum_out_en == 1 && data_type == NNPDataType::FLOAT32)) &&
               extract_2to1 == 1) {
      zeta_offset = 16;
    } else if (psum_out_en == 0 && para_mode == TILE_PARA && cube_enable == 2 &&
               extract_2to1 == 1) {
      zeta_offset = 24;
    } else if (psum_out_en == 1 && data_type != NNPDataType::FLOAT32 &&
               (para_mode == CO_PARA || (para_mode == TILE_PARA && cube_enable == 0)) &&
               extract_2to1 == 1) {
      zeta_offset = 32;
    } else if ((psum_out_en == 1 && data_type == NNPDataType::INT32 &&
                (para_mode == TILE_PARA && cube_enable == 1)) &&
               extract_2to1 == 1) {
      zeta_offset = 64;
    } else if (psum_out_en == 1 && data_type == NNPDataType::INT32 && para_mode == TILE_PARA &&
               cube_enable == 2 && extract_2to1 == 1) {
      zeta_offset = 96;
    } else if (psum_out_en == 0 &&
               (para_mode == CO_PARA || (para_mode == TILE_PARA && cube_enable == 0))) {
      zeta_offset = 16;
    } else if ((psum_out_en == 0 && para_mode == TILE_PARA && cube_enable == 1) ||
               (psum_out_en == 1 && data_type == NNPDataType::FLOAT32)) {
      zeta_offset = 32;
    } else if (psum_out_en == 0 && para_mode == TILE_PARA && cube_enable == 2) {
      zeta_offset = 48;
    } else if (psum_out_en == 1 && data_type == NNPDataType::INT32 &&
               (para_mode == CO_PARA || (para_mode == TILE_PARA && cube_enable == 0))) {
      zeta_offset = 64;
    } else if (psum_out_en == 1 && data_type == NNPDataType::INT32 &&
               (para_mode == TILE_PARA && cube_enable == 1)) {
      zeta_offset = 128;
    } else if (psum_out_en == 1 && data_type == NNPDataType::INT32 && para_mode == TILE_PARA &&
               cube_enable == 2) {
      zeta_offset = 192;
    }
    inject_isa_val_map_.emplace("zeta_offset", zeta_offset);
    // calculate the delta_ch_offset
    int32_t co_ch_offset = GetValueFromMap(collector_.exist_isa_val_map_, "co_ch_offset_odma");
    int32_t delta_ch_offset{0};
    if (((para_mode == CO_PARA && cube_enable == 0) || para_mode == TILE_PARA) &&
        data_type == NNPDataType::FLOAT32) {
      delta_ch_offset = co_ch_offset * 2;
    } else if (((para_mode == CO_PARA && cube_enable == 0) || para_mode == TILE_PARA) &&
               psum_out_en == 0 && int_type == 1) {
      delta_ch_offset = co_ch_offset * 2;
    } else if (para_mode == CO_PARA && cube_enable == 1 && psum_out_en == 0 && int_type == 1) {
      delta_ch_offset = co_ch_offset * 4;
    } else if (para_mode == CO_PARA && cube_enable == 2 && psum_out_en == 0 && int_type == 1) {
      delta_ch_offset = co_ch_offset * 6;
    } else if ((para_mode == CO_PARA && cube_enable == 0) || para_mode == TILE_PARA) {
      delta_ch_offset = co_ch_offset;
    } else if (para_mode == CO_PARA && cube_enable == 1) {
      delta_ch_offset = co_ch_offset * 2;
    } else {
      delta_ch_offset = co_ch_offset * 3;
    }
    inject_isa_val_map_.emplace("delta_ch_offset", delta_ch_offset);
  }

  // Get assisted value from map by specified key.
  int32_t GetValueFromMap(const std::unordered_map<std::string, int32_t>& isa_val_map,
                          const std::string& key) {
    auto it = isa_val_map.find(key);
    if (it != isa_val_map.end()) {
      return it->second;
    } else {
      LOG(ERROR) << "Not find key: " << key;
      return -1;
    }
  }

  /*! \brief The AssistedIsaCollector. */
  AssistedIsaCollector collector_;
  /*! \brief Assign loop key & value. */
  std::unordered_map<std::string, int32_t> inject_isa_val_map_;
  /*! \brief Assign epsilon value. */
  int32_t epsilon_{0};
  /*! \brief Assign delta value. */
  int32_t delta_{0};
  /*! \brief Assign zeta value. */
  int32_t zeta_{0};
  /*! \brief Assign dense value. */
  int32_t dense_{0};
  /*! \brief Assign epsilon times value. */
  int32_t epsilon_times_{0};
  /*! \brief Assign delta times value. */
  int32_t delta_times_{0};
  /*! \brief Assign zeta times value. */
  int32_t zeta_times_{0};
  /*! \brief Assign dense times value. */
  int32_t dense_times_{0};
  /*! \brief Assign group loop value. */
  int32_t group_loop_{0};
};

Stmt InjectCalculatedIsa(Stmt stmt) { return CalculatedIsaInjector().Injector(std::move(stmt)); }

namespace transform {

Pass InjectCalculatedIsa() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = InjectCalculatedIsa(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.edgex.InjectCalculatedIsa", {});
}

TVM_REGISTER_GLOBAL("tir.edgex.transform.InjectCalculatedIsa").set_body_typed(InjectCalculatedIsa);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
