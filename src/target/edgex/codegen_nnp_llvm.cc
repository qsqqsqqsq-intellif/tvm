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
 * \file codegen_nnp400llvm.cc
 * \brief NNP400 llvm code generator.
 */

#if (defined TVM_LLVM_VERSION)
#include <llvm/IR/IntrinsicsNNP.h>
#include <llvm/Support/CommandLine.h>

#include "../../contrib/edgex/tir/attrs.h"
#include "../../contrib/edgex/tir/edgex_ir_utils.h"
#include "../../contrib/edgex/tir/op/builtin.h"
#include "../../runtime/edgex/edgex_common.h"
#include "../../runtime/edgex/edgex_module.h"
#include "../build_common.h"
#include "../llvm/codegen_llvm.h"
#include "../source/codegen_source_base.h"
#include "./codegen_nnp_intrinsic_info.h"
#include "./ibus_intrinsic_gen.h"
#include "nnp400_main.h"

// TODO(bxq): can these macros be included by de-llvm include headers?
#define NNP400_ADDRESS_SPACE_CM 0
#define NNP400_ADDRESS_SPACE_NMI 1
#define NNP400_ADDRESS_SPACE_VM 100

#define NNP400_UNIFORM_ADDRESS_SPACE_OFFSET_DM 0x0
#define NNP400_UNIFORM_ADDRESS_SPACE_OFFSET_VM 0xb00000

namespace tvm {
namespace codegen {

using tvm::tir::edgex::GetValueByKey;
using tvm::tir::edgex::NNPGetDmaDst;
using tvm::tir::edgex::NNPGetDmaSrc;

static const char* BB[] = {
    "cu",     "idma",   "wdma",     "odma",   "bdma",      "eidma",     "ewdma",     "eodma",
    "ebdma",  "nu",     "ccm0",     "mbox0",  "mbox1",     "mbox2",     "mbox3",     "mbox4",
    "mbox5",  "mbox6",  "mbox7",    "pdma",   "extevent0", "extevent1", "extevent2", "extevent3",
    "debug",  "vcu0",   "vcu1",     "vcu2",   "vcu3",      "vidma0",    "vidma1",    "vidma2",
    "vidma3", "vodma0", "vodma1",   "vodma2", "vodma3",    "vpdma0",    "vpdma1",    "vpdma2",
    "vpdma3", "ccmvuc", "debugvuc", "vidma",  "vodma",     "pe"};

/*! \brief Field specification for setmode, -1 denotes unspecified */
struct NNPSetModeSpec {
  int fx2fp_rmode{-1};
  int fp2fx_rmode{-1};
  int asr_rmode{-1};
  int relu_mul{-1};
  int vld0_ext_mode_en{-1};
  int vld1_ext_mode_en{-1};
  int eltmul_relu{-1};
  int veltadd_relu{-1};

  static NNPSetModeSpec FromCall(const CallNode* call) {
    return NNPSetModeSpec{
        GetValueByKey(call, "fx2fp_rmode"),      GetValueByKey(call, "fp2fx_rmode"),
        GetValueByKey(call, "asr_rmode"),        GetValueByKey(call, "relu_mul"),
        GetValueByKey(call, "vld0_ext_mode_en"), GetValueByKey(call, "vld1_ext_mode_en"),
        GetValueByKey(call, "eltmul_relu"),      GetValueByKey(call, "veltadd_relu_mode"),
    };
  }

  bool valid() const {
    return fx2fp_rmode >= 0 || fp2fx_rmode >= 0 || asr_rmode >= 0 || relu_mul >= 0 ||
           vld0_ext_mode_en >= 0 || vld1_ext_mode_en >= 0 || eltmul_relu >= 0 || veltadd_relu >= 0;
  }

  static const NNPSetModeSpec HW_DEFAULT_SPEC;
  static const NNPSetModeSpec TVM_APP_DEFAULT_SPEC;
};

/*! \brief Hardware default setmode specs */
const NNPSetModeSpec NNPSetModeSpec::HW_DEFAULT_SPEC{
    /*fx2fp_rmode=*/3,      /*fp2fx_rmode=*/3,      /*asr_rmode=*/3,
    /*relu_mul=*/0,         /*vld0_ext_mode_en*/ 0,
    /*vld1_ext_mode_en*/ 0, /*eltmul_relu*/ 0,      /*veltadd_relu*/ 0};

/*! \brief TVM application default setmode specs*/
const NNPSetModeSpec NNPSetModeSpec::TVM_APP_DEFAULT_SPEC{
    /*fx2fp_rmode=*/3,      /*fp2fx_rmode=*/2,      /*asr_rmode=*/3,
    /*relu_mul=*/0,         /*vld0_ext_mode_en*/ 0,
    /*vld1_ext_mode_en*/ 0, /*eltmul_relu*/ 0,      /*veltadd_relu*/ 0};

// NNP400 llvm code generator.
class CodeGenNNP400LLVM : public CodeGenLLVM {
 public:
  CodeGenNNP400LLVM() { InitNNPIntrinsics(); }

  void AddFunction(const PrimFunc& f) final {
    // add function as void return value
    CodeGenNNP400LLVM::AddFunctionInternal(f, true);
  }

  void AddFunctionInternal(const PrimFunc& f, bool ret_void);

  void InitPassManagerBuilder(llvm::PassManagerBuilder* builder) final {
    // Additional optimization hook to tweak the builder.
  }

  std::unique_ptr<llvm::Module> Finish();

  // always use this instead of base class's Optimize() routine
  static void OptimizeModule(llvm::Module* module, llvm::TargetMachine* tm);

  llvm::Value* CreateWaitOp(bool wo, const char* bb) {
    auto BBBegin = &BB[0];
    auto BBEnd = &BB[sizeof(BB) / sizeof(BB[0])];

    auto bb_it = std::find_if(BBBegin, BBEnd, [bb](const char* var) { return !strcmp(bb, var); });
    if (bb_it == BBEnd) return nullptr;

    auto f = llvm::Intrinsic::getDeclaration(module_.get(), llvm::Intrinsic::nnp_wait, {});
    return builder_->CreateCall(f, {
                                       MakeValue(wo ? 1 : 0),
                                       MakeValue(static_cast<int32_t>((bb_it - BBBegin))),
                                   });
  }

  llvm::Value* CreateSyncOp(const char* bb, bool wo, const char* bb1) {
    auto BBBegin = &BB[0];
    auto BBEnd = &BB[sizeof(BB) / sizeof(BB[0])];

    auto bb_it = std::find_if(BBBegin, BBEnd, [bb](const char* var) { return !strcmp(bb, var); });
    if (bb_it == BBEnd) return nullptr;

    auto bb1_it = std::find_if(&BB[0], BBEnd, [bb1](const char* var) { return !strcmp(bb1, var); });
    if (bb1_it == BBEnd) return nullptr;

    auto f = llvm::Intrinsic::getDeclaration(module_.get(), llvm::Intrinsic::nnp_sync, {});
    return builder_->CreateCall(f, {
                                       MakeValue(static_cast<int32_t>((bb_it - BBBegin))),
                                       MakeValue(wo ? 1 : 0),
                                       MakeValue(static_cast<int32_t>((bb1_it - BBBegin))),
                                   });
  }

  /*!
   * \brief handle the sync intrinsic's arguments like:
   *  (1) if the argument size is 3:
   *      '{StringImm("eidma"), StringImm("wo"), StringImm("idma")}'
   *      or '{StringImm("eidma"), StringImm("wo"), StringImm("vidma0")}'
   *  (2) if the argument size is 4:
   *      '{StringImm("eidma"), StringImm("wo"), StringImm("vidma"), IntImm(0)}'
   */
  llvm::Value* CreateHandShakeOp(const CallNode* call) {
    if (!call) return nullptr;
    CHECK_GE(call->args.size(), 3U);

    auto bb_node = call->args[0].as<StringImmNode>();
    ICHECK(bb_node != nullptr) << "The sync instrinsic args0 expect a string."
                               << GetRef<Call>(call);
    std::string bb = bb_node->value;
    auto state_node = call->args[1].as<StringImmNode>();
    ICHECK(state_node != nullptr) << "The sync instrinsic args1 expect a string."
                                  << GetRef<Call>(call);
    bool wo = strcmp(state_node->value.c_str(), "wo") == 0;
    auto bb1_node = call->args[2].as<StringImmNode>();
    ICHECK(bb1_node != nullptr) << "The sync instrinsic args2 expect a string."
                                << GetRef<Call>(call);
    std::string bb1 = bb1_node->value;
    if (call->args.size() == 4U) {
      auto idx_node = call->args[3].as<IntImmNode>();
      ICHECK(idx_node != nullptr) << "The sync instrinsic args3 expect a integer."
                                  << GetRef<Call>(call);
      int idx = idx_node->value;
      bb1 += std::to_string(idx);
    } else if (call->args.size() > 4U) {
      LOG(FATAL) << "Not supported arguments size great than 4." << GetRef<Call>(call);
    }
    // If bb is vxdma, and the bb1 is vcu, need modify the vcu to cu.
    if ((bb.find("vidma") != std::string::npos || bb.find("vodma") != std::string::npos) &&
        bb1.find("vcu") != std::string::npos) {
      bb1 = "cu";
    }
    if (bb == "cu" || bb == "vcu") {
      return CreateWaitOp(wo, bb1.c_str());
    } else {
      return CreateSyncOp(bb.c_str(), wo, bb1.c_str());
    }
  }

  void VisitStmt_(const AllocateNode* op) override {
    auto storage_scope = runtime::StorageScope::Create(GetPtrStorageScope(op->buffer_var));
    switch (storage_scope.rank) {
      case runtime::StorageRank::kDM: {
        // DM buffer take no special address space and should never ld/st directly
        ::llvm::Value* buf =
            builder_->CreateIntToPtr(builder_->getInt64(NNP400_UNIFORM_ADDRESS_SPACE_OFFSET_DM),
                                     DTypeToLLVMType(op->dtype)->getPointerTo());
        ICHECK(!var_map_.count(op->buffer_var.get()));
        var_map_[op->buffer_var.get()] = buf;
        VisitStmt(op->body);
        break;
      }
      case runtime::StorageRank::kVM: {
        ::llvm::Value* buf = builder_->CreateIntToPtr(
            builder_->getInt64(NNP400_UNIFORM_ADDRESS_SPACE_OFFSET_VM),
            DTypeToLLVMType(op->dtype)->getPointerTo(NNP400_ADDRESS_SPACE_VM));
        ICHECK(!var_map_.count(op->buffer_var.get()));
        var_map_[op->buffer_var.get()] = buf;
        VisitStmt(op->body);
        break;
      }
      case runtime::StorageRank::kGlobal: {
        LOG(FATAL) << "Do not support allocate ddr memory from device";
      }
      default:
        CodeGenLLVM::VisitStmt_(op);
    }
  }

  void VisitStmt_(const AttrStmtNode* attr) {
    if (attr->attr_key == tvm::tir::attr::nnp_vcore_resource) {
      this->vcore_resource_ = attr->value;
    }
    return CodeGenLLVM::VisitStmt_(attr);
  }

  llvm::Value* CreateDMAOp(const std::vector<llvm::Intrinsic::ID> intrs, const CallNode* op) {
    return CreateDMAOp(intrs, {}, {}, op);
  }

  /**
   * ! \brief DMA intrinsic create utility
   *   \param intrs  corresponding llvm intrinsic sequence.
   *   \param required_keys  attribute names must be filled for tir intrinsic before codegen.
   *   \param bind_keys  additional attributes bind with llvm value in codegen.
   *   \param op  intrinsic call node.
   */
  llvm::Value* CreateDMAOp(const std::vector<llvm::Intrinsic::ID> intrs,
                           const std::set<std::string>& required_keys,
                           const std::map<std::string, llvm::Value*>& bind_keys,
                           const CallNode* op) {
    llvm::Value* I = nullptr;
    for (auto ID : intrs) {
      bool has_non_imm_arg = false;
      auto f = llvm::Intrinsic::getDeclaration(module_.get(), ID, {});
      std::vector<llvm::Value*> arg_values;
      auto& strParams = NNP400IntrinsicInfos[ID];
      for (auto param_name : strParams) {
        llvm::Value* llvm_param_value;
        auto prebind_param_it = bind_keys.find(param_name);
        if (prebind_param_it != bind_keys.end()) {
          llvm_param_value = prebind_param_it->second;
        } else {
          auto param_it =
              std::find_if(op->args.begin(), op->args.end(), [&param_name](const PrimExpr& arg) {
                auto param_node = arg.as<StringImmNode>();
                if (!param_node) return false;
                std::string param_str = param_node->value;
                auto pos = param_str.find(param_name);
                return pos == 0 && param_str.size() > pos && param_str[strlen(param_name)] == '=';
              });
          if (param_it == op->args.end()) {
            if (required_keys.find(param_name) != required_keys.end()) {
              LOG(FATAL) << "Require key " << param_name << " for intrinsic\n" << GetRef<Call>(op);
            }
            arg_values.push_back(MakeValue(0));
            continue;
          }
          auto param_str = std::string((*param_it).as<StringImmNode>()->value.c_str());
          auto equal_pos = param_str.find("=");
          ICHECK(equal_pos != std::string::npos) << "Illegal argument " << param_str;
          auto param_value = std::stoi(param_str.substr(equal_pos + 1), 0, 0);
          llvm_param_value = MakeValue(param_value);
        }
        if (!llvm::isa<llvm::Constant>(llvm_param_value)) {
          has_non_imm_arg = true;
        }
        arg_values.push_back(llvm_param_value);
      }
      if (has_non_imm_arg) {
        I = GenerateIBusIntrinsic(builder_.get(), f, arg_values);
      } else {
        I = builder_->CreateCall(f, arg_values);
      }
    }
    return I;
  }

  llvm::Value* CreateSimpleIntrinsic(llvm::Intrinsic::ID intr, const CallNode* op) {
    auto f = llvm::Intrinsic::getDeclaration(module_.get(), intr, {});
    std::vector<llvm::Value*> arg_value;
    for (PrimExpr arg : op->args) {
      arg_value.push_back(MakeValue(arg));
    }
    return builder_->CreateCall(f, arg_value);
  }

  /**
   * ! \brief extract vcu id from call arguments.
   */
  int GetIntrinsicVcuId(const CallNode* call, size_t arg_idx = 0) const {
    ICHECK_GT(call->args.size(), arg_idx);
    const IntImmNode* idx = call->args[arg_idx].as<IntImmNode>();
    ICHECK(idx != nullptr) << "vcu id expect a integer arg at " << arg_idx << "\n"
                           << GetRef<Call>(call);
    return idx->value;
  }

  /**
   * ! \brief extract vcu BB name from call arguments.
   */
  std::string GetIntrinsicVcuBBName(const CallNode* call, const std::string& funit,
                                    size_t arg_idx = 0) const {
    int idx = GetIntrinsicVcuId(call, arg_idx);
    if (idx >= 0) {
      return funit + std::to_string(idx);
    } else {
      return funit;  // all vcu
    }
  }

  /**
   *! \brief determine whether a expr is a tvm_access_ptr() to constant address offset.
   */
  bool IsConstBufferAddress(const CallNode* call) {
    if (!call) return false;
    if (!call->op.same_as(builtin::tvm_access_ptr())) return false;
    ICHECK_EQ(call->args.size(), 5U);
    if (IsDDRBuffer(call->args[1])) return false;
    return call->args[2]->IsInstance<IntImmNode>();
  }

  llvm::Value* CreateCube(const CallNode* op) {
    std::map<std::string, llvm::Value*> bind_keys;
    return CreateDMAOp(
        {llvm::Intrinsic::nnp_cube_loop_times, llvm::Intrinsic::nnp_cube_loop0,
         llvm::Intrinsic::nnp_cube_loop1, llvm::Intrinsic::nnp_cube_last_loop,
         llvm::Intrinsic::nnp_cube_k_size, llvm::Intrinsic::nnp_cube_bias_value,
         llvm::Intrinsic::nnp_cube_layer_burst, llvm::Intrinsic::nnp_cube_burst_size},
        {}, bind_keys, op);
  }

  llvm::Value* CreateEidmaLoad(const CallNode* op) {
    std::map<std::string, llvm::Value*> bind_keys;
    const CallNode* dst = NNPGetDmaDst(op);
    const CallNode* src = NNPGetDmaSrc(op);
    if (!IsConstBufferAddress(dst)) {
      DataType dtype = dst->args[0].dtype();
      int elem_bytes = dtype.bytes() * dtype.lanes();
      llvm::Value* dm_start_addr = VisitTVMAccessPtr(dst, true);
      llvm::Value* dm_end_addr =
          builder_->CreateAdd(dm_start_addr, MakeValue(dst->args[3] * elem_bytes));
      bind_keys["ei_start_addr1"] = dm_start_addr;
      bind_keys["ei_start_addr2"] = dm_start_addr;
      bind_keys["ei_end_addr1"] = dm_end_addr;
      bind_keys["ei_end_addr2"] = dm_end_addr;  // config addr1=addr2 if addr2 not used
    }
    if (!IsConstBufferAddress(src)) {
      llvm::Value* ddr_start_addr = VisitTVMAccessPtr(src, true);
      ICHECK_EQ(ddr_start_addr->getType(), builder_->getInt64Ty());
      llvm::Value* ei_src_addr_low = builder_->CreateTrunc(ddr_start_addr, builder_->getInt32Ty());
      llvm::Value* ei_src_addr_high = builder_->CreateTrunc(
          builder_->CreateLShr(ddr_start_addr, builder_->getInt64(32)), builder_->getInt32Ty());
      bind_keys["ei_src_addr"] = ei_src_addr_low;
      bind_keys["ei_src_addr_high"] = ei_src_addr_high;
    }
    return CreateDMAOp(
        {llvm::Intrinsic::nnp_eidma_layer_cfg0, llvm::Intrinsic::nnp_eidma_layer_cfg1,
         llvm::Intrinsic::nnp_eidma_layer_cfg2, llvm::Intrinsic::nnp_eidma_layer_cfg3,
         llvm::Intrinsic::nnp_eidma_layer_cfg4, llvm::Intrinsic::nnp_eidma_layer_cfg5,
         llvm::Intrinsic::nnp_eidma_layer_cfg6, llvm::Intrinsic::nnp_eidma_layer_cfg7,
         llvm::Intrinsic::nnp_eidma_layer_cfg8, llvm::Intrinsic::nnp_eidma_layer_cfg9,
         llvm::Intrinsic::nnp_eidma_layer_cfg10, llvm::Intrinsic::nnp_eidma_layer_cfg11,
         llvm::Intrinsic::nnp_eidma_layer_cfg12, llvm::Intrinsic::nnp_eidma_layer_cfg13,
         llvm::Intrinsic::nnp_eidma_layer_cfg14, llvm::Intrinsic::nnp_eidma_layer_burst},
        {}, bind_keys, op);
  }

  llvm::Value* CreateEodmaStore(const CallNode* op) {
    std::map<std::string, llvm::Value*> bind_keys;
    const CallNode* dst = NNPGetDmaDst(op);
    const CallNode* src = NNPGetDmaSrc(op);
    if (!IsConstBufferAddress(src)) {
      DataType dtype = src->args[0].dtype();
      int elem_bytes = dtype.bytes() * dtype.lanes();
      llvm::Value* dm_start_addr = VisitTVMAccessPtr(src, true);
      llvm::Value* dm_end_addr =
          builder_->CreateAdd(dm_start_addr, MakeValue(src->args[3] * elem_bytes));
      bind_keys["eo_start_addr1"] = dm_start_addr;
      bind_keys["eo_start_addr2"] = dm_start_addr;
      bind_keys["eo_end_addr1"] = dm_end_addr;
      bind_keys["eo_end_addr2"] = dm_end_addr;  // config addr1=addr2 if addr2 not used
    }
    if (!IsConstBufferAddress(dst)) {
      llvm::Value* ddr_start_addr = VisitTVMAccessPtr(dst, true);
      ICHECK_EQ(ddr_start_addr->getType(), builder_->getInt64Ty());
      llvm::Value* eo_dst_addr_low = builder_->CreateTrunc(ddr_start_addr, builder_->getInt32Ty());
      llvm::Value* eo_dst_addr_high = builder_->CreateTrunc(
          builder_->CreateLShr(ddr_start_addr, builder_->getInt64(32)), builder_->getInt32Ty());
      bind_keys["eo_dst_addr"] = eo_dst_addr_low;
      bind_keys["eo_dst_addr_high"] = eo_dst_addr_high;
    }
    return CreateDMAOp(
        {llvm::Intrinsic::nnp_eodma_layer_cfg0, llvm::Intrinsic::nnp_eodma_layer_cfg1,
         llvm::Intrinsic::nnp_eodma_layer_cfg2, llvm::Intrinsic::nnp_eodma_layer_cfg3,
         llvm::Intrinsic::nnp_eodma_layer_cfg4, llvm::Intrinsic::nnp_eodma_layer_cfg5,
         llvm::Intrinsic::nnp_eodma_layer_cfg6, llvm::Intrinsic::nnp_eodma_layer_cfg7,
         llvm::Intrinsic::nnp_eodma_layer_cfg8, llvm::Intrinsic::nnp_eodma_layer_cfg9,
         llvm::Intrinsic::nnp_eodma_layer_cfg10, llvm::Intrinsic::nnp_eodma_layer_cfg11,
         llvm::Intrinsic::nnp_eodma_layer_cfg12, llvm::Intrinsic::nnp_eodma_layer_burst},
        {}, bind_keys, op);
  }

  /*! \brief Config the ewdma load address,
   * normal mode: config ew_src_addr, ew_src_addr_high
   * sparse mode: config ew_src_addr, ew_src_addr_high, ew_src_index_addr, ew_src_index_addr_high
   * unzip mode: config ew_src_addr, ew_src_addr_high, ew_src_index_addr, ew_src_index_addr_high
   * TODO(fred): sparse and unzip mode
   */
  llvm::Value* CreateEwdmaLoad(const CallNode* op) {
    std::map<std::string, llvm::Value*> bind_keys;
    const CallNode* dst = NNPGetDmaDst(op);
    const CallNode* src = NNPGetDmaSrc(op);
    if (!IsConstBufferAddress(dst)) {
      DataType dtype = dst->args[0].dtype();
      int elem_bytes = dtype.bytes() * dtype.lanes();
      llvm::Value* dm_start_addr = VisitTVMAccessPtr(dst, true);
      llvm::Value* dm_end_addr =
          builder_->CreateAdd(dm_start_addr, MakeValue(dst->args[3] * elem_bytes));
      bind_keys["ew_start_addr1"] = dm_start_addr;
      bind_keys["ew_start_addr2"] = dm_start_addr;
      bind_keys["ew_end_addr1"] = dm_end_addr;
      bind_keys["ew_end_addr2"] = dm_end_addr;  // config addr1=addr2 if addr2 not used
    }
    if (!IsConstBufferAddress(src)) {
      llvm::Value* ddr_start_addr = VisitTVMAccessPtr(src, true);
      ICHECK_EQ(ddr_start_addr->getType(), builder_->getInt64Ty());
      llvm::Value* ew_src_addr_low = builder_->CreateTrunc(ddr_start_addr, builder_->getInt32Ty());
      llvm::Value* ew_src_addr_high = builder_->CreateTrunc(
          builder_->CreateLShr(ddr_start_addr, builder_->getInt64(32)), builder_->getInt32Ty());
      bind_keys["ew_src_addr"] = ew_src_addr_low;
      bind_keys["ew_src_addr_high"] = ew_src_addr_high;
    }
    return CreateDMAOp(
        {llvm::Intrinsic::nnp_ewdma_layer_cfg0, llvm::Intrinsic::nnp_ewdma_layer_cfg1,
         llvm::Intrinsic::nnp_ewdma_layer_cfg2, llvm::Intrinsic::nnp_ewdma_layer_cfg3,
         llvm::Intrinsic::nnp_ewdma_layer_cfg4, llvm::Intrinsic::nnp_ewdma_layer_cfg5,
         llvm::Intrinsic::nnp_ewdma_layer_cfg6, llvm::Intrinsic::nnp_ewdma_layer_cfg7,
         llvm::Intrinsic::nnp_ewdma_layer_cfg8, llvm::Intrinsic::nnp_ewdma_layer_cfg9,
         llvm::Intrinsic::nnp_ewdma_layer_cfg10, llvm::Intrinsic::nnp_ewdma_layer_cfg11,
         llvm::Intrinsic::nnp_ewdma_layer_burst},
        {}, bind_keys, op);
  }

  llvm::Value* CreateIdmaLoad(const CallNode* op) {
    std::map<std::string, llvm::Value*> bind_keys;
    return CreateDMAOp(
        {llvm::Intrinsic::nnp_idma_layer_cfg0, llvm::Intrinsic::nnp_idma_layer_cfg1,
         llvm::Intrinsic::nnp_idma_layer_cfg2, llvm::Intrinsic::nnp_idma_layer_cfg3,
         llvm::Intrinsic::nnp_idma_layer_cfg4, llvm::Intrinsic::nnp_idma_layer_cfg5,
         llvm::Intrinsic::nnp_idma_layer_cfg6, llvm::Intrinsic::nnp_idma_layer_cfg7,
         llvm::Intrinsic::nnp_idma_layer_cfg8, llvm::Intrinsic::nnp_idma_layer_addr0,
         llvm::Intrinsic::nnp_idma_layer_addr1, llvm::Intrinsic::nnp_idma_layer_addr2,
         llvm::Intrinsic::nnp_idma_layer_addr3, llvm::Intrinsic::nnp_idma_warmup_cfg,
         llvm::Intrinsic::nnp_idma_layer_ub, llvm::Intrinsic::nnp_idma_layer_burst},
        {}, bind_keys, op);
  }

  llvm::Value* CreateOdmaStore(const CallNode* op) {
    std::map<std::string, llvm::Value*> bind_keys;
    return CreateDMAOp(
        {llvm::Intrinsic::nnp_odma_layer_cfg0, llvm::Intrinsic::nnp_odma_layer_cfg1,
         llvm::Intrinsic::nnp_odma_layer_cfg2, llvm::Intrinsic::nnp_odma_layer_cfg3,
         llvm::Intrinsic::nnp_odma_layer_cfg4, llvm::Intrinsic::nnp_odma_layer_cfg5,
         llvm::Intrinsic::nnp_odma_layer_cfg6, llvm::Intrinsic::nnp_odma_layer_addr0,
         llvm::Intrinsic::nnp_odma_layer_addr1, llvm::Intrinsic::nnp_odma_layer_addr2,
         llvm::Intrinsic::nnp_odma_layer_ub, llvm::Intrinsic::nnp_odma_layer_burst,
         llvm::Intrinsic::nnp_odma_layer_cfg7},
        {}, bind_keys, op);
  }

  llvm::Value* CreateVidmaLoad(const CallNode* op) {
    std::map<std::string, llvm::Value*> bind_keys;
    const CallNode* dst = NNPGetDmaDst(op);
    const CallNode* src = NNPGetDmaSrc(op);
    if (!IsConstBufferAddress(dst)) {
      DataType dtype = dst->args[0].dtype();
      int elem_bytes = dtype.bytes() * dtype.lanes();
      llvm::Value* vm_start_addr = VisitTVMAccessPtr(dst, true);
      llvm::Value* vm_end_addr =
          builder_->CreateAdd(vm_start_addr, MakeValue(dst->args[3] * elem_bytes));
      bind_keys["cb_buf_start_addr_vm_vidma"] = vm_start_addr;
      bind_keys["cb_buf_end_addr_vm_vidma"] = vm_end_addr;
    }
    if (!IsConstBufferAddress(src)) {
      DataType dtype = src->args[0].dtype();
      int elem_bytes = dtype.bytes() * dtype.lanes();
      llvm::Value* dm_start_addr = VisitTVMAccessPtr(src, true);
      llvm::Value* dm_end_addr =
          builder_->CreateAdd(dm_start_addr, MakeValue(src->args[3] * elem_bytes));
      bind_keys["start_addr1_dm_vidma"] = dm_start_addr;
      bind_keys["start_addr2_dm_vidma"] = dm_start_addr;
      bind_keys["end_addr1_dm_vidma"] = dm_end_addr;
      bind_keys["end_addr2_dm_vidma"] = dm_end_addr;  // config addr1=addr2 if addr2 not used
    }
    return CreateDMAOp(
        {llvm::Intrinsic::nnp_vidma_layer_cfg0, llvm::Intrinsic::nnp_vidma_layer_cfg1,
         llvm::Intrinsic::nnp_vidma_layer_cfg2, llvm::Intrinsic::nnp_vidma_layer_cfg3,
         llvm::Intrinsic::nnp_vidma_layer_cfg4, llvm::Intrinsic::nnp_vidma_layer_cfg5,
         llvm::Intrinsic::nnp_vidma_layer_cfg6, llvm::Intrinsic::nnp_vidma_layer_cfg7,
         llvm::Intrinsic::nnp_vidma_layer_cfg8, llvm::Intrinsic::nnp_vidma_layer_cfg9,
         llvm::Intrinsic::nnp_vidma_layer_cfg10, llvm::Intrinsic::nnp_vidma_layer_cfg11,
         llvm::Intrinsic::nnp_vidma_layer_cfg12, llvm::Intrinsic::nnp_vidma_layer_burst},
        {}, bind_keys, op);
  }

  llvm::Value* CreateVodmaStore(const CallNode* op) {
    std::map<std::string, llvm::Value*> bind_keys;
    const CallNode* dst = NNPGetDmaDst(op);
    const CallNode* src = NNPGetDmaSrc(op);
    if (!IsConstBufferAddress(dst)) {
      DataType dtype = src->args[0].dtype();
      int elem_bytes = dtype.bytes() * dtype.lanes();
      llvm::Value* dm_start_addr = VisitTVMAccessPtr(dst, true);
      llvm::Value* dm_end_addr =
          builder_->CreateAdd(dm_start_addr, MakeValue(src->args[3] * elem_bytes));
      bind_keys["start_addr1_dm_vodma"] = dm_start_addr;
      bind_keys["start_addr2_dm_vodma"] = dm_start_addr;
      bind_keys["end_addr1_dm_vodma"] = dm_end_addr;
      bind_keys["end_addr2_dm_vodma"] = dm_end_addr;  // config addr1=addr2 if addr2 not used
    }
    if (!IsConstBufferAddress(src)) {
      DataType dtype = src->args[0].dtype();
      int elem_bytes = dtype.bytes() * dtype.lanes();
      llvm::Value* vm_start_addr = VisitTVMAccessPtr(src, true);
      llvm::Value* vm_end_addr =
          builder_->CreateAdd(vm_start_addr, MakeValue(src->args[3] * elem_bytes));
      bind_keys["cb_buf_start_addr_vm_vodma"] = vm_start_addr;
      bind_keys["cb_buf_end_addr_vm_vodma"] = vm_end_addr;
    }
    return CreateDMAOp(
        {llvm::Intrinsic::nnp_vodma_layer_cfg0, llvm::Intrinsic::nnp_vodma_layer_cfg1,
         llvm::Intrinsic::nnp_vodma_layer_cfg2, llvm::Intrinsic::nnp_vodma_layer_cfg3,
         llvm::Intrinsic::nnp_vodma_layer_cfg4, llvm::Intrinsic::nnp_vodma_layer_cfg5,
         llvm::Intrinsic::nnp_vodma_layer_cfg6, llvm::Intrinsic::nnp_vodma_layer_cfg7,
         llvm::Intrinsic::nnp_vodma_layer_cfg8, llvm::Intrinsic::nnp_vodma_layer_cfg9,
         llvm::Intrinsic::nnp_vodma_layer_cfg10, llvm::Intrinsic::nnp_vodma_layer_cfg11,
         llvm::Intrinsic::nnp_vodma_layer_cfg12, llvm::Intrinsic::nnp_vodma_layer_burst},
        {}, bind_keys, op);
  }

  llvm::Value* CreateWdmaLoad(const CallNode* op) {
    std::map<std::string, llvm::Value*> bind_keys;
    return CreateDMAOp(
        {llvm::Intrinsic::nnp_wdma_layer_cfg0, llvm::Intrinsic::nnp_wdma_layer_cfg1,
         llvm::Intrinsic::nnp_wdma_layer_cfg2, llvm::Intrinsic::nnp_wdma_layer_cfg3,
         llvm::Intrinsic::nnp_wdma_layer_addr0, llvm::Intrinsic::nnp_wdma_layer_addr1,
         llvm::Intrinsic::nnp_wdma_layer_addr2, llvm::Intrinsic::nnp_wdma_layer_addr3,
         llvm::Intrinsic::nnp_wdma_layer_addr4, llvm::Intrinsic::nnp_wdma_layer_cfg4,
         llvm::Intrinsic::nnp_wdma_layer_cfg5, llvm::Intrinsic::nnp_wdma_layer_cfg6,
         llvm::Intrinsic::nnp_wdma_warmup_cfg, llvm::Intrinsic::nnp_wdma_layer_ub,
         llvm::Intrinsic::nnp_wdma_layer_burst},
        {}, bind_keys, op);
  }

  llvm::Value* CreateBdmaLoad(const CallNode* op) {
    std::map<std::string, llvm::Value*> bind_keys;
    return CreateDMAOp({llvm::Intrinsic::nnp_bdma_addr1, llvm::Intrinsic::nnp_bdma_addr2,
                        llvm::Intrinsic::nnp_bdma_loop0, llvm::Intrinsic::nnp_bdma_loop1,
                        llvm::Intrinsic::nnp_bdma_last_loop, llvm::Intrinsic::nnp_bdma_layer_burst},
                       {}, bind_keys, op);
  }

  template <typename RetT>
  RetT WithSetModeScope(const NNPSetModeSpec& new_spec, const NNPSetModeSpec& old_spec,
                        const std::function<RetT()>& callback) {
    if (!new_spec.valid()) {
      return callback();
    }
    NNPSetModeSpec recover_spec;
    recover_spec.fx2fp_rmode = new_spec.fx2fp_rmode >= 0 ? old_spec.fx2fp_rmode : -1;
    recover_spec.fp2fx_rmode = new_spec.fp2fx_rmode >= 0 ? old_spec.fp2fx_rmode : -1;
    recover_spec.asr_rmode = new_spec.asr_rmode >= 0 ? old_spec.asr_rmode : -1;
    recover_spec.relu_mul = new_spec.relu_mul >= 0 ? old_spec.relu_mul : -1;
    recover_spec.vld0_ext_mode_en = new_spec.vld0_ext_mode_en >= 0 ? old_spec.vld0_ext_mode_en : -1;
    recover_spec.vld1_ext_mode_en = new_spec.vld1_ext_mode_en >= 0 ? old_spec.vld1_ext_mode_en : -1;
    recover_spec.eltmul_relu = new_spec.eltmul_relu >= 0 ? old_spec.eltmul_relu : -1;
    recover_spec.veltadd_relu = new_spec.veltadd_relu >= 0 ? old_spec.veltadd_relu : -1;
    CreateNNPSetMode(new_spec);
    RetT result = callback();
    CreateNNPSetMode(recover_spec);
    return result;
  }

  void CreateNNPSetMode(const NNPSetModeSpec& spec) {
    std::vector<llvm::Value*> mode_args(
        {builder_->getInt32(spec.fx2fp_rmode), builder_->getInt32(spec.fp2fx_rmode),
         builder_->getInt32(spec.asr_rmode), builder_->getInt32(spec.relu_mul),
         builder_->getInt32(spec.vld0_ext_mode_en), builder_->getInt32(spec.vld1_ext_mode_en),
         builder_->getInt32(spec.eltmul_relu), builder_->getInt32(spec.veltadd_relu)});
    auto setmode_intrin =
        llvm::Intrinsic::getDeclaration(module_.get(), llvm::Intrinsic::nnp_setmode, {});
    builder_->CreateCall(setmode_intrin, mode_args);
  }

  llvm::Value* CreateVeltadd(const CallNode* op) {
    NNPSetModeSpec spec = NNPSetModeSpec::FromCall(op);
    return WithSetModeScope<llvm::Value*>(spec, NNPSetModeSpec::TVM_APP_DEFAULT_SPEC, [this, op]() {
      ICHECK_GE(op->args.size(), 4);
      llvm::Value* vs0 = VisitExpr(op->args[0]);
      llvm::Value* vs1 = VisitExpr(op->args[1]);
      llvm::Value* vs2 = VisitExpr(op->args[2]);
      llvm::Value* vs3 = VisitExpr(op->args[3]);
      auto veltadd_intrin = llvm::Intrinsic::getDeclaration(
          module_.get(), llvm::Intrinsic::nnp_veltadd, {vs0->getType()});
      return builder_->CreateCall(veltadd_intrin, {vs0, vs1, vs2, vs3});
    });
  }

  llvm::Value* CreateVacccMaddRightShift(const CallNode* op) {
    auto f_support_elem_ty = [](llvm::Type* ty) {
      // interger ty safe castable to i32
      if (!ty->isIntegerTy()) return false;
      auto ity = llvm::cast<llvm::IntegerType>(ty);
      return ity->getBitWidth() <= 32;
    };
    ICHECK_GE(op->args.size(), 4);
    llvm::Value* vs0 = VisitExpr(op->args[0]);
    llvm::Value* vs1 = VisitExpr(op->args[1]);
    llvm::Value* vs2 = VisitExpr(op->args[2]);
    llvm::Type* vs0_ty = vs0->getType();
    llvm::Type* vs1_ty = vs1->getType();
    llvm::Type* vs2_ty = vs2->getType();
    ICHECK(llvm::isa<llvm::VectorType>(vs0_ty));
    ICHECK(llvm::isa<llvm::VectorType>(vs1_ty));
    ICHECK(llvm::isa<llvm::VectorType>(vs2_ty));
    llvm::Type* elem0_ty = llvm::cast<llvm::VectorType>(vs0_ty)->getArrayElementType();
    llvm::Type* elem1_ty = llvm::cast<llvm::VectorType>(vs1_ty)->getArrayElementType();
    llvm::Type* elem2_ty = llvm::cast<llvm::VectorType>(vs2_ty)->getArrayElementType();
    auto lanes = llvm::cast<llvm::VectorType>(vs0_ty)->getElementCount();
    llvm::Type* i32_vec_ty = llvm::VectorType::get(builder_->getInt32Ty(), lanes);
    llvm::Type* i8_vec_ty = llvm::VectorType::get(t_int8_, lanes);
    llvm::Type* res_ty = llvm::StructType::get(i8_vec_ty, i32_vec_ty, i32_vec_ty, i32_vec_ty);
    // maybe extent data to i32
    ICHECK(f_support_elem_ty(elem0_ty));
    if (elem0_ty != t_int32_) {
      vs0 = builder_->CreateIntCast(vs0, i32_vec_ty, true);
    }
    ICHECK(f_support_elem_ty(elem1_ty));
    if (elem1_ty != t_int32_) {
      vs1 = builder_->CreateIntCast(vs1, i32_vec_ty, true);
    }
    ICHECK(f_support_elem_ty(elem2_ty));
    if (elem2_ty != t_int32_) {
      vs2 = builder_->CreateIntCast(vs2, i32_vec_ty, true);
    }
    auto fty = llvm::FunctionType::get(res_ty, {i32_vec_ty, i32_vec_ty, i32_vec_ty}, false);
    const char* inline_asm =
        "nop.10\n"
        "vmov.s32 $1 1\n"
        "nop.10\n"
        "vsub.s32 $2 $6 $1\n"
        "nop.10\n"
        "vasl.s32 $2 $1 $2\n"
        "nop.10\n"
        "vmul.s32 $3 $4 $5\n"
        "nop.10\n"
        "vcmp.s32.ge vpp0 $3 $1\n"
        "nop.10\n"
        "vsub.s32.vpp0i $2 $2 $1\n"
        "nop.10\n"
        "vcmp.s32.ge vpp0 $6 $1\n"
        "nop.10\n"
        "vadd.s32.vpp0 $3 $3 $2\n"
        "nop.10\n"
        "vasr.s32 $0 $3 $6\n"
        "nop.10\n"
        "vint.s32ts8 $0 $0 0\n"
        "nop.10\n";
    auto val =
        llvm::InlineAsm::get(fty, inline_asm, "={vv},=&{vv},=&{vv},={vacc},{vv},{vv},{vv}", true);
    return builder_->CreateExtractValue(builder_->CreateCall(val, {vs0, vs1, vs2}), {0});
  }

  llvm::Value* CreateLockVcu(const CallNode* op) {
    auto ftype_get_resource = llvm::FunctionType::get(t_void_, {}, false);
    const char* GET_RESOURCE_FUNCNAME = "GET_MBX_LOCK_ASM__GET_RESOURCE";
    auto f_get_resource = module_->getOrInsertFunction(GET_RESOURCE_FUNCNAME, ftype_get_resource);
    builder_->CreateCall(f_get_resource, {});

    const char* CFG_VU_DESP_FUNCNAME = "CFG_VU_DESP_ASM__MAIN";
    auto ftype_cfg_vux = llvm::FunctionType::get(t_void_, {t_int32_, t_int32_}, false);
    auto f_cfg_vux = module_->getOrInsertFunction(CFG_VU_DESP_FUNCNAME, ftype_cfg_vux);
    llvm::Value* blx_pc_en = builder_->getInt32(0);
    llvm::Value* blx_pc = builder_->getInt32(0);
    return builder_->CreateCall(f_cfg_vux, {blx_pc_en, blx_pc});
  }

  llvm::Value* CreateUnlockVcu(const CallNode* op) {
    auto ftype_release_resource = llvm::FunctionType::get(t_void_, {}, false);
    const char* RELEASE_RESOURCE_FUNCNAME = "GET_MBX_LOCK_ASM__RELEASE_RESOURCE";
    auto f_release_resource =
        module_->getOrInsertFunction(RELEASE_RESOURCE_FUNCNAME, ftype_release_resource);
    return builder_->CreateCall(f_release_resource, {});
  }

  llvm::Value* VisitExpr_(const CallNode* op) {
    Op op_ref = Downcast<Op>(op->op);
    if (op_ref.defined()) {
      std::string op_name = op_ref->name;
      if (op_name.find("nnp") != std::string::npos) {
        return CreateNNPIntrinsic(op, op_ref);
      } else if (op->op.same_as(builtin::tvm_access_ptr())) {
        return VisitTVMAccessPtr(op, false);
      } else if (op->op.same_as(builtin::likely())) {
        return CodeGenLLVM::VisitExpr(op->args[0]);
      }
    }
    return CodeGenLLVM::VisitExpr_(op);
  }

  bool IsDDRBuffer(const PrimExpr& expr) const {
    if (const VarNode* buf_var = expr.as<VarNode>()) {
      auto storage_scope = runtime::StorageScope::Create(GetPtrStorageScope(GetRef<Var>(buf_var)));
      return storage_scope.rank == runtime::StorageRank::kGlobal;
    }
    return false;
  }

  /**
   * \brief Return address range of tvm access ptr call.
   * \param in_dma_intrinsic  if in dma call, return relative byte offset respect to storage
   *                          scope's base address. i64 for ddr and i32 for other scopes.
   */
  llvm::Value* VisitTVMAccessPtr(const CallNode* op, bool in_dma_intrinsic) {
    DataType i8_type = DataType::Int(8);
    DataType elemtype = op->args[0].dtype();
    llvm::Value* buffer = MakeValue(op->args[1]);
    bool is_ddr = IsDDRBuffer(op->args[1]);
    llvm::Value* index = MakeValue(op->args[2]);
    ICHECK(index->getType()->isIntegerTy());
    if (in_dma_intrinsic) {
      if (is_ddr) {
        llvm::Value* ddr_addr = CreateBufferPtr(buffer, elemtype, index, i8_type).addr;
        return builder_->CreatePtrToInt(ddr_addr, builder_->getInt64Ty());
      } else {
        int elem_bytes = op->args[0].dtype().bytes() * op->args[0].dtype().lanes();
        return builder_->CreateMul(index, llvm::ConstantInt::get(index->getType(), elem_bytes));
      }
    } else {
      return CreateBufferPtr(buffer, elemtype, index, i8_type).addr;
    }
  }

  /**
   * \brief Update input buffer var mapping at nnp_iss_bind_input_buffer() call.
   */
  llvm::Value* VisitISSBindInputBuffer(const CallNode* op) {
    llvm::Value* tensor_desc_addr;
    llvm::Function* cur_func = builder_->GetInsertBlock()->getParent();
    tensor_desc_addr = cur_func->arg_begin() + 1;  // i8* r1
    tensor_desc_addr =
        builder_->CreateGEP(tensor_desc_addr, builder_->getInt32(CM_DATA_IN_ADDR_PTR_OFFSET));
    tensor_desc_addr = builder_->CreatePointerCast(tensor_desc_addr,
                                                   t_int8_->getPointerTo(NNP400_ADDRESS_SPACE_NMI)
                                                       ->getPointerTo(NNP400_ADDRESS_SPACE_NMI)
                                                       ->getPointerTo(NNP400_ADDRESS_SPACE_CM));
    tensor_desc_addr = builder_->CreateLoad(tensor_desc_addr);  // i8**

    for (size_t i = 0; i < input_buffer_vars_.size(); ++i) {
      const VarNode* input_buffer_var = input_buffer_vars_[i];
      llvm::Value* buffer_ddr_addr =
          builder_->CreateLoad(tensor_desc_addr, input_buffer_var->name_hint.c_str());
      var_map_[input_buffer_var] = buffer_ddr_addr;
      if (i < op->args.size() - 1) {
        tensor_desc_addr = builder_->CreateGEP(tensor_desc_addr, builder_->getInt32(1));
      }
    }
    return tensor_desc_addr;
  }

 protected:
  void InitTarget(llvm::TargetMachine* tm) final {
    // Maximum vector lane = float4
    native_vector_bits_ = 16 * 32;
    CodeGenLLVM::InitTarget(tm);
  }

 private:
  using IntrinGenF = std::function<llvm::Value*(const CallNode*)>;
  void RegisterNNPIntrinsic(const tvm::Op& op, const IntrinGenF& gen) { intrinsic_dict_[op] = gen; }

  typedef llvm::Value* (CodeGenNNP400LLVM::*IntrinGenMemF)(const CallNode*);
  void RegisterNNPIntrinsic(const tvm::Op& op, const IntrinGenMemF& gen) {
    IntrinGenF f = std::bind(gen, this, std::placeholders::_1);
    intrinsic_dict_[op] = f;
  }

  llvm::Value* CreateNNPIntrinsic(const CallNode* call, const Op& op) {
    auto it = intrinsic_dict_.find(op);
    ICHECK(it != intrinsic_dict_.end()) << "Do not support nnp intrinsic: " << op;
    const IntrinGenF& f = it->second;
    return f(call);
  }

  void InitNNPIntrinsics() {
    // iss buffer binding
    RegisterNNPIntrinsic(edgex::builtin::nnp_iss_bind_input_buffer(),
                         &CodeGenNNP400LLVM::VisitISSBindInputBuffer);

    // dma intrinsics
    RegisterNNPIntrinsic(edgex::builtin::nnp_cube(), &CodeGenNNP400LLVM::CreateCube);
    RegisterNNPIntrinsic(edgex::builtin::nnp_eidma_load(), &CodeGenNNP400LLVM::CreateEidmaLoad);
    RegisterNNPIntrinsic(edgex::builtin::nnp_eodma_store(), &CodeGenNNP400LLVM::CreateEodmaStore);
    RegisterNNPIntrinsic(edgex::builtin::nnp_ewdma_load(), &CodeGenNNP400LLVM::CreateEwdmaLoad);
    RegisterNNPIntrinsic(edgex::builtin::nnp_idma_load(), &CodeGenNNP400LLVM::CreateIdmaLoad);
    RegisterNNPIntrinsic(edgex::builtin::nnp_odma_store(), &CodeGenNNP400LLVM::CreateOdmaStore);
    RegisterNNPIntrinsic(edgex::builtin::nnp_vidma_load(), &CodeGenNNP400LLVM::CreateVidmaLoad);
    RegisterNNPIntrinsic(edgex::builtin::nnp_vodma_store(), &CodeGenNNP400LLVM::CreateVodmaStore);
    RegisterNNPIntrinsic(edgex::builtin::nnp_wdma_load(), &CodeGenNNP400LLVM::CreateWdmaLoad);
    RegisterNNPIntrinsic(edgex::builtin::nnp_bdma_load(), &CodeGenNNP400LLVM::CreateBdmaLoad);
    RegisterNNPIntrinsic(edgex::builtin::nnp_lock_vcu(), &CodeGenNNP400LLVM::CreateLockVcu);
    RegisterNNPIntrinsic(edgex::builtin::nnp_unlock_vcu(), &CodeGenNNP400LLVM::CreateUnlockVcu);

    // handshakes
    RegisterNNPIntrinsic(edgex::builtin::nnp_sync(),
                         [this](auto op) { return CreateHandShakeOp(op); });

    // special registers
    RegisterNNPIntrinsic(edgex::builtin::nnp_cuid(), [this](auto op) {
      return CreateSimpleIntrinsic(llvm::Intrinsic::nnp_cuid, op);
    });

    // vu intrinsics
    RegisterNNPIntrinsic(edgex::builtin::nnp_veltadd(), &CodeGenNNP400LLVM::CreateVeltadd);
    RegisterNNPIntrinsic(edgex::builtin::nnp_vacc_madd_right_shift(),
                         &CodeGenNNP400LLVM::CreateVacccMaddRightShift);
  }

  /*! \brief keep input buffer variables from tir function arguments. */
  std::vector<const VarNode*> input_buffer_vars_;

  /*! \brief nnp intrinsic codegen router dict. */
  std::map<tvm::Op, IntrinGenF> intrinsic_dict_;

  /*! \brief vcu resource info */
  PrimExpr vcore_resource_;
};

/**
 * Currently, the iss calling convention require us to generate function with
 * specialized signature: void f(_, i8*) where the second argument is the
 * start ddr address for input tensor addresses list. The second argument
 * is assumed to be always mapped to r1 register to match iss convention.
 * All actual params from tir function are dropped since we actually cannot
 * support it now.
 * TODO(someone): Implement llvm calling convention for nnp target?
 */
void CodeGenNNP400LLVM::AddFunctionInternal(const PrimFunc& f, bool ret_void) {
  this->InitFuncState();
  ICHECK_EQ(f->buffer_map.size(), 0U)
      << "Cannot codegen function with buffer_map, please lower them first";

  std::vector<llvm::Type*> param_types;
  param_types.push_back(t_int32_);  // the first param is just unused placeholder
  param_types.push_back(t_int8_->getPointerTo(NNP400_ADDRESS_SPACE_CM));
  llvm::FunctionType* ftype = llvm::FunctionType::get(t_void_, param_types, false);

  // original params are dropped in llvm function, keep the information
  // to be used by generate nnp_iss_bind_input_buffer() call.
  input_buffer_vars_.clear();
  for (size_t i = 0; i < f->params.size(); ++i) {
    const Var& var = f->params[i];
    CHECK(var.dtype().is_handle()) << "Only support handle type argument";
    input_buffer_vars_.push_back(var.get());
  }

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.defined())
      << "CodeGenLLVM: Expect PrimFunc to have the global_symbol attribute";
  ICHECK(module_->getFunction(static_cast<std::string>(global_symbol.value())) == nullptr)
      << "Function " << global_symbol << " already exist in module";

  function_ = llvm::Function::Create(ftype, llvm::Function::ExternalLinkage,
                                     global_symbol.value().operator std::string(), module_.get());
  function_->setCallingConv(llvm::CallingConv::C);
  function_->setDLLStorageClass(llvm::GlobalValue::DLLStorageClassTypes::DLLExportStorageClass);

  llvm::BasicBlock* entry = llvm::BasicBlock::Create(*ctx_, "entry", function_);
  builder_->SetInsertPoint(entry);

  // set default setmode config of tvm application convention
  CreateNNPSetMode(NNPSetModeSpec::TVM_APP_DEFAULT_SPEC);

  this->VisitStmt(f->body);
  llvm::StringRef fs = target_machine_->getTargetFeatureString();
  if (!fs.empty()) {
    function_->addFnAttr("target-features", fs);
  }

  // recover hardware default setmode config
  CreateNNPSetMode(NNPSetModeSpec::HW_DEFAULT_SPEC);

  builder_->CreateRetVoid();

  if (vcore_resource_.defined() && analyzer_->CanProveGreaterEqual(vcore_resource_, 1)) {
    auto md_value = llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(t_int32_, 1));
    llvm::MDNode* md_node = llvm::MDNode::get(module_->getContext(), {md_value});
    function_->setMetadata(tvm::tir::attr::nnp_vcore_resource, md_node);
    vcore_resource_ = PrimExpr();
  }
}

std::unique_ptr<llvm::Module> CodeGenNNP400LLVM::Finish() {
  this->AddStartupFunction();
  for (size_t i = 0; i < link_modules_.size(); ++i) {
    ICHECK(!llvm::Linker::linkModules(*module_, std::move(link_modules_[i])))
        << "Failed to link modules";
  }
  link_modules_.clear();
  return std::move(module_);
}

void CodeGenNNP400LLVM::OptimizeModule(llvm::Module* module, llvm::TargetMachine* tm) {
  // pass manager
  llvm::legacy::FunctionPassManager fpass(module);
  llvm::legacy::PassManager mpass;
  mpass.add(llvm::createTargetTransformInfoWrapperPass(tm ? tm->getTargetIRAnalysis()
                                                          : llvm::TargetIRAnalysis()));
  fpass.add(llvm::createTargetTransformInfoWrapperPass(tm ? tm->getTargetIRAnalysis()
                                                          : llvm::TargetIRAnalysis()));

  // place optimization pass
  llvm::PassManagerBuilder builder;
  builder.OptLevel = 2;

#if TVM_LLVM_VERSION >= 50
  builder.Inliner = llvm::createFunctionInliningPass(builder.OptLevel, 0, false);
#else
  builder.Inliner = llvm::createFunctionInliningPass(builder.OptLevel, 0);
#endif
  builder.LoopVectorize = false;
  builder.SLPVectorize = false;
#if TVM_LLVM_VERSION >= 50
  tm->adjustPassManager(builder);
#endif

  builder.populateFunctionPassManager(fpass);
  builder.populateModulePassManager(mpass);

  fpass.doInitialization();
  for (auto it = module->begin(); it != module->end(); ++it) {
    fpass.run(*it);
  }
  fpass.doFinalization();
  mpass.run(*module);
}

static std::once_flag init_llvm_cl_options_flag;

void InitLLVMClOptions() {
  std::unordered_map<std::string, std::string> arg_dict;
  arg_dict["-enable-packets"] = "1";
  arg_dict["-enable-pload"] = "0";
  const std::string cmdline_arg_str = dmlc::GetEnv<std::string>("EDGEX_LLVM_CMDLINE_OPTIONS", "");
  if (!cmdline_arg_str.empty()) {
    std::string str;
    for (std::istringstream is(cmdline_arg_str); is >> str;) {
      auto pos = str.find("=");
      if (pos != std::string::npos) {
        std::string key = str.substr(0, pos);
        std::string value = str.substr(pos + 1, str.size() - pos - 1);
        arg_dict[key] = value;
      } else {
        arg_dict[str] = "";
      }
    }
  }
  std::vector<std::string> arg_vec;
  arg_vec.push_back("placeholder");
  for (const auto& p : arg_dict) {
    if (p.second.empty()) {
      arg_vec.push_back(p.first);
    } else {
      arg_vec.push_back(p.first + "=" + p.second);
    }
  }
  std::vector<const char*> argv(arg_vec.size());
  for (size_t i = 0; i < arg_vec.size(); ++i) {
    argv[i] = arg_vec[i].c_str();
  }
  llvm::cl::ParseCommandLineOptions(argv.size(), argv.data());
}

static std::unique_ptr<llvm::TargetMachine> GetNNPTargetMachine() {
  const char* target_triple = "nnp";
  const char* mcpu = "nnp";
  const char* mattr = "";
  llvm::TargetOptions opt;
  opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  opt.UnsafeFPMath = false;
  opt.NoInfsFPMath = false;
  opt.NoNaNsFPMath = true;
  opt.FloatABIType = llvm::FloatABI::Hard;
  opt.DisableIntegratedAS = true;
  std::string err;
  const llvm::Target* llvm_target = llvm::TargetRegistry::lookupTarget(target_triple, err);
  ICHECK(llvm_target) << "Lookup target " << target_triple << " failed";
  llvm::TargetMachine* tm =
      llvm_target->createTargetMachine(target_triple, mcpu, mattr, opt, llvm::Reloc::PIC_);
  return std::unique_ptr<llvm::TargetMachine>(tm);
}

std::unique_ptr<llvm::Module> NNP400CreateLLVMModule(IRModule mod, Target target,
                                                     llvm::TargetMachine* tm,
                                                     llvm::LLVMContext* ctx, bool do_optimize) {
  // init llvm and config cl options, reentrancy ensured
  std::call_once(init_llvm_cl_options_flag, InitLLVMClOptions);

  std::unique_ptr<CodeGenNNP400LLVM> cg(new CodeGenNNP400LLVM());
  cg->Init("TVMLLVMModule", tm, ctx, false, false, false);

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "Can only lower IR Module with PrimFuncs";
    auto f = Downcast<PrimFunc>(kv.second);
    cg->AddFunction(f);
  }

  std::unique_ptr<llvm::Module> module = cg->Finish();
  if (do_optimize) {
    CodeGenNNP400LLVM::OptimizeModule(module.get(), tm);
  }
  return std::move(module);
}

runtime::Module BuildNNP400LLVM(IRModule mod, Target target) {
  InitializeLLVM();
  std::unique_ptr<llvm::TargetMachine> tm = GetNNPTargetMachine();
  std::unique_ptr<llvm::LLVMContext> ctx(new llvm::LLVMContext());
  std::unordered_map<std::string, std::string> kernel_map;

  // llvm output filetype: either asm or obj
  std::string llvm_output_filetype = dmlc::GetEnv<std::string>("EDGEX_LLVM_OUTPUT_FILETYPE", "asm");
  bool gen_obj;
  if (llvm_output_filetype == "asm") {
    gen_obj = false;
  } else if (llvm_output_filetype == "obj") {
    gen_obj = true;
  } else {
    LOG(FATAL) << "Invalid EDGEX_LLVM_OUTPUT_FILETYPE: " << llvm_output_filetype;
  }

  // llvm dump working dir
  const std::string working_dir = dmlc::GetEnv<std::string>("EDGEX_DEBUG_WORKING_DIR", "");
  bool dump_llvm_ir = !dmlc::GetEnv<std::string>("EDGEX_DEBUG_DUMP_LLVM", "").empty();
  if (dump_llvm_ir && working_dir.empty()) {
    LOG(WARNING) << "No `EDGEX_DEBUG_WORKING_DIR` specified for llvm ir dumping, use logging.";
  }

  // currently we have to build per-function
  for (auto kv : mod->functions) {
    IRModule single_func_mod = mod;
    single_func_mod.CopyOnWrite()->functions = {kv};
    auto llvm_mod = NNP400CreateLLVMModule(single_func_mod, target, tm.get(), ctx.get(), false);

    // dump ll before optimize
    const std::string kernel_name = kv.first->name_hint;
    std::string llvm_dump_dir;
    if (dump_llvm_ir) {
      if (!working_dir.empty()) {
        llvm_dump_dir = working_dir + "/" + kernel_name + "/llvm/";
        int status = tvm::runtime::mkdir_recursive(llvm_dump_dir.c_str());
        CHECK(status == 0 || errno == EEXIST) << "Create llvm ir dump directory failed";
        std::error_code ferr;
        llvm::raw_fd_ostream fstream(llvm_dump_dir + "/" + kernel_name + ".origin.ll", ferr,
                                     llvm::sys::fs::FA_Write);
        llvm_mod->print(fstream, nullptr);
        fstream.close();
      }
    }

    // optimize the ll module
    CodeGenNNP400LLVM::OptimizeModule(llvm_mod.get(), tm.get());

    // dump ll after optimize
    if (dump_llvm_ir) {
      if (!llvm_dump_dir.empty()) {
        std::error_code ferr;
        llvm::raw_fd_ostream fstream(llvm_dump_dir + "/" + kernel_name + ".ll", ferr,
                                     llvm::sys::fs::FA_Write);
        llvm_mod->print(fstream, nullptr);
        fstream.close();
      } else {
        llvm::SmallString<8> data_ll;
        llvm::raw_svector_ostream dest_ll(data_ll);
        dest_ll.SetUnbuffered();
        llvm_mod->print(dest_ll, nullptr);
        LOG(INFO) << "LLVM IR for " << kernel_name << ":\n" << data_ll.c_str();
      }
    }

    std::string verify_errors_storage;
    llvm::raw_string_ostream verify_errors(verify_errors_storage);
    LOG_IF(FATAL, llvm::verifyModule(*llvm_mod, &verify_errors))
        << "LLVM module verification failed with the following errors: \n"
        << verify_errors.str();
    auto llvm_func = llvm_mod->getFunction(kernel_name);
    ICHECK(llvm_func) << kernel_name << " not found in llvm module";
    llvm::legacy::PassManager pass;

    if (gen_obj) {
      llvm::SmallString<8> data_obj;
      llvm::raw_svector_ostream dest_obj(data_obj);
      dest_obj.SetUnbuffered();
      CHECK_EQ(tm->addPassesToEmitFile(pass, dest_obj, nullptr, llvm::CGFT_ObjectFile), 0);
      pass.run(*llvm_mod);
      kernel_map[kernel_name] = std::string(data_obj.begin(), data_obj.end());
    } else {
      llvm::SmallString<8> data_asm;
      llvm::raw_svector_ostream dest_asm(data_asm);
      dest_asm.SetUnbuffered();
      CHECK_EQ(tm->addPassesToEmitFile(pass, dest_asm, nullptr, llvm::CGFT_AssemblyFile), 0);
      pass.run(*llvm_mod);
      std::stringstream ss;
      ss << "#include \"nnp400_main.h\"\n";
      ss << data_asm.c_str();

      // whether we should link vcu utilities
      auto md_node = llvm_func->getMetadata(tvm::tir::attr::nnp_vcore_resource);
      if (md_node) {
        auto md_oprand = llvm::dyn_cast<llvm::ConstantAsMetadata>(md_node->getOperand(0));
        ICHECK(md_oprand);
        llvm::Constant* const_node = md_oprand->getValue();
        ICHECK(const_node && const_node->getType()->isIntegerTy());
        int64_t vcore_resource = const_node->getUniqueInteger().getSExtValue();
        if (vcore_resource > 0) {
          ss << "\n#include \"get_mbx_lock.asm\"";
          ss << "\n#include \"cfg_vu_desp.asm\"";
        }
      }
      kernel_map[kernel_name] = ss.str();
    }
  }
  if (gen_obj) {
    return tvm::runtime::EdgeXModuleCreateFromObjects(mod, kernel_map, working_dir);
  } else {
    return tvm::runtime::EdgeXModuleCreateFromAsm(mod, kernel_map, working_dir);
  }
}

TVM_REGISTER_GLOBAL("target.build.edgex").set_body_typed(BuildNNP400LLVM);

TVM_REGISTER_GLOBAL("target.build.edgex.create_llvm_module")
    .set_body_typed([](IRModule mod, Target target, bool do_optimize) {
      InitializeLLVM();
      std::unique_ptr<llvm::TargetMachine> tm = GetNNPTargetMachine();
      std::unique_ptr<llvm::LLVMContext> ctx(new llvm::LLVMContext());
      auto llvm_mod = NNP400CreateLLVMModule(mod, target, tm.get(), ctx.get(), do_optimize);

      std::string error_msg;
      llvm::raw_string_ostream error_msg_stream(error_msg);
      ICHECK(!llvm::verifyModule(*llvm_mod, &error_msg_stream))
          << "LLVM module verification failed with the following errors: \n"
          << error_msg;

      std::string ll_data;
      llvm::raw_string_ostream ll_stream(ll_data);
      llvm_mod->print(ll_stream, nullptr);
      return ll_data;
    });

}  // namespace codegen
}  // namespace tvm
#endif  // (defined TVM_LLVM_VERSION) && (defined NNP400_LLVM_CG)