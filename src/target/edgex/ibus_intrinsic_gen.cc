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
 * \file ibus_intrinsic_gen.cc
 * \brief NNP400 ibus instruction generate helper.
 */
#include "./ibus_intrinsic_gen.h"

#include <tvm/runtime/logging.h>

#include "./codegen_nnp_intrinsic_info.h"
#include "nnp400_main.h"

namespace tvm {
namespace codegen {

static llvm::Value* CreateIBusIntrin(llvm::IRBuilder<>* builder, llvm::Value* r1, llvm::Value* r2) {
  llvm::Function* ibus = llvm::Intrinsic::getDeclaration(builder->GetInsertBlock()->getModule(),
                                                         llvm::Intrinsic::nnp_movr2ibus, {});
  ICHECK(r1->getType() == builder->getInt32Ty()) << "ibus intrin's arg0 must be i32";
  ICHECK(r2->getType() == builder->getInt32Ty()) << "ibus intrin's arg1 must be i32";
  return builder->CreateCall(ibus, {r1, r2});
}

/**
 *! \brief Encode values to a 64bit instruction and invoke ibus intrin. Parts are ordered from high
 *address to low address.
 */
static llvm::Value* CreateIBusIntrin(llvm::IRBuilder<>* builder,
                                     const std::vector<llvm::Value*>& parts,
                                     const std::vector<size_t>& sizes) {
  size_t total = 0;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (parts[i]->getType() != builder->getInt32Ty()) {
      LOG(FATAL) << "Illegal llvm value type, requires i32";
    }
    total += sizes[i];
  }
  if (total != 64) {
    LOG(FATAL) << "Illegal total bits";
  }
  llvm::Value* p1 = nullptr;
  llvm::Value* p2 = nullptr;
  size_t part_remain_bits = 32;
  size_t remain_bits = 64;
  llvm::Value* cur = nullptr;
  size_t i = 0;
  llvm::Value* value = parts[0];
  size_t bits = sizes[0];
  while (remain_bits > 0 && i < parts.size()) {
    if (bits > part_remain_bits) {
      size_t out_bits = bits - part_remain_bits;
      llvm::Value* left = builder->CreateLShr(value, builder->getInt32(out_bits));
      llvm::Value* right = builder->CreateAnd(value, builder->getInt32((1 << out_bits) - 1));
      cur = cur ? builder->CreateOr(cur, left) : left;
      remain_bits -= part_remain_bits;
      part_remain_bits = 0;
      value = right;
      bits = out_bits;
    } else {
      if (bits < part_remain_bits) {
        value = builder->CreateShl(value, part_remain_bits - bits);
      }
      cur = cur ? builder->CreateOr(cur, value) : value;
      remain_bits -= bits;
      part_remain_bits -= bits;
      ++i;
      value = parts[i];
      bits = sizes[i];
    }
    if (part_remain_bits == 0) {
      if (p1 == nullptr) {
        p1 = cur;
        cur = nullptr;
        part_remain_bits = 32;
      } else {
        p2 = cur;
        break;
      }
    }
  }
  return CreateIBusIntrin(builder, p1, p2);
}

/**
 * ! \brief Utility to extract dma intrin argument by name.
 */
static llvm::Value* GetLLVMIntrinArgByName(int intrin_id, const std::string& name,
                                           const llvm::ArrayRef<llvm::Value*>& args) {
  const auto& arg_names = NNP400IntrinsicInfos[intrin_id];
  auto it = std::find(arg_names.begin(), arg_names.end(), name);
  ICHECK(it != arg_names.end());
  return args[it - arg_names.begin()];
}

llvm::Value* GenerateEidmaCfg2IBus(llvm::IRBuilder<>* builder,
                                   const llvm::ArrayRef<llvm::Value*>& args) {
  int intrin_id = llvm::Intrinsic::nnp_eidma_layer_cfg2;
  llvm::Value* ei_src_addr_low = GetLLVMIntrinArgByName(intrin_id, "ei_src_addr", args);
  llvm::Value* ei_src_addr_high = GetLLVMIntrinArgByName(intrin_id, "ei_src_addr_high", args);
  return CreateIBusIntrin(
      builder, {ei_src_addr_low, ei_src_addr_high, builder->getInt32(EIDMA_EXT_ADDR_INST_HEAD)},
      {32, 6, 26});
}

llvm::Value* GenerateEodmaCfg2IBus(llvm::IRBuilder<>* builder,
                                   const llvm::ArrayRef<llvm::Value*>& args) {
  int intrin_id = llvm::Intrinsic::nnp_eodma_layer_cfg2;
  llvm::Value* ei_src_addr_low = GetLLVMIntrinArgByName(intrin_id, "eo_dst_addr", args);
  llvm::Value* ei_src_addr_high = GetLLVMIntrinArgByName(intrin_id, "eo_dst_addr_high", args);
  return CreateIBusIntrin(
      builder, {ei_src_addr_low, ei_src_addr_high, builder->getInt32(EODMA_EXT_ADDR_INST_HEAD)},
      {32, 6, 26});
}

llvm::Value* GenerateEwdmaCfg2IBus(llvm::IRBuilder<>* builder,
                                   const llvm::ArrayRef<llvm::Value*>& args) {
  int intrin_id = llvm::Intrinsic::nnp_ewdma_layer_cfg2;
  llvm::Value* ew_src_addr_low = GetLLVMIntrinArgByName(intrin_id, "ew_src_addr", args);
  llvm::Value* ew_src_addr_high = GetLLVMIntrinArgByName(intrin_id, "ew_src_addr_high", args);
  return CreateIBusIntrin(
      builder, {ew_src_addr_low, ew_src_addr_high, builder->getInt32(EWDMA_EXT_ADDR_INST_HEAD)},
      {32, 6, 26});
}

llvm::Value* GenerateVidmaCfg7IBus(llvm::IRBuilder<>* builder,
                                   const llvm::ArrayRef<llvm::Value*>& args) {
  int32_t VIDMA_CFG7_INST_HEAD = 0x605f;  // 0b 0110000001011111
  int intrin_id = llvm::Intrinsic::nnp_vidma_layer_cfg7;
  llvm::Value* start_addr1_dm = GetLLVMIntrinArgByName(intrin_id, "start_addr1_dm_vidma", args);
  llvm::Value* end_addr1_dm = GetLLVMIntrinArgByName(intrin_id, "end_addr1_dm_vidma", args);
  return CreateIBusIntrin(builder,
                          {end_addr1_dm, start_addr1_dm, builder->getInt32(VIDMA_CFG7_INST_HEAD)},
                          {24, 24, 16});
}

llvm::Value* GenerateVidmaCfg12IBus(llvm::IRBuilder<>* builder,
                                    const llvm::ArrayRef<llvm::Value*>& args) {
  int32_t VIDMA_CFG12_INST_HEAD = 0x6073;  // 0b 0110000001110011
  int intrin_id = llvm::Intrinsic::nnp_vidma_layer_cfg12;
  llvm::Value* start_addr2_dm = GetLLVMIntrinArgByName(intrin_id, "start_addr2_dm_vidma", args);
  llvm::Value* end_addr2_dm = GetLLVMIntrinArgByName(intrin_id, "end_addr2_dm_vidma", args);
  return CreateIBusIntrin(builder,
                          {end_addr2_dm, start_addr2_dm, builder->getInt32(VIDMA_CFG12_INST_HEAD)},
                          {24, 24, 16});
}

llvm::Value* GenerateVodmaCfg7IBus(llvm::IRBuilder<>* builder,
                                   const llvm::ArrayRef<llvm::Value*>& args) {
  int32_t VODMA_CFG7_INST_HEAD = 0x609f;  // 0b 0110000010011111
  int intrin_id = llvm::Intrinsic::nnp_vodma_layer_cfg7;
  llvm::Value* start_addr1_dm = GetLLVMIntrinArgByName(intrin_id, "start_addr1_dm_vodma", args);
  llvm::Value* end_addr1_dm = GetLLVMIntrinArgByName(intrin_id, "end_addr1_dm_vodma", args);
  return CreateIBusIntrin(builder,
                          {end_addr1_dm, start_addr1_dm, builder->getInt32(VODMA_CFG7_INST_HEAD)},
                          {24, 24, 16});
}

llvm::Value* GenerateVodmaCfg12IBus(llvm::IRBuilder<>* builder,
                                    const llvm::ArrayRef<llvm::Value*>& args) {
  int32_t VODMA_CFG12_INST_HEAD = 0x60b3;  // 0b 0110000010110011
  int intrin_id = llvm::Intrinsic::nnp_vodma_layer_cfg12;
  llvm::Value* start_addr2_dm = GetLLVMIntrinArgByName(intrin_id, "start_addr2_dm_vodma", args);
  llvm::Value* end_addr2_dm = GetLLVMIntrinArgByName(intrin_id, "end_addr2_dm_vodma", args);
  return CreateIBusIntrin(builder,
                          {end_addr2_dm, start_addr2_dm, builder->getInt32(VODMA_CFG12_INST_HEAD)},
                          {24, 24, 16});
}

using GenIntrinFType =
    std::function<llvm::Value*(llvm::IRBuilder<>*, const llvm::ArrayRef<llvm::Value*>&)>;

static std::map<std::string, GenIntrinFType> ibus_gen_dict = {
    {"llvm.nnp.eidma.layer.cfg2", GenerateEidmaCfg2IBus},
    {"llvm.nnp.eodma.layer.cfg2", GenerateEodmaCfg2IBus},
    {"llvm.nnp.ewdma.layer.cfg2", GenerateEwdmaCfg2IBus},
    {"llvm.nnp.vidma.layer.cfg7", GenerateVidmaCfg7IBus},
    {"llvm.nnp.vidma.layer.cfg12", GenerateVidmaCfg12IBus},
    {"llvm.nnp.vodma.layer.cfg7", GenerateVodmaCfg7IBus},
    {"llvm.nnp.vodma.layer.cfg12", GenerateVodmaCfg12IBus}};

llvm::Value* GenerateIBusIntrinsic(llvm::IRBuilder<>* builder, llvm::Function* f,
                                   const llvm::ArrayRef<llvm::Value*>& args) {
  std::string func_name = f->getName().str();
  auto it = ibus_gen_dict.find(func_name);
  ICHECK(it != ibus_gen_dict.end()) << "Can not find ibus gen support for " << func_name;
  const GenIntrinFType& gen = it->second;
  return gen(builder, args);
}

}  // namespace codegen
}  // namespace tvm
