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
 * \file codegen_nnp_intrisic_info.h
 * \brief NNP400 nu instruction desciption.
 */

#ifndef TVM_TARGET_EDGEX_CODEGEN_NNP_INTRINSIC_INFO_H_
#define TVM_TARGET_EDGEX_CODEGEN_NNP_INTRINSIC_INFO_H_

#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IntrinsicsNNP.h>

#include <map>
#include <vector>

namespace tvm {
namespace codegen {

struct IntrinsicInfo {
  llvm::Intrinsic::ID intrinsicID;
  std::vector<const char*> params;
};

extern std::map<llvm::Intrinsic::ID, std::vector<const char*>> NNP400IntrinsicInfos;

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_TARGET_EDGEX_CODEGEN_NNP_INTRINSIC_INFO_H_
