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
 * \file hardware_info.h
 * \brief Helper functions to edgex hardware info.
 */
#ifndef TVM_CONTRIB_EDGEX_TIR_HARDWARE_INFO_H_
#define TVM_CONTRIB_EDGEX_TIR_HARDWARE_INFO_H_

#include <tvm/ir/expr.h>
#include <tvm/runtime/registry.h>

#define DEFAULT_DM_SIZE (3 * 1024 * 1024)
#define DEFAULT_DM_STACK_PER_VCU (4 * 1024)
#define DM_OFFSET_ADDR 0x00000000

#define VCU_NUM 4

namespace tvm {
namespace tir {
namespace edgex {

namespace attr {
/*********************************** Hardware configuration keys
 * ************************************/
constexpr const char* kDMSize = "DM_SIZE";
constexpr const char* kDMStackPerVcu = "DM_STACK_PER_VCU";
}  // namespace attr

/*!
 * \brief Get hardware configuration dict.
 */
Map<String, ObjectRef> GetHardwareConfigDict();

/*!
 * \brief Get available DM space address range.
 */
Range GetDMUserAddrSpaceRange();

}  // namespace edgex
}  // namespace tir
}  // namespace tvm

#endif  // TVM_CONTRIB_EDGEX_TIR_HARDWARE_INFO_H_
