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
 * \file hardware_info.cc
 * \brief Helper functions to edgex hardware info.
 */
#include "./hardware_info.h"

namespace tvm {
namespace tir {
namespace edgex {

Map<String, ObjectRef> GetHardwareConfigDict() {
  auto get_hw_config = ::tvm::runtime::Registry::Get("tvm.edgex.get_current_hw_config");
  ICHECK(get_hw_config);
  return (*get_hw_config)();
}

Range GetDMUserAddrSpaceRange() {
  auto hw_config = GetHardwareConfigDict();
  Integer dm_size =
      Downcast<Integer>(hw_config.Get(attr::kDMSize).value_or(Integer(DEFAULT_DM_SIZE)));
  Integer vu_dm_stack_size = Downcast<Integer>(
      hw_config.Get(attr::kDMStackPerVcu).value_or(Integer(DEFAULT_DM_STACK_PER_VCU)));
  return Range(0, Integer(dm_size->value - vu_dm_stack_size->value));
}

}  // namespace edgex
}  // namespace tir
}  // namespace tvm
