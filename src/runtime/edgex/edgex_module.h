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
 * \file edgex_module.h
 * \brief Execution handling of EdgeX kernels
 */
#ifndef TVM_RUNTIME_EDGEX_EDGEX_MODULE_H_
#define TVM_RUNTIME_EDGEX_EDGEX_MODULE_H_

#include <tvm/ir/module.h>
#include <tvm/runtime/module.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../meta_data.h"

namespace tvm {
namespace runtime {

// TODO(@yiheng): kMaxNumGPUs, edgex_source
// TODO(@yiheng): name(edgex, dcl) and their format
/*! \brief Maximum number of GPU supported in EdgeXModule */
static constexpr const int kMaxNumGPUs = 32;

/*! \brief function information needed by device */
struct EdgeXFunctionInfo {
  std::string name;
  std::vector<DLDataType> input_types;
  std::vector<std::vector<uint64_t>> input_shapes;
  std::vector<DLDataType> output_types;
  std::vector<std::vector<uint64_t>> output_shapes;

  void AddInput(DLDataType dtype) {}
  void AddOutput(DLDataType dtype) {}

  void Save(dmlc::JSONWriter* writer) const {}
  void Load(dmlc::JSONReader* reader) {}
  void Save(dmlc::Stream* writer) const {}
  bool Load(dmlc::Stream* reader) { return false; }
};

/*!
 * \brief create a edgex module from data.
 *
 * \param bin_data The bin data
 * \param lst_data The lst data
 * \param fmt The format of the data
 * \param fmap The map function information map of each function.
 * \param cuda_source Optional, edgex source code.
 */
Module EdgeXModuleCreate(const std::string& bin_data, const std::string& lst_data,
                         const std::string& fmt,
                         const std::unordered_map<std::string, EdgeXFunctionInfo>& fmap,
                         const std::string& edgex_source);

/*!
 * \brief create a edgex module with original ir module and ass tool chain.
 *
 * \param mod Original ir module.
 * \param asm_map Edgex source asm code.
 * \param working_dir Edgex working directory.
 */
Module EdgeXModuleCreateFromAsm(tvm::IRModule mod,
                                const std::unordered_map<std::string, std::string>& asm_map,
                                const std::string& working_dir);

/*!
 * \brief create a edgex module with original ir module and per-kernel objects.
 *
 * \param mod Original ir module.
 * \param obj_map Edgex per-kernel objects.
 * \param working_dir Edgex working directory.
 */
Module EdgeXModuleCreateFromObjects(tvm::IRModule mod,
                                    const std::unordered_map<std::string, std::string>& obj_map,
                                    const std::string& working_dir);

}  // namespace runtime
}  // namespace tvm

namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, ::tvm::runtime::EdgeXFunctionInfo, true);
}  // namespace dmlc
#endif  // TVM_RUNTIME_EDGEX_EDGEX_MODULE_H_
