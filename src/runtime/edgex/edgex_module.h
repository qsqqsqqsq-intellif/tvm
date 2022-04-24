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

extern "C" {
#include <libgen.h>
#include <sys/stat.h>
}

#include <tvm/ir/module.h>
#include <tvm/runtime/module.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../meta_data.h"

namespace tvm {
namespace runtime {

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
                         const std::unordered_map<std::string, FunctionInfo>& fmap,
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

/**
 *! \brief Get the start address and total bytes of the allocated range
 * the input address belongs to. Return (nullptr, 0) if the input address
 * is not valid device address.
 */
std::pair<void*, size_t> EdgeXQueryDeviceAddress(void* addr);

/**
 *! \brief Return whether iss debug mode is enabled.
 */
bool EdgeXIsISSDebugMode();

/*! \brief mkdir utility */
inline int mkdir_recursive(const char* path, int mode = 0777) {
  int status = 0;
  char* tmp = strdup(path);
  const char* parent = dirname(tmp);
  struct stat st;
  if (strcmp(parent, ".") != 0) {
    if (stat(parent, &st) != 0) {
      status = mkdir_recursive(parent, mode);
    }
  }
  free(tmp);
  if (status != 0) return status;
  return mkdir(path, mode);
}

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_EDGEX_EDGEX_MODULE_H_
