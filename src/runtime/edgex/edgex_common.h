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
 * \file edgex_common.h
 * \brief Common utilities for EdgeX
 */
#ifndef TVM_RUNTIME_EDGEX_EDGEX_COMMON_H_
#define TVM_RUNTIME_EDGEX_EDGEX_COMMON_H_

extern "C" {
#include <dcl_base.h>
#include <libgen.h>
#include <sys/stat.h>
}
#include <tvm/runtime/packed_func.h>

#include <string>
#include <utility>

#include "../workspace_pool.h"

namespace tvm {
namespace runtime {

#define EDGEX_CALL(func)                                                                  \
  {                                                                                       \
    dclError e = func;                                                                    \
    if (e != DCL_ERROR_NONE && e != DCL_ERROR_UNINITIALIZE) {                             \
      LOG(FATAL) << "EdgeX Error: " #func " failed with error: " << dclGetErrorString(e); \
    }                                                                                     \
  }

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
#endif  // TVM_RUNTIME_EDGEX_EDGEX_COMMON_H_
