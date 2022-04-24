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
 * \file dcl_dependency.h
 * \brief Replacement header for dcl.h
 */
#ifndef TVM_RUNTIME_EDGEX_DCL_DEPENDENCY_H_
#define TVM_RUNTIME_EDGEX_DCL_DEPENDENCY_H_

#define EDGEX_CALL(func)                                                                  \
  {                                                                                       \
    dclError e = func;                                                                    \
    if (e != DCL_ERROR_NONE && e != DCL_ERROR_UNINITIALIZE) {                             \
      LOG(FATAL) << "EdgeX Error: " #func " failed with error: " << dclGetErrorString(e); \
    }                                                                                     \
  }

#ifndef EDGEX_USE_DYN_DCL
extern "C" {
#include <dcl.h>

namespace tvm {
namespace runtime {

/*! \brief Initialize dcl apis in dyn load mode. */
inline void InitDCLDependencies() {}

}  // namespace runtime
}  // namespace tvm

}  // extern "C"

#else
/* load via dynamic loading to decouple dcl dependencies. */
#include <stddef.h>
#include <stdint.h>

namespace tvm {
namespace runtime {

void InitDCLDependencies();

}  // namespace runtime
}  // namespace tvm

extern "C" {
typedef void* dclrtStream;
typedef int32_t dclError;
typedef void* dclrtEvent;

const int32_t DCL_ERROR_NONE = 0;
const int32_t DCL_ERROR_UNINITIALIZE = 100001;
const int32_t DCL_ERROR_OP_NOT_FOUND = 100024;

/*! \brief Initialize dcl apis in dyn load mode. */
void InitDCLDependencies();

typedef enum dclrtMemcpyKind {
  DCL_MEMCPY_HOST_TO_HOST,
  DCL_MEMCPY_HOST_TO_DEVICE,
  DCL_MEMCPY_DEVICE_TO_HOST,
  DCL_MEMCPY_DEVICE_TO_DEVICE,
} dclrtMemcpyKind;

typedef enum dclrtMemMallocPolicy {
  DCL_MEM_MALLOC_HUGE_FIRST,
  DCL_MEM_MALLOC_HUGE_ONLY,
  DCL_MEM_MALLOC_NORMAL_ONLY,
} dclrtMemMallocPolicy;

extern dclError (*dclInit)(const char* configPath);
extern dclError (*dclFinalize)();
extern dclError (*dclrtSetDevice)(int32_t deviceId);
extern const char* (*dclGetErrorString)(dclError err);

extern dclError (*dclrtCreateStream)(dclrtStream* stream);
extern dclError (*dclrtSynchronizeStream)(dclrtStream stream);
extern dclError (*dclrtDestroyStream)(dclrtStream stream);

extern dclError (*dclopUnloadByName)(const char* opName);
extern dclError (*dclopLoadWithLst)(const char* opName, const void* model, size_t modelSize,
                                    const void* lst, size_t lstSize);
extern dclError (*dclKernelLaunch)(const char* opType, int numParam, void* paramData[],
                                   dclrtStream stream);

extern dclError (*dclrtMalloc)(void** devPtr, size_t size, dclrtMemMallocPolicy policy);
extern dclError (*dclrtMallocHost)(void** hostPtr, size_t size);
extern dclError (*dclrtFree)(void* devPtr);
extern dclError (*dclrtFreeHost)(void* hostPtr);
extern dclError (*dclrtMemcpyAsync)(void* dst, size_t destMax, const void* src, size_t count,
                                    dclrtMemcpyKind kind, dclrtStream stream);
extern dclError (*dclrtMemcpy)(void* dst, size_t destMax, const void* src, size_t count,
                               dclrtMemcpyKind kind);

extern dclError (*dclrtCreateEvent)(dclrtEvent* event);
extern dclError (*dclrtDestroyEvent)(dclrtEvent event);
extern dclError (*dclrtRecordEvent)(dclrtEvent event, dclrtStream stream);
extern dclError (*dclrtStreamWaitEvent)(dclrtStream stream, dclrtEvent event);

}  // extern "C"
#endif

#endif  // TVM_RUNTIME_EDGEX_DCL_DEPENDENCY_H_
