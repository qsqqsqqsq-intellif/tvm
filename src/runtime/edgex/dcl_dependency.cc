/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License") = 0; you may not use this file except in compliance
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
 * \file dcl_dependency.cc
 * \brief Replacement header initialization for dcl.h
 */
#ifdef EDGEX_USE_DYN_DCL
#include "dcl_dependency.h"

#include <dlfcn.h>
#include <tvm/runtime/logging.h>

extern "C" {

dclError (*dclInit)(const char* configPath) = 0;
dclError (*dclFinalize)() = 0;
dclError (*dclrtSetDevice)(int32_t deviceId) = 0;
const char* (*dclGetErrorString)(dclError err) = 0;

dclError (*dclrtCreateStream)(dclrtStream* stream) = 0;
dclError (*dclrtSynchronizeStream)(dclrtStream stream) = 0;
dclError (*dclrtDestroyStream)(dclrtStream stream) = 0;

dclError (*dclopUnloadByName)(const char* opName) = 0;
dclError (*dclopLoadWithLst)(const char* opName, const void* model, size_t modelSize,
                             const void* lst, size_t lstSize) = 0;
dclError (*dclKernelLaunch)(const char* opType, int numParam, void* paramData[],
                            dclrtStream stream) = 0;

dclError (*dclrtMalloc)(void** devPtr, size_t size, dclrtMemMallocPolicy policy) = 0;
dclError (*dclrtMallocHost)(void** hostPtr, size_t size) = 0;
dclError (*dclrtFree)(void* devPtr) = 0;
dclError (*dclrtFreeHost)(void* hostPtr) = 0;
dclError (*dclrtMemcpyAsync)(void* dst, size_t destMax, const void* src, size_t count,
                             dclrtMemcpyKind kind, dclrtStream stream) = 0;
dclError (*dclrtMemcpy)(void* dst, size_t destMax, const void* src, size_t count,
                        dclrtMemcpyKind kind) = 0;

dclError (*dclrtCreateEvent)(dclrtEvent* event) = 0;
dclError (*dclrtDestroyEvent)(dclrtEvent event) = 0;
dclError (*dclrtRecordEvent)(dclrtEvent event, dclrtStream stream) = 0;
dclError (*dclrtStreamWaitEvent)(dclrtStream stream, dclrtEvent event) = 0;

}  // extern "C"

namespace tvm {
namespace runtime {

template <typename FType>
inline void LoadDCLSymbol(FType** func_ptr, void* lib_handle, const char* name) {
  char* error = NULL;
  void* symbol = dlsym(lib_handle, name);
  if ((error = dlerror()) != NULL) {
    LOG(ERROR) << "Fail to not find dcl symbol: " << name;
    exit(-1);
  }
  *func_ptr = reinterpret_cast<FType*>(symbol);
}

static void DoInitDCLDependencies() {
  void* liberror = dlopen("liberror_string.so", RTLD_GLOBAL | RTLD_LAZY);
  if (!liberror) {
    LOG(ERROR) << "Cannot load liberror_string: " << dlerror();
    exit(-1);
  }
  void* libutil = dlopen("libdcl_utils.so", RTLD_GLOBAL | RTLD_LAZY);
  if (!libutil) {
    LOG(ERROR) << "Cannot load libdcl_utils: " << dlerror();
    exit(-1);
  }
  void* libstream = dlopen("libdcl_streamsched.so", RTLD_GLOBAL | RTLD_LAZY);
  if (!libstream) {
    LOG(ERROR) << "Cannot load libdcl_streamsched: " << dlerror();
    exit(-1);
  }
  void* libpb = dlopen("libprotobuf.so", RTLD_GLOBAL | RTLD_LAZY);
  if (!libpb) {
    LOG(ERROR) << "Cannot load libprotobuf: " << dlerror();
    exit(-1);
  }
  void* libgrpc = dlopen("libgrpc++.so", RTLD_GLOBAL | RTLD_LAZY);
  if (!libgrpc) {
    LOG(ERROR) << "Cannot load libgrpc++.so: " << dlerror();
    exit(-1);
  }
  void* librpc = dlopen("librpc.so", RTLD_GLOBAL | RTLD_LAZY);
  if (!librpc) {
    LOG(ERROR) << "Cannot load librpc: " << dlerror();
    exit(-1);
  }
  void* libdrv = dlopen("libdcl_simudrv.so", RTLD_GLOBAL | RTLD_LAZY);
  if (!libdrv) {
    LOG(ERROR) << "Cannot load libdcl_simudrv: " << dlerror();
    exit(-1);
  }
  void* lib = dlopen("libdcl_runtime.so", RTLD_GLOBAL | RTLD_LAZY);
  if (!lib) {
    LOG(ERROR) << "Cannot load libdcl_runtime: " << dlerror();
    exit(-1);
  }
  dlerror();
  LoadDCLSymbol(&dclInit, lib, "dclInit");
  LoadDCLSymbol(&dclFinalize, lib, "dclFinalize");
  LoadDCLSymbol(&dclrtSetDevice, lib, "dclrtSetDevice");
  LoadDCLSymbol(&dclGetErrorString, liberror, "dclGetErrorString");

  LoadDCLSymbol(&dclrtCreateStream, lib, "dclrtCreateStream");
  LoadDCLSymbol(&dclrtSynchronizeStream, lib, "dclrtSynchronizeStream");
  LoadDCLSymbol(&dclrtDestroyStream, lib, "dclrtDestroyStream");

  LoadDCLSymbol(&dclrtCreateEvent, lib, "dclrtCreateEvent");
  LoadDCLSymbol(&dclrtDestroyEvent, lib, "dclrtDestroyEvent");
  LoadDCLSymbol(&dclrtRecordEvent, lib, "dclrtRecordEvent");
  LoadDCLSymbol(&dclrtStreamWaitEvent, lib, "dclrtStreamWaitEvent");

  LoadDCLSymbol(&dclrtMalloc, lib, "dclrtMalloc");
  LoadDCLSymbol(&dclrtMallocHost, lib, "dclrtMallocHost");
  LoadDCLSymbol(&dclrtFree, lib, "dclrtFree");
  LoadDCLSymbol(&dclrtFreeHost, lib, "dclrtFreeHost");
  LoadDCLSymbol(&dclrtMemcpy, lib, "dclrtMemcpy");
  LoadDCLSymbol(&dclrtMemcpyAsync, lib, "dclrtMemcpyAsync");

  LoadDCLSymbol(&dclopUnloadByName, lib, "dclopUnloadByName");
  LoadDCLSymbol(&dclopLoadWithLst, lib, "dclopLoadWithLst");
  LoadDCLSymbol(&dclKernelLaunch, lib, "dclKernelLaunch");
}

static std::once_flag dcl_init_flag;

/*! \brief Initialize dcl apis in dyn load mode. */
void InitDCLDependencies() { std::call_once(dcl_init_flag, DoInitDCLDependencies); }

}  // namespace runtime
}  // namespace tvm
#endif
