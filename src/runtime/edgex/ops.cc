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
 * \file src/runtime/contrib/edgex/add.cc
 * \brief Use external edgex add function
 */
#include <stdio.h>
#include <stdlib.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

#include "./dcl_dependency.h"

namespace tvm {
namespace contrib {

using namespace runtime;

TVM_REGISTER_GLOBAL("tvm.contrib.edgex.add_example").set_body([](TVMArgs args, TVMRetValue* ret) {
  // edgex routine
  EDGEX_CALL(dclrtSetDevice(0));
  dclrtStream stream;
  EDGEX_CALL(dclrtCreateStream(&stream));

  // data in device
  DLTensor* tensor_in = args[0];
  DLTensor* tensor_out = args[1];

  // bin/lst should come from compiler
  std::string base_dir = std::getenv("EDGEX_ROOT_DIR");
  std::string bin_dir = base_dir + "/tests/drv_case0_ncore_vcore0/drv_case0_ncore_vcore0.bin";
  std::FILE* bin_file = fopen(bin_dir.c_str(), "r");
  std::fseek(bin_file, 0, SEEK_END);
  size_t bin_size = ftell(bin_file);
  std::fseek(bin_file, 0, SEEK_SET);
  void* bin_buffer = malloc(bin_size);
  ICHECK(std::fread(bin_buffer, 1, bin_size, bin_file) == bin_size) << "fread error";
  std::fclose(bin_file);

  std::string lst_dir = base_dir + "/tests/drv_case0_ncore_vcore0/drv_case0_ncore_vcore0_cpp.lst";
  std::FILE* lst_file = fopen(lst_dir.c_str(), "r");
  std::fseek(lst_file, 0, SEEK_END);
  size_t lst_size = ftell(lst_file);
  std::fseek(lst_file, 0, SEEK_SET);
  void* lst_buffer = malloc(lst_size);
  ICHECK(std::fread(lst_buffer, 1, lst_size, lst_file) == lst_size) << "fread error";
  std::fclose(lst_file);

  // in EdgeXModuleNode
  const char* op_type = "drv_case0_ncore_vcore0";
  dclError e = dclopUnloadByName(op_type);
  ICHECK(e == DCL_ERROR_NONE || e == DCL_ERROR_OP_NOT_FOUND)
      << "EdgeX Error: " << dclGetErrorString(e);
  EDGEX_CALL(dclopLoadWithLst(op_type, bin_buffer, bin_size, lst_buffer, lst_size));

  void* data[2] = {tensor_in->data, tensor_out->data};
  EDGEX_CALL(dclKernelLaunch(op_type, 2, data, stream));

  // get outputs
  EDGEX_CALL(dclrtSynchronizeStream(stream));
});

}  // namespace contrib
}  // namespace tvm
