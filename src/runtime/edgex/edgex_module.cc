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
 * \file edgex_module.cc
 */
#include "edgex_module.h"

extern "C" {
#include <dcl.h>
}
#include <dmlc/parameter.h>
#include <tvm/ir/module.h>
#include <tvm/runtime/registry.h>

#include <array>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../file_utils.h"
#include "../meta_data.h"
#include "../pack_args.h"
#include "../thread_storage_scope.h"
#include "edgex_common.h"

namespace tvm {
namespace runtime {

/*! \brief Get edgex global function */
const tvm::runtime::PackedFunc* EdgexFuncGetGlobal(const char* name) {
  auto* func = tvm::runtime::Registry::Get(name);
  CHECK(func) << "Cannot find " << name;
  return func;
}

/*! \brief Get iss start pc constant */
const int GetIssStartPC() {
  auto* func = EdgexFuncGetGlobal("tvm.edgex.get_iss_start_pc");
  int pc = (*func)("pm");
  return pc;
}

class EdgeXModuleNode : public runtime::ModuleNode {
 public:
  explicit EdgeXModuleNode(const std::unordered_map<std::string, std::string>& bin_map,
                           const std::unordered_map<std::string, std::string>& lst_map,
                           const std::string& fmt,
                           const std::unordered_map<std::string, FunctionInfo>& fmap,
                           const std::unordered_map<std::string, std::string>& asm_map,
                           const std::string& full_obj)
      : bin_map_(bin_map),
        lst_map_(lst_map),
        fmt_(fmt),
        f_map_(fmap),
        asm_map_(asm_map),
        full_obj_(full_obj) {
    debug_iss_ = dmlc::GetEnv("EDGEX_DEBUG_ISS", std::string("off")) != "off";
  }

  // destructor
  ~EdgeXModuleNode() {
    // for (size_t i = 0; i < module_.size(); ++i) {
    //   if (module_[i] != nullptr) {
    //     EDGEX_CALL(dclrtSetDevice(static_cast<int>(i)));
    //     EDGEX_DRIVER_CALL(dclmdlUnload(module_[i]));
    //   }
    // }
  }

  const char* type_key() const final { return "edgex"; }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final;

  void SaveToFile(const std::string& file_name, const std::string& format) final {
    CHECK(0) << "Unsupported function: SaveToFile (TODO:cww)";
    std::string fmt = GetFileFormat(file_name, format);
    std::string meta_file = GetMetaFilePath(file_name);
    if (fmt == "edgex") {
      // ICHECK_NE(edgex_source_.length(), 0);
      // SaveMetaDataToFile(meta_file, f_map_);
      // SaveBinaryToFile(file_name, edgex_source_);
    } else {
      ICHECK_EQ(fmt, fmt_) << "Can only save to format=" << fmt_;
      // SaveMetaDataToFile(meta_file, f_map_);
      // SaveBinaryToFile(file_name, bin_data_);
    }
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(fmt_);
    stream->Write(f_map_);
    stream->Write(bin_map_);
    stream->Write(lst_map_);
    stream->Write(asm_map_);
  }

  // TODO(@yiheng): edgex's fmt
  std::string GetSource(const std::string& format) final {
    CHECK(0) << "Unsupported function: GetSource (TODO:cww)";
    // if (format == fmt_) return bin_data_;
    // if (edgex_source_.length() != 0) {
    //   return edgex_source_;
    // } else {
    //   if (fmt_ == "ptx") return bin_data_;
    //   return "";
    // }
    return "";
  }

 private:
  void ISSDebugRun(const std::string& name, const TVMArgs& args, TVMRetValue* rv);

  // the binary bin data
  std::unordered_map<std::string, std::string> bin_map_;
  // the lst data
  std::unordered_map<std::string, std::string> lst_map_;
  // The format
  std::string fmt_;
  // function information table.
  std::unordered_map<std::string, FunctionInfo> f_map_;
  // The edgex source.
  std::unordered_map<std::string, std::string> asm_map_;
  // The full elf object contains all functions
  std::string full_obj_;

  bool debug_iss_;
};

PackedFunc EdgeXModuleNode::GetFunction(const std::string& name,
                                        const ObjectPtr<Object>& sptr_to_self) {
  ICHECK_EQ(sptr_to_self.get(), this);
  ICHECK_NE(name, symbol::tvm_module_main) << "Device function do not have main";
  if (f_map_.find(name) == f_map_.end()) {
    return PackedFunc();
  }

  if (full_obj_ != "") {
    const int start_pc = GetIssStartPC();
    auto* extract_bin = EdgexFuncGetGlobal("tvm.edgex.extract_bin_data");
    auto* bin2lst = EdgexFuncGetGlobal("tvm.edgex.bin2lst");

    TVMByteArray linked_obj_bytes{full_obj_.c_str(), full_obj_.size()};
    std::string kernel_bin = (*extract_bin)(linked_obj_bytes, name);
    CHECK_EQ(kernel_bin, bin_map_[name]) << "Mismatched kernel bin";

    TVMByteArray bin_bytes{kernel_bin.c_str(), kernel_bin.size()};
    std::string kernel_lst = (*bin2lst)(bin_bytes, start_pc);
    CHECK_EQ(kernel_bin, bin_map_[name]) << "Mismatched kernel lst";
  }

  if (EdgeXIsISSDebugMode()) {
    return PackedFunc([name, this](TVMArgs args, TVMRetValue* rv) { ISSDebugRun(name, args, rv); });
  }

  return PackedFunc([name, this](TVMArgs args, TVMRetValue* rv) {
    const std::string& bin_data = bin_map_[name];
    const std::string& lst_data = lst_map_[name];

    EDGEX_CALL(dclrtSetDevice(0));
    dclrtStream stream;
    EDGEX_CALL(dclrtCreateStream(&stream));

    dclError e = dclopUnloadByName(name.c_str());
    ICHECK(e == DCL_ERROR_NONE || e == DCL_ERROR_OP_NOT_FOUND)
        << "EdgeX Error: " << dclGetErrorString(e);
    EDGEX_CALL(dclopLoadWithLst(name.c_str(), bin_data.c_str(), bin_data.size(), lst_data.c_str(),
                                lst_data.size()));

    int arg_num = args.size();
    std::vector<void*> arg_buffers(arg_num);
    for (int i = 0; i < arg_num; ++i) {
      arg_buffers[i] = static_cast<void*>(args[i]);
    }
    EDGEX_CALL(dclKernelLaunch(name.c_str(), arg_num, arg_buffers.data(), stream));
    EDGEX_CALL(dclrtSynchronizeStream(stream));
  });
}

void EdgeXModuleNode::ISSDebugRun(const std::string& name, const TVMArgs& args, TVMRetValue* rv) {
  const std::string debug_mode = dmlc::GetEnv<std::string>("EDGEX_DEBUG_ISS", "");
  bool interactive = debug_mode == "interactive";
  std::string working_dir = dmlc::GetEnv<std::string>("EDGEX_DEBUG_WORKING_DIR", "");

  const std::string& bin_data = bin_map_[name];
  const std::string& lst_data = lst_map_[name];

  TVMByteArray bin_bytes({bin_data.data(), bin_data.size()});
  Array<NDArray> tensor_arr;
  for (auto i = 0; i < args.size(); ++i) {
    if (args[i].type_code() == kTVMNDArrayHandle) {
      tensor_arr.push_back(args[i]);
    } else if (args[i].type_code() == kTVMOpaqueHandle) {
      void* addr = args[i];
      std::pair<void*, size_t> addr_info = EdgeXQueryDeviceAddress(addr);
      CHECK(addr_info.first) << "Illegal device address " << addr;
      int64_t offset = reinterpret_cast<int64_t>(addr) - reinterpret_cast<int64_t>(addr_info.first);
      int64_t shape[1] = {static_cast<int64_t>(addr_info.second) - offset};
      DLTensor t;
      t.data = addr;
      t.ndim = 1;
      t.shape = shape;
      t.dtype = DLDataType{0, 8, 1};
      t.strides = nullptr;
      t.byte_offset = 0;
      t.device = Device{kDLCPU, 0};
      DLManagedTensor dl{t, nullptr};
      tensor_arr.push_back(NDArray::FromDLPack(&dl));
    } else {
      LOG(FATAL) << "Illegal iss run argument type " << ArgTypeCode2Str(args[i].type_code());
    }
  }
  auto* iss_func = EdgexFuncGetGlobal("tvm.edgex.launch_iss");
  (*iss_func)(bin_bytes, lst_data, tensor_arr, interactive, name, working_dir);
}

Module EdgeXModuleCreate(const std::unordered_map<std::string, std::string>& bin_map,
                         const std::unordered_map<std::string, std::string>& lst_map,
                         const std::string& fmt,
                         const std::unordered_map<std::string, FunctionInfo>& fmap,
                         const std::unordered_map<std::string, std::string>& asm_map,
                         const std::string& full_obj) {
  auto n = make_object<EdgeXModuleNode>(bin_map, lst_map, fmt, fmap, asm_map, full_obj);
  return Module(n);
}

Module EdgeXModuleCreateFromObjects(tvm::IRModule mod,
                                    const std::unordered_map<std::string, std::string>& obj_map,
                                    const std::string& working_dir) {
  const int start_pc = GetIssStartPC();
  auto* get_linked_obj = EdgexFuncGetGlobal("tvm.edgex.get_linked_obj");
  auto* extract_bin = EdgexFuncGetGlobal("tvm.edgex.extract_bin_data");
  auto* bin2lst = EdgexFuncGetGlobal("tvm.edgex.bin2lst");
  auto* merge_objs = EdgexFuncGetGlobal("tvm.edgex.create_full_kernels_obj");

  std::unordered_map<std::string, FunctionInfo> f_map;
  std::unordered_map<std::string, std::string> bin_map;
  std::unordered_map<std::string, std::string> lst_map;
  std::unordered_map<std::string, std::string> asm_map;
  std::vector<String> to_merge_objs;

  for (auto fitem : mod->functions) {
    // fill function info map, currently only single op function exists
    const std::string& kernel_name = fitem.first->name_hint;
    FunctionInfo& finfo = f_map[kernel_name];
    finfo.name = kernel_name;
    ICHECK(obj_map.count(kernel_name)) << kernel_name << "'s object is not generated";

    // linking per kernel
    const std::string& origin_obj = obj_map.at(kernel_name);
    TVMByteArray obj_bytes{origin_obj.c_str(), origin_obj.size()};
    String linked_obj = (*get_linked_obj)(obj_bytes, kernel_name, working_dir, true);
    to_merge_objs.push_back(linked_obj);

    // extract bin
    TVMByteArray linked_obj_bytes{linked_obj.c_str(), linked_obj.size()};
    String kernel_bin = (*extract_bin)(linked_obj_bytes, kernel_name);
    bin_map[kernel_name] = kernel_bin;

    // make lst
    TVMByteArray bin_bytes{kernel_bin.c_str(), kernel_bin.size()};
    String kernel_lst = (*bin2lst)(bin_bytes, start_pc);
    lst_map[kernel_name] = kernel_lst;
  }

  // create fat elf object which contains all kernel functions
  size_t num_objs = to_merge_objs.size();
  std::vector<TVMValue> tvm_values(num_objs + 1);
  std::vector<int> tvm_type_codes(num_objs + 1);
  TVMArgsSetter setter(tvm_values.data(), tvm_type_codes.data());
  setter(0, working_dir);
  std::vector<TVMByteArray> to_merge_bytes_ref(num_objs);
  for (size_t k = 0; k < num_objs; ++k) {
    to_merge_bytes_ref[k] = TVMByteArray{to_merge_objs[k].c_str(), to_merge_objs[k].size()};
    setter(k + 1, to_merge_bytes_ref[k]);
  }
  TVMRetValue rv;
  (*merge_objs).CallPacked(TVMArgs(tvm_values.data(), tvm_type_codes.data(), num_objs + 1), &rv);
  String full_obj = rv;

  // create module
  auto n = make_object<EdgeXModuleNode>(bin_map, lst_map, "edgex", f_map, asm_map, full_obj);

  // delete working directory
  // TODO(bxq): remove temp directory for ass outputs
  return Module(n);
}

Module EdgeXModuleCreateFromAsm(tvm::IRModule mod,
                                const std::unordered_map<std::string, std::string>& asm_map,
                                const std::string& working_dir) {
  // try extract op info from ir module
  // use function name as op name
  auto* ass_func = EdgexFuncGetGlobal("tvm.edgex.invoke_assembler");
  const int start_pc = GetIssStartPC();

  std::unordered_map<std::string, FunctionInfo> f_map;
  std::unordered_map<std::string, std::string> bin_map;
  std::unordered_map<std::string, std::string> lst_map;

  for (auto fitem : mod->functions) {
    const std::string& kernel_name = fitem.first->name_hint;
    auto& asm_code = asm_map.at(kernel_name);

    // invoke assembler
    std::string bin_dir = (*ass_func)(kernel_name, asm_code, start_pc, working_dir);
    // load bin data
    const std::string bin_path = bin_dir + "/" + kernel_name + ".bin";
    std::ifstream bin_file(bin_path);
    std::ostringstream bin_buffer;
    bin_buffer << bin_file.rdbuf();

    // load lst data
    const std::string lst_path = bin_dir + "/" + kernel_name + "_cpp.lst";
    std::ifstream lst_file(lst_path);
    std::ostringstream lst_buffer;
    lst_buffer << lst_file.rdbuf();

    // fill function info map, currently only single op function exists
    FunctionInfo& finfo = f_map[kernel_name];
    finfo.name = kernel_name;

    bin_map[kernel_name] = bin_buffer.str();
    lst_map[kernel_name] = lst_buffer.str();
  }

  // create module
  auto n = make_object<EdgeXModuleNode>(bin_map, lst_map, "edgex", f_map, asm_map, "");

  // delete working directory
  // TODO(bxq): remove temp directory for ass outputs
  return Module(n);
}

// Load module from module.
Module EdgeXModuleLoadFile(const std::string& file_name, const std::string& format) {
  CHECK(0) << "Unsupported function: EdgeXModuleLoadFile";
  return Module();
}

Module EdgeXModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::unordered_map<std::string, std::string> bin_map;
  std::unordered_map<std::string, std::string> lst_map;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::unordered_map<std::string, std::string> asm_map;
  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&bin_map);
  stream->Read(&lst_map);
  stream->Read(&asm_map);
  return EdgeXModuleCreate(bin_map, lst_map, fmt, fmap, asm_map, "");
}

TVM_REGISTER_GLOBAL("runtime.module.loadfile_exbin").set_body_typed(EdgeXModuleLoadFile);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_edgex").set_body_typed(EdgeXModuleLoadBinary);

}  // namespace runtime
}  // namespace tvm