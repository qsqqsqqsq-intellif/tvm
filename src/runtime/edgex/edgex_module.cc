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

const int DEFAULT_ISS_START_PC = 0x318;

/*! \brief Get iss start pc constant */
static int GetIssStartPC() { return dmlc::GetEnv<int>("EDGEX_ISS_START_PC", DEFAULT_ISS_START_PC); }

TVM_REGISTER_GLOBAL("tvm.edgex.get_iss_start_pc").set_body_typed(GetIssStartPC);

class EdgeXModuleNode : public runtime::ModuleNode {
 public:
  explicit EdgeXModuleNode(const std::unordered_map<std::string, std::string>& bin_map,
                           const std::unordered_map<std::string, std::string>& lst_map,
                           const std::string& fmt,
                           const std::unordered_map<std::string, EdgeXFunctionInfo>& fmap,
                           const std::unordered_map<std::string, std::string>& asm_map)
      : bin_map_(bin_map), lst_map_(lst_map), fmt_(fmt), f_map_(fmap), asm_map_(asm_map) {
    debug_iss_ = dmlc::GetEnv("EDGEX_DEBUG_ISS", std::string("")) == "true";
    // std::fill(module_.begin(), module_.end(), nullptr);
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
    // TODO(@yiheng): edgex's fmt
    if (fmt == "cu") {
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
    CHECK(0) << "Unsupported function: SaveToBinary (TODO:cww)";
    // stream->Write(fmt_);
    // stream->Write(f_map_);
    // stream->Write(bin_data_);
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

  // get a EXfunction from primary context in device_id
  // EXfunction GetFunc(int device_id, const std::string& func_name) {
  //   std::lock_guard<std::mutex> lock(mutex_);
  //   // TODO(@yiheng): dclmdlLoadData, dclmdlGetFunction, EXfunction, EXmodule
  //   // must recheck under the lock scope
  //   if (module_[device_id] == nullptr) {
  //     EDGEX_DRIVER_CALL(dclmdlLoadData(&(module_[device_id]), data_.c_str()));
  //   }
  //   EXfunction func;
  //   dclError result = dclmdlGetFunction(&func, module_[device_id], func_name.c_str());
  //   if (result != DCL_ERROR_NONE) {
  //     const char* msg;
  //     LOG(FATAL) << "EdgeX Error: dclmdlGetFunction " << func_name << " failed with error: " <<
  //     dclGetErrorString(result);
  //   }
  //   return func;
  // }

 private:
  void ISSDebugRun(const std::string& name, const TVMArgs& args, TVMRetValue* rv) {
    const std::string debug_mode = dmlc::GetEnv<std::string>("EDGEX_DEBUG_ISS", "");
    bool interactive = debug_mode == "interactive";
    std::string working_dir = dmlc::GetEnv<std::string>("EDGEX_DEBUG_WORKING_DIR", "");

    std::string iss_funcname = "tvm.edgex.launch_iss";
    auto* iss_func = tvm::runtime::Registry::Get(iss_funcname);
    CHECK(iss_func) << "Cannot find " << iss_funcname;

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
        int64_t offset =
            reinterpret_cast<int64_t>(addr) - reinterpret_cast<int64_t>(addr_info.first);
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
    (*iss_func)(bin_bytes, lst_data, tensor_arr, interactive, name, working_dir);
  }

  // the binary bin data
  std::unordered_map<std::string, std::string> bin_map_;

  // the lst data
  std::unordered_map<std::string, std::string> lst_map_;

  // The format
  std::string fmt_;

  // function information table.
  std::unordered_map<std::string, EdgeXFunctionInfo> f_map_;

  // The edgex source.
  std::unordered_map<std::string, std::string> asm_map_;

  // TODO(@yiheng): check this
  // the internal modules per GPU, to be lazily initialized.
  // std::array<EXmodule, kMaxNumGPUs> module_;
  // internal mutex when updating the module
  std::mutex mutex_;

  bool debug_iss_;
};

class EdgeXWrappedFunc {
 public:
  // initialize the EdgeX function.
  // TODO(@yiheng): thread
  void Init(EdgeXModuleNode* m, ObjectPtr<Object> sptr, const std::string& func_name,
            size_t num_void_args, const std::vector<std::string>& thread_axis_tags) {
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
    // std::fill(fcache_.begin(), fcache_.end(), nullptr);
  }
  // invoke the function with void arguments
  void operator()(TVMArgs args, TVMRetValue* rv, void** void_args) const {
    int device_id;
    EDGEX_CALL(dclrtGetDevice(&device_id));
    // if (fcache_[device_id] == nullptr) {
    //   fcache_[device_id] = m_->GetFunc(device_id, func_name_);
    // }

    // TODO(@yiheng): dclopLaunchKernel
    // EDGEX_DRIVER_CALL(dclopLaunchKernel(fcache_[device_id], void_args));
  }

 private:
  // internal module
  EdgeXModuleNode* m_;
  // the resource holder
  ObjectPtr<Object> sptr_;
  // The name of the function.
  std::string func_name_;
  // Device function cache per device.
  // mark as mutable, to enable lazy initialization
  // TODO(@yiheng): EXfunction
  // mutable std::array<EXfunction, kMaxNumGPUs> fcache_;
};

PackedFunc EdgeXModuleNode::GetFunction(const std::string& name,
                                        const ObjectPtr<Object>& sptr_to_self) {
  ICHECK_EQ(sptr_to_self.get(), this);
  ICHECK_NE(name, symbol::tvm_module_main) << "Device function do not have main";
  // auto it = fmap_.find(name);
  // if (it == fmap_.end()) return PackedFunc();
  // const EdgeXFunctionInfo& info = it->second;
  // EdgeXWrappedFunc f;
  // f.Init(this, sptr_to_self, name, info.arg_types.size(), info.thread_axis_tags);
  // return PackFuncPackedArg(f, info.arg_types);
  if (f_map_.find(name) == f_map_.end()) {
    return PackedFunc();
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

Module EdgeXModuleCreate(const std::string& bin_data, const std::string& lst_data,
                         const std::string& fmt,
                         const std::unordered_map<std::string, EdgeXFunctionInfo>& fmap,
                         const std::string& edgex_source) {
  CHECK(0) << "Unsupported function: EdgeXModuleCreate (TODO:cww)";
  // auto n = make_object<EdgeXModuleNode>(bin_data, lst_data, fmt, fmap, edgex_source);
  return Module();
}

Module EdgeXModuleCreateFromAsm(tvm::IRModule mod,
                                const std::unordered_map<std::string, std::string>& asm_map,
                                const std::string& working_dir) {
  // try extract op info from ir module
  // use function name as op name
  // ICHECK_EQ(mod->functions.size(), 1) << "Currently we only support single func module";
  const std::string ass_funcname = "tvm.edgex.invoke_assembler";
  auto* ass_func = tvm::runtime::Registry::Get(ass_funcname);
  CHECK(ass_func) << "Cannot find " << ass_funcname;
  const int start_pc = GetIssStartPC();

  std::unordered_map<std::string, EdgeXFunctionInfo> f_map;
  std::unordered_map<std::string, std::string> bin_map;
  std::unordered_map<std::string, std::string> lst_map;

  for (auto fitem : mod->functions) {
    const std::string& kernel_name = fitem.first->name_hint;
    auto& asm_code = asm_map.at(kernel_name);

    // invoke assembler
    std::string bin_dir = (*ass_func)(kernel_name, asm_code, start_pc, working_dir);

    // load bin and lst data
    std::string bin_path = bin_dir + "/" + kernel_name + ".bin";
    std::ifstream bin_file(bin_path);
    std::ostringstream bin_buffer;
    bin_buffer << bin_file.rdbuf();
    const std::string lst_path = bin_dir + "/" + kernel_name + "_cpp.lst";
    std::ifstream lst_file(lst_path);
    std::ostringstream lst_buffer;
    lst_buffer << lst_file.rdbuf();

    // fill function info map, currently only single op function exists
    EdgeXFunctionInfo& finfo = f_map[kernel_name];
    finfo.name = kernel_name;

    bin_map[kernel_name] = bin_buffer.str();
    lst_map[kernel_name] = lst_buffer.str();
  }

  // create module
  auto n = make_object<EdgeXModuleNode>(bin_map, lst_map, "edgex", f_map, asm_map);

  // delete working directory
  // TODO(bxq): remove temp directory for ass outputs
  return Module(n);
}

// Load module from module.
Module EdgeXModuleLoadFile(const std::string& file_name, const std::string& format) {
  std::string bin_data;
  std::string lst_data;
  std::unordered_map<std::string, EdgeXFunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &bin_data);
  // LoadMetaDataFromFile(meta_file, &fmap);
  return EdgeXModuleCreate(bin_data, lst_data, fmt, fmap, std::string());
}

Module EdgeXModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string bin_data;
  std::string lst_data;
  std::unordered_map<std::string, EdgeXFunctionInfo> fmap;
  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&bin_data);
  stream->Read(&lst_data);
  return EdgeXModuleCreate(bin_data, lst_data, fmt, fmap, std::string());
}

TVM_REGISTER_GLOBAL("runtime.module.loadfile_exbin").set_body_typed(EdgeXModuleLoadFile);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_edgex").set_body_typed(EdgeXModuleLoadBinary);

}  // namespace runtime
}  // namespace tvm
