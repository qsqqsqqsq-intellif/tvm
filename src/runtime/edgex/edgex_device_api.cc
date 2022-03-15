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
 * \file edgex_device_api.cc
 * \brief GPU specific API
 */
extern "C" {
#include <dcl.h>
}
#include <dmlc/parameter.h>
#include <dmlc/thread_local.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

#include <cstring>
#include <map>

#include "edgex_common.h"

namespace tvm {
namespace runtime {

/**
 *! \brief Helper class to record the ddr address allocations.
 */
class EdgeXAddressRecorder {
 public:
  void RecordAlloc(void* addr, size_t bytes);

  void RecordFree(void* addr);

  /**
   *! \brief Get the start address and total bytes of the allocated range
   * the input address belongs to. Return (nullptr, 0) if the input address
   * is not valid.
   */
  std::pair<void*, size_t> QueryAddress(void* addr) const;

  /*! \brief we will add two AddrPoint into addr_map_ for each >1 bytes allocation:
   * begin point: begin_offset -> AddrPoint(begin_offset, bytes, true)
   * end point: (begin_offset + bytes - 1) -> AddrPoint(begin_offset, bytes, false)
   * then we can query the addr_map_ to search the allocation range for each address.
   */
  struct AddrPoint {
    size_t begin_offset;
    size_t bytes;
    bool is_begin;
  };

 private:
  std::map<size_t, AddrPoint> addr_map_;
};

class EdgeXDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(Device dev) final { EDGEX_CALL(dclrtSetDevice(dev.device_id)); }

  // TODO(@yiheng): GetAttr
  void GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) final {
    int value = 0;
    LOG(INFO) << "GetAttr not supported yet";
    *rv = value;
  }

  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) final {
    EDGEX_CALL(dclrtSetDevice(dev.device_id));
    // TODO(@yiheng): alignment 32
    // ICHECK_EQ(32 % alignment, 0U) << "EdgeX space is aligned at 32 bytes";
    void* ret;
    EDGEX_CALL(dclrtMalloc(&ret, nbytes, DCL_MEM_MALLOC_HUGE_FIRST));
    // TODO(team): alignment
    if (reinterpret_cast<uint64_t>(ret) % alignment != 0) {
      LOG(WARNING) << "DCL Allocation do not match alignment requirement " << alignment << ": "
                   << ret;
    }
    if (iss_debug_mode_) {
      addr_recorder_.RecordAlloc(ret, nbytes);
    }
    return ret;
  }

  void FreeDataSpace(Device dev, void* ptr) final {
    EDGEX_CALL(dclrtSetDevice(dev.device_id));
    EDGEX_CALL(dclrtFree(ptr));
    if (iss_debug_mode_) {
      addr_recorder_.RecordFree(ptr);
    }
  }

  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    dclrtStream dcl_stream = static_cast<dclrtStream>(stream);
    from = static_cast<const char*>(from) + from_offset;
    to = static_cast<char*>(to) + to_offset;

    // generally host memory is not allocated by dclrtMallocHost()
    const bool is_dcl_host = false;

    if (TVMDeviceExtType(dev_from.device_type) == kDLEdgeX &&
        TVMDeviceExtType(dev_to.device_type) == kDLEdgeX) {
      // device to device
      EDGEX_CALL(dclrtSetDevice(dev_from.device_id));
      if (dev_from.device_id == dev_to.device_id) {
        GPUCopy(from, to, size, DCL_MEMCPY_DEVICE_TO_DEVICE, dcl_stream);
      } else {
        // TODO(@yiheng): dclrtMemcpyPeerAsync
        // dclrtMemcpyPeerAsync(to, dev_to.device_id, from, dev_from.device_id, size, dcl_stream);
      }
    } else if (TVMDeviceExtType(dev_from.device_type) == kDLEdgeX && dev_to.device_type == kDLCPU) {
      // device to cpu
      EDGEX_CALL(dclrtSetDevice(dev_from.device_id));
      DeviceToHostCopy(from, to, size, dcl_stream, is_dcl_host);
    } else if (dev_from.device_type == kDLCPU && TVMDeviceExtType(dev_to.device_type) == kDLEdgeX) {
      // cpu to device
      EDGEX_CALL(dclrtSetDevice(dev_to.device_id));
      HostToDeviceCopy(from, to, size, dcl_stream, is_dcl_host);
    } else {
      LOG(FATAL) << "expect copy from/to GPU or between GPU";
    }
  }

  TVMStreamHandle CreateStream(Device dev) {
    EDGEX_CALL(dclrtSetDevice(dev.device_id));
    dclrtStream retval;
    EDGEX_CALL(dclrtCreateStream(&retval));
    return static_cast<TVMStreamHandle>(retval);
  }

  void FreeStream(Device dev, TVMStreamHandle stream) {
    EDGEX_CALL(dclrtSetDevice(dev.device_id));
    dclrtStream dcl_stream = static_cast<dclrtStream>(stream);
    EDGEX_CALL(dclrtDestroyStream(dcl_stream));
  }

  void SyncStreamFromTo(Device dev, TVMStreamHandle event_src, TVMStreamHandle event_dst) {
    EDGEX_CALL(dclrtSetDevice(dev.device_id));
    dclrtStream src_stream = static_cast<dclrtStream>(event_src);
    dclrtStream dst_stream = static_cast<dclrtStream>(event_dst);
    dclrtEvent evt;
    EDGEX_CALL(dclrtCreateEvent(&evt));
    EDGEX_CALL(dclrtRecordEvent(evt, src_stream));
    EDGEX_CALL(dclrtStreamWaitEvent(dst_stream, evt));
    EDGEX_CALL(dclrtDestroyEvent(evt));
  }

  void StreamSync(Device dev, TVMStreamHandle stream) final {
    EDGEX_CALL(dclrtSetDevice(dev.device_id));
    EDGEX_CALL(dclrtSynchronizeStream(static_cast<dclrtStream>(stream)));
  }

  void SetStream(Device dev, TVMStreamHandle stream) final {
    LOG(FATAL) << "SetStream not supported yet";
  }

  void* AllocWorkspace(Device dev, size_t size, DLDataType type_hint) final {
    return AllocDataSpace(dev, size, 32, type_hint);
  }

  void FreeWorkspace(Device dev, void* data) final { return FreeDataSpace(dev, data); }

  static EdgeXDeviceAPI* Global() {
    // NOTE: explicitly use new to avoid exit-time destruction of global state
    // Global state will be recycled by OS as the process exits.
    static auto* inst = new EdgeXDeviceAPI();
    return inst;
  }

  // NOTE: Global api instance is never deconstructed.
  ~EdgeXDeviceAPI() {}

  static void Finalize() {
    LOG(INFO) << "Finalize edgex environment";
    EDGEX_CALL(dclFinalize());
  }

  bool IsISSDebugMode() const { return iss_debug_mode_; }

  const EdgeXAddressRecorder& GetAddressRecorder() const { return addr_recorder_; }

 private:
  // Ensure EdgeXDeviceAPI as singelaton with dcl environment initialized.
  EdgeXDeviceAPI() {
    std::string cfg_file = dmlc::GetEnv("EDGEX_CLIENT_CONFIG", std::string(""));
    if (cfg_file.empty()) {
      EDGEX_CALL(dclInit(nullptr));
    } else {
      LOG(INFO) << "Use edgex client config file " << cfg_file;
      EDGEX_CALL(dclInit(cfg_file.c_str()));
    }
    // EDGEX_DEBUG_ISS could be on/off/interactive
    iss_debug_mode_ = dmlc::GetEnv("EDGEX_DEBUG_ISS", std::string("off")) != "off";
    atexit(EdgeXDeviceAPI::Finalize);
  }

  static void GPUCopy(const void* from, void* to, size_t size, dclrtMemcpyKind kind,
                      dclrtStream stream) {
    // TODO(@yiheng): toMax
    size_t toMax = size;
    if (stream != nullptr) {
      EDGEX_CALL(dclrtMemcpyAsync(to, toMax, from, size, kind, stream));
    } else {
      EDGEX_CALL(dclrtMemcpy(to, toMax, from, size, kind));
    }
  }

  static void HostToDeviceCopy(const void* from, void* to, size_t size, dclrtStream stream,
                               bool is_dcl_host) {
    // TODO(team): toMax
    size_t toMax = size;
    void* dcl_from = nullptr;
    if (is_dcl_host) {
      dcl_from = const_cast<void*>(from);
    } else {
      // dclrtMemcpy() do not support copy from non-dcl-managed host memory
      EDGEX_CALL(dclrtMallocHost(&dcl_from, size));
      memcpy(dcl_from, from, size);
    }
    if (stream != nullptr) {
      EDGEX_CALL(dclrtMemcpyAsync(to, toMax, dcl_from, size, DCL_MEMCPY_HOST_TO_DEVICE, stream));
    } else {
      EDGEX_CALL(dclrtMemcpy(to, toMax, dcl_from, size, DCL_MEMCPY_HOST_TO_DEVICE));
    }
    if (!is_dcl_host) {
      // free intermediate memory after copy done
      if (stream != nullptr) {
        EDGEX_CALL(dclrtSynchronizeStream(stream));
      }
      EDGEX_CALL(dclrtFreeHost(dcl_from));
    }
  }

  static void DeviceToHostCopy(const void* from, void* to, size_t size, dclrtStream stream,
                               bool is_dcl_host) {
    // TODO(team): toMax
    size_t toMax = size;
    void* dcl_to = nullptr;
    if (is_dcl_host) {
      dcl_to = to;
    } else {
      // dclrtMemcpy() do not support copy from non-dcl-managed host memory
      EDGEX_CALL(dclrtMallocHost(&dcl_to, toMax));
    }
    if (stream != nullptr) {
      EDGEX_CALL(dclrtMemcpyAsync(dcl_to, toMax, from, size, DCL_MEMCPY_DEVICE_TO_HOST, stream));
    } else {
      EDGEX_CALL(dclrtMemcpy(dcl_to, toMax, from, size, DCL_MEMCPY_DEVICE_TO_HOST));
    }
    if (!is_dcl_host) {
      // copy to non-dcl host mem and free intermediate memory
      if (stream != nullptr) {
        EDGEX_CALL(dclrtSynchronizeStream(stream));
      }
      memcpy(to, dcl_to, toMax);
      EDGEX_CALL(dclrtFreeHost(dcl_to));
    }
  }

  bool iss_debug_mode_{false};
  EdgeXAddressRecorder addr_recorder_;
};

void EdgeXAddressRecorder::RecordAlloc(void* addr, size_t bytes) {
  auto pair = QueryAddress(addr);
  CHECK(!pair.first) << "Address overlap with allocated";
  size_t offset = reinterpret_cast<size_t>(addr);
  addr_map_[offset] = AddrPoint{offset, bytes, true};
  if (bytes > 1) {
    addr_map_[offset + bytes - 1] = AddrPoint{offset, bytes, false};
  }
}

void EdgeXAddressRecorder::RecordFree(void* addr) {
  auto pair = QueryAddress(addr);
  size_t offset = reinterpret_cast<size_t>(addr);
  CHECK(pair.first == addr) << "Illegal freed address";
  addr_map_.erase(offset);
  if (pair.second > 1) {
    addr_map_.erase(offset + pair.second - 1);
  }
}

std::pair<void*, size_t> EdgeXAddressRecorder::QueryAddress(void* addr) const {
  size_t offset = reinterpret_cast<size_t>(addr);
  auto lower_bound = addr_map_.lower_bound(offset);
  if (lower_bound == addr_map_.end()) {
    return {nullptr, 0};
  }
  const AddrPoint& point = lower_bound->second;
  if (point.is_begin) {
    if (point.begin_offset == offset) {
      return {reinterpret_cast<void*>(offset), point.bytes};
    } else {
      CHECK(point.begin_offset > offset);
      return {nullptr, 0};
    }
  } else {
    CHECK(point.begin_offset <= offset);
    return {reinterpret_cast<void*>(point.begin_offset), point.bytes};
  }
}

std::pair<void*, size_t> EdgeXQueryDeviceAddress(void* addr) {
  EdgeXDeviceAPI* ptr = EdgeXDeviceAPI::Global();
  if (!ptr->IsISSDebugMode()) {
    LOG(WARNING) << "Address query is only supported in iss debug mode";
    return {nullptr, 0};
  }
  return ptr->GetAddressRecorder().QueryAddress(addr);
}

bool EdgeXIsISSDebugMode() { return EdgeXDeviceAPI::Global()->IsISSDebugMode(); }

TVM_REGISTER_GLOBAL("device_api.edgex").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = EdgeXDeviceAPI::Global();
  *rv = static_cast<void*>(ptr);
});

TVM_REGISTER_GLOBAL("device_api.edgex.finalize").set_body([](TVMArgs args, TVMRetValue* rv) {
  EdgeXDeviceAPI* ptr = EdgeXDeviceAPI::Global();
  ptr->Finalize();
});

}  // namespace runtime
}  // namespace tvm
