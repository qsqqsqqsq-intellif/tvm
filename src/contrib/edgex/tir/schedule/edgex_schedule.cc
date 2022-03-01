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
#include <tvm/tir/schedule/schedule.h>

#include "../../../../tir/schedule/analysis.h"
#include "../../../../tir/schedule/concrete_schedule.h"
#include "./edgex_primitives.h"

namespace tvm {
namespace tir {

class EdgexSchedule;

class EdgexScheduleNode : public ConcreteScheduleNode {
 public:
  friend class EdgexSchedule;

  void LoopPartition(const Array<LoopRV>& loop_rvs, bool lazy) {
    Array<StmtSRef> loop_srefs;
    for (const LoopRV& loop_rv : loop_rvs) {
      loop_srefs.push_back(this->GetSRef(loop_rv));
    }
    schedule::LoopPartition(this->state(), loop_srefs, lazy);
  }

  void AnnotateBlock(const BlockRV& block_rv, const std::string& attr_key,
                     const ObjectRef& annotation) {
    StmtSRef block_sref = this->GetSRef(block_rv);
    return schedule::AnnotateBlock(this->state(), block_sref, attr_key, annotation);
  }

  void Pragma(const LoopRV& loop_rv, const String& pragma_type, const PrimExpr& pragma_value) {
    StmtSRef loop_sref = this->GetSRef(loop_rv);
    return schedule::Pragma(this->state(), loop_sref, pragma_type, pragma_value);
  }

  Array<BufferAxisRV> GetBlockWriteBufferAxes(const BlockRV& block_rv, int64_t buffer_idx) {
    StmtSRef block_sref = this->GetSRef(block_rv);
    return schedule::GetBlockAccessBufferAxes(this->state(), block_sref, buffer_idx, true);
  }

  Array<BufferAxisRV> GetBlockReadBufferAxes(const BlockRV& block_rv, int64_t buffer_idx) {
    StmtSRef block_sref = this->GetSRef(block_rv);
    return schedule::GetBlockAccessBufferAxes(this->state(), block_sref, buffer_idx, false);
  }

  Array<BufferAxisRV> SplitBuffer(const BufferAxisRV& buffer_axis,
                                  const Array<Optional<PrimExpr>>& factor_rvs) {
    return schedule::SplitBuffer(this->state(), buffer_axis, factor_rvs);
  }

  BufferAxisRV FuseBuffer(Array<BufferAxisRV> buffer_axes) {
    CHECK(!buffer_axes.empty()) << "ValueError: 'fuse_buffer' requires at least 1 axis(s)";
    while (buffer_axes.size() >= 2) {
      BufferAxisRV inner = buffer_axes.back();
      buffer_axes.pop_back();
      BufferAxisRV outer = buffer_axes.back();
      buffer_axes.pop_back();
      BufferAxisRV fused = schedule::FuseBuffer(this->state(), outer, inner);
      buffer_axes.push_back(fused);
    }
    return buffer_axes[0];
  }

  void ReorderBuffer(const Array<BufferAxisRV>& order) {
    return schedule::ReorderBuffer(this->state(), order);
  }

  void StackBuffer(const BufferAxisRV& axis0, const BufferAxisRV& axis1) {
    return schedule::StackBuffer(this->state(), axis0, axis1);
  }

  Buffer GetBufferOf(const BufferAxisRV& buffer_axis) { return buffer_axis->buffer(); }

  void ReplaceBuffer(const BlockRV& block_rv, const Buffer& origin_buffer, const Buffer& new_buffer,
                     PackedFunc load_rewrite_func, PackedFunc store_rewrite_func,
                     PackedFunc region_rewrite_func) {
    StmtSRef stmt_sref = this->GetSRef(block_rv);
    return schedule::ReplaceBuffer(this->state(), stmt_sref, origin_buffer, new_buffer,
                                   load_rewrite_func, store_rewrite_func, region_rewrite_func);
  }

  bool CanComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv, bool preserve_unit_loop) {
    return tir::CanComputeAt(this->state(), this->GetSRef(block_rv), this->GetSRef(loop_rv),
                             preserve_unit_loop);
  }

  bool CanReverseComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv,
                           bool preserve_unit_loop) {
    return tir::CanReverseComputeAt(this->state(), this->GetSRef(block_rv), this->GetSRef(loop_rv),
                                    preserve_unit_loop);
  }

  bool CanComputeInline(const BlockRV& block_rv) {
    return tir::CanComputeInline(this->state(), this->GetSRef(block_rv));
  }

  bool CanReverseComputeInline(const BlockRV& block_rv) {
    return tir::CanReverseComputeInline(this->state(), this->GetSRef(block_rv));
  }

  static constexpr const char* _type_key = "tir.edgex.EdgexSchedule";
  TVM_DECLARE_BASE_OBJECT_INFO(EdgexScheduleNode, ConcreteScheduleNode);
};

class EdgexSchedule {
 public:
  static Schedule Create(IRModule mod, int64_t seed, int debug_mode) {
    ObjectPtr<EdgexScheduleNode> n = make_object<EdgexScheduleNode>();
    n->state_ = ScheduleState(mod, debug_mode);
    n->symbol_table_ = {};
    n->analyzer_ = std::make_unique<arith::Analyzer>();
    return Schedule(std::move(n));
  }

  static Schedule Create(PrimFunc func, int64_t seed, int debug_mode) {
    return Create(IRModule({{GlobalVar("main"), func}}), seed, debug_mode);
  }
};

TVM_REGISTER_NODE_TYPE(EdgexScheduleNode);
TVM_REGISTER_GLOBAL("tir.edgex.schedule.EdgexSchedule")
    .set_body_typed([](ObjectRef obj, int64_t seed, int debug_mode) -> Schedule {
      if (!obj.defined()) {
        LOG(FATAL) << "TypeError: Expects non-null input";
        throw;
      }
      if (const auto* func = obj.as<PrimFuncNode>()) {
        return EdgexSchedule::Create(GetRef<PrimFunc>(func), seed, debug_mode);
      }
      if (const auto* mod = obj.as<IRModuleNode>()) {
        return EdgexSchedule::Create(GetRef<IRModule>(mod), seed, debug_mode);
      }
      LOG(FATAL) << "TypeError: Expects `IRModule` or `PrimFunc`, but gets: " << obj->GetTypeKey();
      throw;
    });

TVM_REGISTER_GLOBAL("tir.edgex.schedule.ScheduleLoopPartition")
    .set_body_typed([](Schedule self, Array<LoopRV> loop_rvs, bool lazy) {
      auto edgex_schd = const_cast<EdgexScheduleNode*>(self.as<EdgexScheduleNode>());
      return edgex_schd->LoopPartition(loop_rvs, lazy);
    });

TVM_REGISTER_GLOBAL("tir.edgex.schedule.SchedulePragma")
    .set_body_typed([](Schedule self, LoopRV loop_rv, const String& attr_key,
                       const PrimExpr& attr_value) {
      auto edgex_schd = const_cast<EdgexScheduleNode*>(self.as<EdgexScheduleNode>());
      return edgex_schd->Pragma(loop_rv, attr_key, attr_value);
    });

TVM_REGISTER_GLOBAL("tir.edgex.schedule.ScheduleAnnotateBlock")
    .set_body_typed([](Schedule self, BlockRV block_rv, const String& attr_key,
                       const ObjectRef& annotation) {
      auto edgex_schd = const_cast<EdgexScheduleNode*>(self.as<EdgexScheduleNode>());
      return edgex_schd->AnnotateBlock(block_rv, attr_key, annotation);
    });

TVM_REGISTER_GLOBAL("tir.edgex.schedule.ScheduleGetBlockWriteBufferAxes")
    .set_body_typed([](Schedule self, BlockRV block_rv, int64_t buffer_idx) {
      auto edgex_schd = const_cast<EdgexScheduleNode*>(self.as<EdgexScheduleNode>());
      return edgex_schd->GetBlockWriteBufferAxes(block_rv, buffer_idx);
    });

TVM_REGISTER_GLOBAL("tir.edgex.schedule.ScheduleGetBlockReadBufferAxes")
    .set_body_typed([](Schedule self, BlockRV block_rv, int64_t buffer_idx) {
      auto edgex_schd = const_cast<EdgexScheduleNode*>(self.as<EdgexScheduleNode>());
      return edgex_schd->GetBlockReadBufferAxes(block_rv, buffer_idx);
    });

TVM_REGISTER_GLOBAL("tir.edgex.schedule.ScheduleSplitBuffer")
    .set_body_typed([](Schedule self, const BufferAxisRV& buffer_axis,
                       const Array<Optional<PrimExpr>>& factors) {
      auto edgex_schd = const_cast<EdgexScheduleNode*>(self.as<EdgexScheduleNode>());
      return edgex_schd->SplitBuffer(buffer_axis, factors);
    });

TVM_REGISTER_GLOBAL("tir.edgex.schedule.ScheduleFuseBuffer")
    .set_body_typed([](Schedule self, const Array<BufferAxisRV>& buffer_axes) {
      auto edgex_schd = const_cast<EdgexScheduleNode*>(self.as<EdgexScheduleNode>());
      return edgex_schd->FuseBuffer(buffer_axes);
    });

TVM_REGISTER_GLOBAL("tir.edgex.schedule.ScheduleReorderBuffer")
    .set_body_typed([](Schedule self, const Array<BufferAxisRV>& order) {
      auto edgex_schd = const_cast<EdgexScheduleNode*>(self.as<EdgexScheduleNode>());
      return edgex_schd->ReorderBuffer(order);
    });

TVM_REGISTER_GLOBAL("tir.edgex.schedule.ScheduleStackBuffer")
    .set_body_typed([](Schedule self, const BufferAxisRV& axis0, const BufferAxisRV& axis1) {
      auto edgex_schd = const_cast<EdgexScheduleNode*>(self.as<EdgexScheduleNode>());
      return edgex_schd->StackBuffer(axis0, axis1);
    });

TVM_REGISTER_GLOBAL("tir.edgex.schedule.ScheduleGetBufferOf")
    .set_body_typed([](Schedule self, const BufferAxisRV& axis) {
      auto edgex_schd = const_cast<EdgexScheduleNode*>(self.as<EdgexScheduleNode>());
      return edgex_schd->GetBufferOf(axis);
    });

TVM_REGISTER_GLOBAL("tir.edgex.schedule.ScheduleReplaceBuffer")
    .set_body_typed([](Schedule self, BlockRV block_rv, const Buffer& origin_buffer,
                       const Buffer& new_buffer, PackedFunc load_rewrite_func,
                       PackedFunc store_rewrite_func, PackedFunc region_rewrite_func) {
      auto edgex_schd = const_cast<EdgexScheduleNode*>(self.as<EdgexScheduleNode>());
      return edgex_schd->ReplaceBuffer(block_rv, origin_buffer, new_buffer, load_rewrite_func,
                                       store_rewrite_func, region_rewrite_func);
    });

TVM_REGISTER_GLOBAL("tir.edgex.schedule.CanComputeAt")
    .set_body_typed([](Schedule self, BlockRV block_rv, LoopRV loop_rv, bool preserve_unit_loop) {
      auto edgex_schd = const_cast<EdgexScheduleNode*>(self.as<EdgexScheduleNode>());
      return edgex_schd->CanComputeAt(block_rv, loop_rv, preserve_unit_loop);
    });

TVM_REGISTER_GLOBAL("tir.edgex.schedule.CanReverseComputeAt")
    .set_body_typed([](Schedule self, BlockRV block_rv, LoopRV loop_rv, bool preserve_unit_loop) {
      auto edgex_schd = const_cast<EdgexScheduleNode*>(self.as<EdgexScheduleNode>());
      return edgex_schd->CanReverseComputeAt(block_rv, loop_rv, preserve_unit_loop);
    });

TVM_REGISTER_GLOBAL("tir.edgex.schedule.CanComputeInline")
    .set_body_typed([](Schedule self, BlockRV block_rv) {
      auto edgex_schd = const_cast<EdgexScheduleNode*>(self.as<EdgexScheduleNode>());
      return edgex_schd->CanComputeInline(block_rv);
    });

TVM_REGISTER_GLOBAL("tir.edgex.schedule.CanReverseComputeInline")
    .set_body_typed([](Schedule self, BlockRV block_rv) {
      auto edgex_schd = const_cast<EdgexScheduleNode*>(self.as<EdgexScheduleNode>());
      return edgex_schd->CanReverseComputeInline(block_rv);
    });

}  // namespace tir
}  // namespace tvm
