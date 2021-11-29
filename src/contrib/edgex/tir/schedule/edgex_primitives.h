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
#ifndef TVM_CONTRIB_EDGEX_TIR_SCHEDULE_EDGEX_PRIMITIVES_H_
#define TVM_CONTRIB_EDGEX_TIR_SCHEDULE_EDGEX_PRIMITIVES_H_

#include <tvm/tir/schedule/schedule.h>

#include <string>
#include <vector>

#include "./buffer_axis_rv.h"

namespace tvm {
namespace tir {
namespace schedule {

/*!
 * \brief Add annotation to a loop
 * \param loop_sref the loop of interest
 * \param pragma_type the attribute key
 * \param pragma_value the attribute value
 */
TVM_DLL void Pragma(ScheduleState self, const StmtSRef& loop_sref, const String& pragma_type,
                    const PrimExpr& pragma_value);

/*!
 * \brief Split a specified loop nests into multiple sub-loops.
 * \param loop_srefs The candidate loop axes to be partitioned for target block.
 * \param lazy Delay loop partition to lower stage.
 * \return The block refs after partitioned.
 */
TVM_DLL void LoopPartition(ScheduleState self, const Array<StmtSRef>& loop_srefs, bool lazy);

/*!
 * \brief add annotation to block
 * \param block_sref the block of interest
 * \param attr_key annotation key
 * \param annotation annotation value
 */
TVM_DLL void AnnotateBlock(ScheduleState self, const StmtSRef& block_sref,
                           const std::string& attr_key, const ObjectRef& annotation);

/*!
 * \brief get buffer axes of a block accessed buffer
 * \param block_sref the target block
 * \param buffer_idx the index of the buffer in block read/write buffers
 * \param is_write find in block write buffers, or else in read buffers
 * \return the buffer axes array
 */
TVM_DLL Array<BufferAxisRV> GetBlockAccessBufferAxes(ScheduleState self, StmtSRef block_sref,
                                                     int64_t buffer_idx, bool is_write);

/*!
 * \brief split a specified buffer axis.
 * \param buffer_axis the buffer axis to be split
 * \param nparts the extent of the new outer axis
 * \param factor the extent of the new inner axis
 * \return the buffer axes after splitting
 */
TVM_DLL Array<BufferAxisRV> SplitBuffer(ScheduleState self, BufferAxisRV buffer_axis,
                                        const PrimExpr& nparts, const PrimExpr& factor);

/*!
 * \brief split a specified buffer axis.
 * \param buffer_axis the buffer axis to be split
 * \param factor_rvs the extents of the split axes
 * \return the buffer axes after splitting
 */
TVM_DLL Array<BufferAxisRV> SplitBuffer(ScheduleState self, BufferAxisRV buffer_axis,
                                        const Array<Optional<PrimExpr>>& factor_rvs);

/*!
 * \brief fuse two consecutive buffer axis of one buffer.
 * \param outer_sref the outer axis
 * \param inner_sref the inner axis
 * \return the fused buffer axis
 */
TVM_DLL BufferAxisRV FuseBuffer(ScheduleState self, BufferAxisRV outer, BufferAxisRV inner);

/*!
 * \brief reorder a list of buffer axes
 * \param order the order of buffer axes
 */
TVM_DLL void ReorderBuffer(ScheduleState self, const Array<BufferAxisRV>& order);

/*!
 * \brief stack two buffer on specified axis
 * \param axis0 the first buffer axis at stacked dimension
 * \param axis1 the second buffer axis at stacked dimension
 */
TVM_DLL void StackBuffer(ScheduleState self, BufferAxisRV axis0, BufferAxisRV axis1);

/*!
 * \brief get belonging buffer of specific axis
 * \param axis the buffer axis
 */
TVM_DLL Buffer GetBufferOf(ScheduleState self, BufferAxisRV axis);

/*!
 * \brief replace belonging buffer of specific axis
 * \param stmt_sref the stmt the buffer is accessed in
 * \param origin_buffer the origin buffer to replace
 * \param new_buffer the new buffer to replace
 * \param load_rewrite_func function to rewrite buffer load access
 * \param store_rewrite_func function to rewrite buffer store access
 * \param region_rewrite_func function to rewrite buffer access region
 */
TVM_DLL void ReplaceBuffer(ScheduleState self, StmtSRef stmt_sref, Buffer origin_buffer,
                           Buffer new_buffer, PackedFunc load_rewrite_func,
                           PackedFunc store_rewrite_func, PackedFunc region_rewrite_func);

}  // namespace schedule
}  // namespace tir
}  // namespace tvm

#endif  // TVM_CONTRIB_EDGEX_TIR_SCHEDULE_EDGEX_PRIMITIVES_H_
