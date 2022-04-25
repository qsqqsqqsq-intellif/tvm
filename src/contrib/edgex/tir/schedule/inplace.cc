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
#include "../../../../arith/ir_mutator_with_analyzer.h"
#include "../../../../tir/schedule/analysis.h"
#include "../../../../tir/schedule/utils.h"
#include "./edgex_primitives.h"
#include "./schedule_utils.h"

namespace tvm {
namespace tir {
namespace schedule {

void InplaceBufferUnsafe(ScheduleState self, StmtSRef block_sref, size_t read_idx,
                         size_t write_idx) {
  const BlockNode* block = block_sref->StmtAs<BlockNode>();
  ICHECK(block) << "Input sref should be a block";
  ICHECK_LT(read_idx, block->reads.size()) << "Read index out of bound " << read_idx;
  ICHECK_LT(write_idx, block->writes.size()) << "Write index out of bound " << write_idx;
  Buffer read_buffer = block->reads[read_idx]->buffer;
  Buffer write_buffer = block->writes[write_idx]->buffer;
  ReplaceBuffer(self, block_sref, write_buffer, read_buffer, nullptr, nullptr, nullptr);
}

}  // namespace schedule
}  // namespace tir
}  // namespace tvm
