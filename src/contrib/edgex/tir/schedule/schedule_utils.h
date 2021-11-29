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
#ifndef TVM_CONTRIB_EDGEX_TIR_SCHEDULE_SCHEDULE_UTILS_H_
#define TVM_CONTRIB_EDGEX_TIR_SCHEDULE_SCHEDULE_UTILS_H_

#include <tvm/tir/schedule/schedule.h>

#include <vector>

namespace tvm {
namespace tir {

Stmt Substitute(const Stmt& stmt, const Map<Stmt, Stmt>& replace_plan);

namespace schedule {

/*!
 * \brief Create a new loop with the given annotation added
 * \param loop The loop with original annotation
 * \param attr_key The annotation key to be added
 * \param attr_value The annotation value to be added
 * \return A new loop with the given annotation as its last annotation
 */
For WithAnnotation(const ForNode* loop, const String& attr_key, const ObjectRef& attr_value);

StmtSRef LowestCommonAncestor(const std::vector<StmtSRef>& nodes, const StmtSRef& root);

/*!
 * \brief Work around utility to replace stmt with different type.
 */
void ReplaceStmt(ScheduleState self, const StmtSRef& src, Stmt tgt_stmt,
                 const Map<Block, Block>& block_sref_reuse);
}  // namespace schedule
}  // namespace tir
}  // namespace tvm

#endif  // TVM_CONTRIB_EDGEX_TIR_SCHEDULE_SCHEDULE_UTILS_H_
