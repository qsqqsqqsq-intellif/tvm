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
#include "../../../../tir/schedule/analysis.h"
#include "./edgex_primitives.h"
#include "./schedule_utils.h"

namespace tvm {
namespace tir {
namespace schedule {

void Pragma(ScheduleState self, const StmtSRef& loop_sref, const String& pragma_type,
            const PrimExpr& pragma_value) {
  const auto* loop_ptr = loop_sref->StmtAs<ForNode>();
  CHECK(loop_ptr) << "TypeError: pragma expects a Loop as its first argument";
  self->Replace(loop_sref, WithAnnotation(loop_ptr, "pragma_" + pragma_type, pragma_value), {});
}

void AnnotateBlock(ScheduleState self, const StmtSRef& block_sref, const std::string& attr_key,
                   const ObjectRef& annotation) {
  const auto* block_ptr = block_sref->StmtAs<BlockNode>();
  CHECK(block_ptr) << "TypeError: block annotation expects 'block' as its argument";
  Map<String, ObjectRef> annotations = block_ptr->annotations;
  ICHECK(!annotations.count(attr_key)) << "Duplicate annotation key " << attr_key;
  annotations.Set(attr_key, annotation);
  ObjectPtr<BlockNode> n = make_object<BlockNode>(*block_ptr);
  n->annotations = std::move(annotations);
  Block new_block = Block(n);
  self->Replace(block_sref, new_block, {{GetRef<Block>(block_ptr), new_block}});
}

}  // namespace schedule
}  // namespace tir
}  // namespace tvm
