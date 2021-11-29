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
#ifndef TVM_CONTRIB_EDGEX_TIR_SCHEDULE_BUFFER_AXIS_RV_H_
#define TVM_CONTRIB_EDGEX_TIR_SCHEDULE_BUFFER_AXIS_RV_H_

#include <tvm/runtime/object.h>

#include <utility>
#include <vector>

namespace tvm {
namespace tir {

class BufferAxisRVNode;

/*!
 * \brief representation for dimensions of buffer
 * The container is only accessed via BufferAxisRV and hold weak ref back to rvs
 * to help determine whether a particular rv is expired. After each buffer schedule,
 * the container's buffer should get updated to new transformed one. The block sref
 * which allocate the buffer or the function declare the buffer is also recorded.
 */
class BufferAxisRVContainerNode : public runtime::Object {
 public:
  Buffer buffer;
  std::vector<const BufferAxisRVNode*> axes;  // non-managed
  ObjectRef alloc_ref;                        // either global var or block stmt sref
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("buffer", &buffer);
    v->Visit("alloc_ref", &alloc_ref);
  }
  bool IsFunctionBuffer() const { return alloc_ref->IsInstance<GlobalVarNode>(); }
  bool IsAllocatedBuffer() const { return alloc_ref->IsInstance<StmtSRefNode>(); }
  static constexpr const char* _type_key = "tir.edgex.BufferAxisRVContainer";
  TVM_DECLARE_FINAL_OBJECT_INFO(BufferAxisRVContainerNode, runtime::Object);
};

/*!
 * \brief Managed reference to BufferAxisRVNodeContainer
 * \sa BufferAxisRVNode
 */
class BufferAxisRVContainer : public runtime::ObjectRef {
 public:
  /*! \brief constructor, actual init is done by `GetBufferAxisRVs` */
  BufferAxisRVContainer() { data_ = std::move(make_object<BufferAxisRVContainerNode>()); }
  BufferAxisRVContainerNode* get_mutable() {
    return static_cast<BufferAxisRVContainerNode*>(ObjectRef::get_mutable());
  }
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BufferAxisRVContainerNode);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(BufferAxisRVContainer, runtime::ObjectRef,
                                            BufferAxisRVContainerNode);
};

/*!
 *\brief representation for single dimension of one buffer.
 */
class BufferAxisRVNode : public runtime::Object {
 public:
  BufferAxisRVContainer container;
  PrimExpr extent;
  Buffer buffer() const { return container->buffer; }
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("container", &container);
    v->Visit("extent", &extent);
  }
  static constexpr const char* _type_key = "tir.edgex.BufferAxisRV";
  TVM_DECLARE_FINAL_OBJECT_INFO(BufferAxisRVNode, runtime::Object);
};

/*!
 * \brief Managed reference to BufferAxisRVNode
 * \sa BufferAxisRVNode
 */
class BufferAxisRV : public runtime::ObjectRef {
 public:
  /*! \brief constructor */
  explicit BufferAxisRV(BufferAxisRVContainer container, PrimExpr extent) {
    auto node = make_object<BufferAxisRVNode>();
    node->container = std::move(container);
    node->extent = std::move(extent);
    data_ = std::move(node);
  }
  BufferAxisRVNode* get_mutable() {
    return static_cast<BufferAxisRVNode*>(ObjectRef::get_mutable());
  }
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(BufferAxisRV, runtime::ObjectRef, BufferAxisRVNode);
};

}  // namespace tir
}  // namespace tvm
#endif  // TVM_CONTRIB_EDGEX_TIR_SCHEDULE_BUFFER_AXIS_RV_H_
