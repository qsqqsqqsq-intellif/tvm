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
 * \file edgex_ir_utils.h
 * \brief Helper functions to edgex tirs.
 */
#ifndef TVM_CONTRIB_EDGEX_TIR_EDGEX_IR_UTILS_H_
#define TVM_CONTRIB_EDGEX_TIR_EDGEX_IR_UTILS_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>

#include <string>

#include "../../../runtime/thread_storage_scope.h"
#include "../../../tir/transforms/ir_utils.h"

namespace tvm {
namespace tir {
namespace edgex {

/*! \brief NNP Data type id. */
enum NNPDataType { INT8 = 0, UINT8 = 1, FLOAT16 = 2, FLOAT32 = 3, INT32 = 4, INT16 = 5 };

/*! \brief NNP unit kind enum */
enum class NNPUnitKind : int {
  // if stmt, means a stmt should be kept on both vcu/cu.
  // if expr, means a expr can be evaluated safely on either vcu/cu.
  ALL = 0,

  // if stmt, means a stmt should be on vcu only.
  // if expr, means a expr can be evaluated only on vcu.
  VCU = 1,

  // if stmt, means a stmt should be on cu only.
  // if expr, means a expr can be evaluated only on cu.
  CU = 2
};

std::ostream& operator<<(std::ostream& os, NNPUnitKind kind);

/*! \brief Use integer to record the kind. */
using TNNPUnitKind = Integer;

/*!
 * \brief utility to determine whether a call node's
 *   op is a nnp intrinsic operator.
 */
inline bool IsNNPIntrinsic(RelayExpr op) {
  auto op_node = op.as<OpNode>();
  if (!op_node) {
    return false;
  }
  std::string name = op_node->name;
  return name.find("nnp_") != std::string::npos;
}

/*!
 * \brief utility to determine whether a call node's
 *   op is a nnp dma intrinsic operator.
 */
inline bool IsNNPDMAIntrinsic(RelayExpr op) {
  auto op_node = op.as<OpNode>();
  if (!op_node) {
    return false;
  }
  std::string name = op_node->name;
  return name.find("nnp_") != std::string::npos && (name.find("dma_load") != std::string::npos ||
                                                    name.find("dma_store") != std::string::npos);
}

/*!
 * \brief utility to add key=value string argument to nnp intrinsic.
 */
template <typename ValueT>
void NNPAddArg(CallNode* op, const std::string& key, const ValueT& value) {
  std::stringstream ss;
  ss << key << "=";
  std::string prefix = ss.str();
  ss << value;
  for (const PrimExpr& arg : op->args) {
    if (const StringImmNode* sn = arg.as<StringImmNode>()) {
      std::string exist_argval = sn->value;
      std::size_t pos = exist_argval.find("=");
      if (pos != std::string::npos) {
        std::string sub_str = exist_argval.substr(0, pos + 1);
        CHECK(strcmp(sub_str.c_str(), prefix.c_str()) != 0)
            << "Duplicate nnp argument key " << key << " at " << GetRef<Call>(op);
      }
    }
  }
  op->args.push_back(StringImm(ss.str()));
}

/*!
 * \brief get dst argument of dma intrinsic call.
 * ensure that the result is a valid tvm_access_ptr() call.
 */
inline const CallNode* NNPGetDmaDst(const CallNode* call) {
  CHECK_GE(call->args.size(), 3U);
  const CallNode* res = call->args[1].as<CallNode>();
  CHECK(res != nullptr);
  CHECK(res->op.same_as(tvm::tir::builtin::tvm_access_ptr())) << call->args[1];
  CHECK_EQ(res->args.size(), 5U);
  return res;
}

/*!
 * \brief get src argument of dma intrinsic call.
 * ensure that the result is a valid tvm_access_ptr() call.
 */
inline const CallNode* NNPGetDmaSrc(const CallNode* call) {
  CHECK_GE(call->args.size(), 3U);
  const CallNode* res = call->args[2].as<CallNode>();
  CHECK(res != nullptr);
  CHECK(res->op.same_as(tvm::tir::builtin::tvm_access_ptr()));
  CHECK_EQ(res->args.size(), 5U);
  return res;
}

/*!
 * \brief get output buffer access of cube intrinsic call.
 * ensure that the result is a valid tvm_access_ptr() call.
 */
inline const CallNode* NNPGetCubeOutput(const CallNode* call) {
  CHECK_GE(call->args.size(), 3U);
  const CallNode* res = call->args[0].as<CallNode>();
  CHECK(res != nullptr);
  CHECK(res->op.same_as(tvm::tir::builtin::tvm_access_ptr()));
  CHECK_EQ(res->args.size(), 5U);
  return res;
}

/*!
 * \brief get input buffer access of cube intrinsic call.
 * ensure that the result is a valid tvm_access_ptr() call.
 */
inline const CallNode* NNPGetCubeInput(const CallNode* call) {
  CHECK_GE(call->args.size(), 3U);
  const CallNode* res = call->args[1].as<CallNode>();
  CHECK(res != nullptr);
  CHECK(res->op.same_as(tvm::tir::builtin::tvm_access_ptr()));
  CHECK_EQ(res->args.size(), 5U);
  return res;
}

/*!
 * \brief get weight buffer access of cube intrinsic call.
 * ensure that the result is a valid tvm_access_ptr() call.
 */
inline const CallNode* NNPGetCubeWeight(const CallNode* call) {
  CHECK_GE(call->args.size(), 3U);
  const CallNode* res = call->args[2].as<CallNode>();
  CHECK(res != nullptr);
  CHECK(res->op.same_as(tvm::tir::builtin::tvm_access_ptr()));
  CHECK_EQ(res->args.size(), 5U);
  return res;
}

/*!
 * \brief Get the parameter's value of the op(intrinsic).
 * \param call The op(intrinsic) to get parameter's value.
 * \param key The key of the op(intrinsic)'s parameter.
 * \return The key's value of the specified op.
 */
int GetValueByKey(const CallNode* call, const std::string& key);

}  // namespace edgex

/*! \brief Get storage scope from buffer var. */
inline runtime::StorageScope GetStorageScope(const Var& buffer_var) {
  return runtime::StorageScope::Create(GetPtrStorageScope(buffer_var));
}

/*! \brief Get element datatype from buffer var. */
DataType GetBufferElementType(const Var& buffer_var);

/*!
 * \brief Align the size by specified align.
 * \param size The expr need aligned.
 * \param align The specified align parameter.
 * \return The aligned expr.
 */
inline PrimExpr Align(const PrimExpr& size, int align = 4) {
  return floordiv(size + (align - 1), align) * align;
}

/*!
 * \brief Align the size by specified align.
 * \param size The size need aligned.
 * \param align The specified align parameter.
 * \return The aligned size.
 */
inline int64_t Align(int64_t size, int align = 4) { return (size - 1 + align) / align * align; }

}  // namespace tir
}  // namespace tvm

#endif  // TVM_CONTRIB_EDGEX_TIR_EDGEX_IR_UTILS_H_
