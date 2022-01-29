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
 * \file edgex_transform.h
 * \brief TIR specific transformation passes for edgex.
 */
#ifndef TVM_CONTRIB_EDGEX_TIR_TRANSFORM_EDGEX_TRANSFORM_H_
#define TVM_CONTRIB_EDGEX_TIR_TRANSFORM_EDGEX_TRANSFORM_H_

#include <tvm/ir/transform.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>

#include <string>

namespace tvm {
namespace tir {
namespace transform {

using tvm::transform::Pass;
using tvm::transform::PassContext;
using tvm::transform::PassContextNode;
using tvm::transform::PassInfo;
using tvm::transform::PassInfoNode;
using tvm::transform::PassNode;
using tvm::transform::Sequential;

/*!
 * \brief Inject handshake intrinsics.
 * \return The pass.
 */
TVM_DLL Pass InjectHandShakeIntrin();

/*!
 * \brief Inject dma intrinsics.
 * \return The pass.
 */
TVM_DLL Pass InjectDmaIntrin();

/*!
 * \brief Use the existed isa's value calculate the
 *  complicated isa's value, such as epsilon, delta, etc.
 * \return The pass
 */
TVM_DLL Pass InjectCalculatedIsa();

/*!
 * \brief Inline all sub-primfunc calls
 * \param extern_prim_funcs  External function dict that allow inline `call_externed`
 * \return The pass
 */
TVM_DLL Pass InlinePrimFuncCalls(Map<String, PrimFunc> extern_prim_funcs = {});

/*!
 * \brief Inline all sub-primfunc calls
 * \param func  Primfunc to conduct inline transformation
 * \param import_module  Module to search sub functions
 * \param extern_prim_funcs  External function dict that allow inline `call_externed`
 * \return The inlined primfunc
 */
TVM_DLL PrimFunc InlinePrimFuncCalls(PrimFunc func, IRModule import_module,
                                     Map<String, PrimFunc> extern_prim_funcs = {});

/*!
 * \brief Handle the flat storage address or memory size according to the constraint.
 * \return The pass.
 */
TVM_DLL Pass FlatStorageConstraintHandler();

/*!
 * \brief Lift global scope buffer allocations out of device scope.
 * \return The pass.
 */
TVM_DLL Pass LiftGlobalAllocation();

/*!
 * \brief Rewrite vcu computation pattern to vectorized intrinsics.
 * \return The pass.
 */
TVM_DLL Pass RewriteVcuOps();

/*!
 * \brief Split cu/vcu control flow codes into separate code segments.
 * \return The pass.
 */
TVM_DLL Pass SplitVcuControlFlow();

/*!
 * \brief Rewrite buffer allocation into static memory addresses.
 * \return The pass.
 */
TVM_DLL Pass StorageRewriteNNP400();

}  // namespace transform
}  // namespace tir
}  // namespace tvm

#endif  // TVM_CONTRIB_EDGEX_TIR_TRANSFORM_EDGEX_TRANSFORM_H_
