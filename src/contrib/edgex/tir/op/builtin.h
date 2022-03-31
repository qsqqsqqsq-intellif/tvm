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
 * \file src/contrib/edgex/tir/op/builtin.h
 * \brief Edgex TIR builtin intrinsics.
 *
 * TIR builtin intrinsics are stored as tvm:Op.
 * They are processed in the same way as we process Ops.
 *
 * It is not necessary to create a function for every Op,
 * as we can obtain them through Op::Get.
 *
 * This file contains the most commonly used intrinsics or
 * those that have special semantics and need compiler support.
 *
 * NOTE: Please add the intrinsic as follow order:
 *       1.Add intrinsics in alphabetical order;
 *       2.The intrinsics block order is:
 *         load&store -> handshake -> others
 */
#ifndef TVM_CONTRIB_EDGEX_TIR_OP_BUILTIN_H_
#define TVM_CONTRIB_EDGEX_TIR_OP_BUILTIN_H_

#include <tvm/ir/op.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>

#include <string>

namespace tvm {
namespace tir {
namespace edgex {

/*! \brief Collection of builtin intrinsics as ops */
namespace builtin {

/*!
 * \brief tvm intrinsic for nnp load data from dm to bbuf operator.
 */
TVM_DLL const Op& nnp_bdma_load();

/*!
 * \brief tvm intrinsic for nnp load data from ddr to dm operator.
 */
TVM_DLL const Op& nnp_eidma_load();

/*!
 * \brief tvm intrinsic for nnp store data from dm to ddr operator.
 */
TVM_DLL const Op& nnp_eodma_store();

/*!
 * \brief tvm intrinsic for nnp load data from ddr to dm operator.
 */
TVM_DLL const Op& nnp_ewdma_load();

/*!
 * \brief tvm intrinsic for nnp load data from dm to ibuf operator.
 */
TVM_DLL const Op& nnp_idma_load();

/*!
 * \brief tvm intrinsic for nnp store data from obuf to dm operator.
 */
TVM_DLL const Op& nnp_odma_store();

/*!
 * \brief tvm intrinsic for nnp load data from dm to vm operator.
 */
TVM_DLL const Op& nnp_vidma_load();

/*!
 * \brief tvm intrinsic for nnp load data from dm to nlfc operator.
 */
TVM_DLL const Op& nnp_vidma_load_nlfc();

/*!
 * \brief tvm intrinsic for nnp store data from vm to dm operator.
 */
TVM_DLL const Op& nnp_vodma_store();

/*!
 * \brief tvm intrinsic for nnp load data from dm to wbuf operator.
 */
TVM_DLL const Op& nnp_wdma_load();

/*!
 * \brief tvm intrinsic for nnp handshake operator.
 */
TVM_DLL const Op& nnp_sync();

/*!
 * \brief tvm intrinsic for nnp cube configuration.
 */
TVM_DLL const Op& nnp_cube();

/*!
 * \brief tvm intrinsic for nnp cuid register read.
 */
TVM_DLL const Op& nnp_cuid();

/*!
 * \brief inline asm intrinsic for vcu computation.
 * nnp_inline_asm(constraint_str, asm_str, vector_factor,
 *                state_num, state_type_annotations...,
 *                input_num, input_args...,
 *                placeholder_num, placeholder_args...)
 * the inputs and states should match with constraint string, eg:
 * if the constraint_str is "={vv},=&{vv},=&{vv},={vacc},{vv},{vv}", then
 *    - the first "={vv}" denotes the output, resides in vv reg
 *    - the "=&{vv},=&{vv},={vacc}" part denotes there are 3 state arguments, the first two in vv
 * and the last in vacc
 *    - the final "{vv},{vv}" denotes there are two inputs, both in vv
 */
TVM_DLL const Op& nnp_inline_asm_vcu();

/*!
 * \brief tvm intrinsic for nnp iss buffer handling.
 */
TVM_DLL const Op& nnp_iss_bind_input_buffer();

/*!
 * \brief tvm intrinsic for nnp vcu resource locking.
 */
TVM_DLL const Op& nnp_lock_vcu();

/*!
 * \brief tvm intrinsic for nnp vcu resource unlocking.
 */
TVM_DLL const Op& nnp_unlock_vcu();

/*!
 * \brief tvm intrinsic for nnp veltadd intrinsic.
 */
TVM_DLL const Op& nnp_veltadd();

/*!
 * \brief tvm intrinsic for nnp vint intrinsic.
 */
TVM_DLL const Op& nnp_vint();

/*!
 * \brief tvm intrinsic for nnp round right shift.
 */
TVM_DLL const Op& nnp_round_right_shift();

/*!
 * \brief tvm intrinsic for nnp nlfc sigmoid.
 */
TVM_DLL const Op& nnp_nlfc_sigmoid();

}  // namespace builtin
}  // namespace edgex
}  // namespace tir
}  // namespace tvm
#endif  // TVM_CONTRIB_EDGEX_TIR_OP_BUILTIN_H_
