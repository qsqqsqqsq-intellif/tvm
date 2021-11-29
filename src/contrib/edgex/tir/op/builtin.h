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

/*! \brief Use integer to record the kind. */
using TNNPUnitKind = Integer;

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
 * \brief tvm intrinsic for nnp iss buffer handling.
 */
TVM_DLL const Op& nnp_iss_bind_input_buffer();

/*!
 * \brief tvm intrinsic for nnp veltadd intrinsic.
 */
TVM_DLL const Op& nnp_veltadd();

/*!
 * \brief tvm intrinsic for nnp vu madd and right shift in vacc.
 */
TVM_DLL const Op& nnp_vacc_madd_right_shift();

/*!
 * \brief tvm intrinsic for nnp round right shift.
 */
TVM_DLL const Op& nnp_round_right_shift();

}  // namespace builtin
}  // namespace edgex
}  // namespace tir
}  // namespace tvm
#endif  // TVM_CONTRIB_EDGEX_TIR_OP_BUILTIN_H_
