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
 * \file tvm/contrib/edgex/tir/attrs.h
 * \brief edgex tir attributes.
 */
#ifndef TVM_CONTRIB_EDGEX_TIR_ATTRS_H_
#define TVM_CONTRIB_EDGEX_TIR_ATTRS_H_

#include <tvm/tir/expr.h>

namespace tvm {
namespace tir {
namespace attr {

/******************************** NLFC related attrs ***********************************/
/*! \brief Attr key for FEdgexGetNlfcOp */
constexpr const char* kFEdgexGetNlfcOp = "FEdgexGetNlfcOp";
/*! \brief Attr key for nlfc op information */
constexpr const char* kNlfcOpInfo = "NlfcOpInfo";
/*! \brief Primfunc attr to mark nlfc param vars */
constexpr const char* kNlfcTableParams = "NlfcTableParams";
/*! \brief Primfunc attr to mark nlfc table data */
constexpr const char* kNlfcTableData = "NlfcTableData";

/******************************** Schedule pragma keys *********************************/

/*! \brief Mark underlying block should lower to nnp dma intrinsic. */
constexpr const char* nnp_dma_scope = "pragma_nnp_dma_scope";

/*! \brief Tag for conv co info. */
constexpr const char* nnp_num_co_scope = "pragma_nnp_num_co";

/*! \brief Mark allow vectorized scattered store in underlying scope. */
constexpr const char* nnp_scatter_store_scope = "pragma_nnp_scatter_store_scope";

/*! \brief Mark allow vectorized gathered load in underlying scope. */
constexpr const char* nnp_gather_load_scope = "pragma_nnp_gather_load_scope";

/*! \brief Vcore resource usage mark on primfunc, kth bit encode it uses vcu[k] */
constexpr const char* nnp_vcore_resource = "vcore_resource";

/*! \brief Relay op name annotation on block */
constexpr const char* relay_op_name = "relay_op_name";

/*! \brief Relay op attrs annotation on block */
constexpr const char* relay_op_attrs = "relay_op_attrs";

/*! \brief Tag for op data layout format. */
constexpr const char* nnp_data_layout = "pragma_nnp_data_layout";

/*! \brief Mark underlying block should codegen as standalone function,
    value=0 denotes cu func, value=1 denotes vcu func. */
constexpr const char* nnp_local_func_scope = "pragma_nnp_local_func_scope";

}  // namespace attr
}  // namespace tir
}  // namespace tvm
#endif  // TVM_CONTRIB_EDGEX_TIR_ATTRS_H_
