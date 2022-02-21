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

/*! \brief Mark underlying block should lower to nnp dma intrinsic. */
constexpr const char* nnp_dma_scope = "pragma_nnp_dma_scope";

/*! \brief Tag for conv co info. */
constexpr const char* nnp_num_co_scope = "pragma_nnp_num_co";

/*! \brief Vcore resource usage mark on primfunc, kth bit encode it uses vcu[k] */
constexpr const char* nnp_vcore_resource = "vcore_resource";

/*! \brief Relay op name annotation on block */
constexpr const char* relay_op_name = "relay_op_name";

/*! \brief Relay op attrs annotation on block */
constexpr const char* relay_op_attrs = "relay_op_attrs";

}  // namespace attr
}  // namespace tir
}  // namespace tvm
#endif  // TVM_CONTRIB_EDGEX_TIR_ATTRS_H_
