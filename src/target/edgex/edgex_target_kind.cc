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
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

namespace tvm {

/*! \brief relay to tir target specification */
namespace relay {
namespace transform {

/*!
 * \brief relay_to_tir pass implementation
 *
 */
tvm::transform::Pass EdgeXRelayToTIR(String entry_name, PackedFunc renamer,
                                     bool post_schedule_rewrite, bool fold_constants);

}  // namespace transform
}  // namespace relay

TVM_REGISTER_TARGET_KIND("edgex", kDLEdgeX)
    .add_attr_option<Array<String>>("mattr")
    .add_attr_option<String>("mcpu")
    .add_attr_option<String>("mtriple")
    .add_attr_option<String>("mfloat-abi")
    .add_attr_option<Bool>("system-lib")
    .add_attr_option<String>("runtime")
    .add_attr_option<Bool>("link-params", Bool(false))
    .set_default_keys({"edgex"})
    .set_attr<FTVMRelayToTIR>("RelayToTIR",
                              relay::transform::EdgeXRelayToTIR("main", nullptr, true, true));

}  // namespace tvm
