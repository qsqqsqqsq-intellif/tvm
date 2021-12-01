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
#include "./schedule_cache.h"

#include <tvm/runtime/object.h>
#include <tvm/support/with.h>

#include "../../../../relay/backend/te_compiler.h"

namespace tvm {
namespace relay {
namespace tec {

TVM_REGISTER_NODE_TYPE(ScheduleCacheNode);

TVM_REGISTER_GLOBAL("relay.backend.CreateScheduleCache").set_body_typed([]() {
  auto n = make_object<ScheduleCacheNode>();
  return ScheduleCache(n);
});
TVM_REGISTER_GLOBAL("relay.backend.ScheduleCacheEnterScope").set_body_typed([](ScheduleCache obj) {
  obj.EnterWithScope();
});
TVM_REGISTER_GLOBAL("relay.backend.ScheduleCacheExitScope").set_body_typed([](ScheduleCache obj) {
  obj.ExitWithScope();
});
TVM_REGISTER_GLOBAL("relay.backend.ScheduleCacheCurrent").set_body_typed(ScheduleCache::Current);

}  // namespace tec
}  // namespace relay
}  // namespace tvm
