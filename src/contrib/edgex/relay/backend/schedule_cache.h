
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
#ifndef TVM_CONTRIB_EDGEX_RELAY_BACKEND_SCHEDULE_CACHE_H_
#define TVM_CONTRIB_EDGEX_RELAY_BACKEND_SCHEDULE_CACHE_H_

#include <tvm/support/with.h>

#include <stack>
#include <string>
#include <unordered_map>
#include <utility>

#include "../../../../relay/backend/te_compiler.h"

namespace tvm {
namespace relay {
namespace tec {

using tvm::transform::PassContext;

class ScheduleCacheNode : public Object {
 public:
  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const char* _type_key = "ScheduleCache";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleCacheNode, Object);

 private:
  std::string GetUniqueName(std::string name) {
    for (size_t i = 0; i < name.length(); ++i) {
      if (name[i] == '.') name[i] = '_';
    }
    while (true) {
      auto it = name_map_.find(name);
      if (it == name_map_.end()) {
        name_map_[name] = 1;
        return name;
      } else {
        std::ostringstream os;
        os << name << "_" << it->second;
        ++(it->second);
        name = os.str();
      }
    }
    return name;
  }

  friend class ScheduleCache;
  std::unordered_map<int64_t, CachedFunc> cache_;
  std::unordered_map<std::string, int> name_map_;
};

class ScheduleCache : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(ScheduleCache, ObjectRef, ScheduleCacheNode);

  CachedFunc GetSchedule(int64_t key) const {
    auto it = get()->cache_.find(key);
    if (it == get()->cache_.end()) {
      return CachedFunc();
    } else {
      return it->second;
    }
  }

  void AddSchedule(int64_t key, CachedFunc cfunc) {
    auto n = static_cast<ScheduleCacheNode*>(get_mutable());
    n->cache_[key] = cfunc;
  }

  std::pair<int64_t, CachedFunc> Lower(const relay::Function& function, const Target& target) {
    tec::TECompiler compiler;
    CCacheKey ckey(function, target);
    auto n = static_cast<ScheduleCacheNode*>(get_mutable());
    With<Target> target_scope(target);
    auto cfunc =
        PrimFuncFor(function, target, [n](const String name) { return n->GetUniqueName(name); });
    int64_t hash = n->cache_.size();
    AddSchedule(hash, cfunc);
    return {hash, cfunc};
  }

  /*! \brief Entry to hold the context stack. */
  struct ScheduleCacheThreadLocalEntry {
    /*! \brief The current target context */
    std::stack<ScheduleCache> context_stack;
  };

  /*! \brief Thread local store to hold the context stack. */
  using ScheduleCacheThreadLocalStore = dmlc::ThreadLocalStore<ScheduleCacheThreadLocalEntry>;

  TVM_DLL static ScheduleCache Current() {
    ScheduleCacheThreadLocalEntry* entry = ScheduleCacheThreadLocalStore::Get();
    if (entry->context_stack.size() > 0) {
      return entry->context_stack.top();
    }
    return ScheduleCache();
  }

  friend class With<ScheduleCache>;
  TVM_DLL void EnterWithScope() {
    ScheduleCacheThreadLocalEntry* entry = ScheduleCacheThreadLocalStore::Get();
    entry->context_stack.push(*this);
  }
  TVM_DLL void ExitWithScope() {
    ScheduleCacheThreadLocalEntry* entry = ScheduleCacheThreadLocalStore::Get();
    ICHECK(!entry->context_stack.empty());
    ICHECK(entry->context_stack.top().same_as(*this));
    entry->context_stack.pop();
  }
};

}  // namespace tec
}  // namespace relay
}  // namespace tvm

#endif  // TVM_CONTRIB_EDGEX_RELAY_BACKEND_SCHEDULE_CACHE_H_
