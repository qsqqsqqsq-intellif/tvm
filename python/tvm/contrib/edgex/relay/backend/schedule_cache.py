# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=protected-access, no-member, missing-function-docstring
"""Schedule cache python interface."""
import tvm
from tvm.relay.backend import _backend


@tvm._ffi.register_object
class ScheduleCache(tvm.Object):
    """Schedule cache object to work around te compiler"""

    def __init__(self):
        self.__init_handle_by_constructor__(_backend.CreateScheduleCache)

    def __enter__(self):
        _backend.ScheduleCacheEnterScope(self)
        return self

    def __exit__(self, ptype, value, trace):
        _backend.ScheduleCacheExitScope(self)

    @staticmethod
    def current():
        return _backend.ScheduleCacheCurrent()
