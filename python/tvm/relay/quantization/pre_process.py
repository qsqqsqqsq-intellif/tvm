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
# pylint: disable=unused-argument,inconsistent-return-statements,E1102
"""pre process"""

from .pre_processes import (
    leaky_relu,
    origin_pass,
    pattern_match,
    divide_to_multiply,
    ExpandAddParam,
)


def pre_process(cls, norm):
    mod = cls.origin_mod
    mod = leaky_relu(mod)
    mod = origin_pass(mod, norm)
    mod = divide_to_multiply(mod)
    mod = pattern_match(mod)
    mod = ExpandAddParam()(mod)
    cls.pre_processed_mod = mod
