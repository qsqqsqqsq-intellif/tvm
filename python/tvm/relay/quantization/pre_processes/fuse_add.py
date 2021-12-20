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
# pylint: disable=unused-argument,inconsistent-return-statements
"""Automatic quantization toolkit."""

import tvm
from tvm.relay import transform


def fuse_add(mod):
    """fuse add"""

    optimize_pass = []
    optimize_pass.append(transform.FuseAdd())
    optimize_pass.append(transform.FoldConstant())
    optimize_pass.append(transform.InferType())

    optimize = tvm.transform.Sequential(optimize_pass, opt_level=3, name="optimize")
    with tvm.transform.PassContext(opt_level=3):
        mod = optimize(mod)
    return mod
