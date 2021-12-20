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
"""__init__"""

import os
import glob

__all__ = ("OPCONFIGS",)

OPCONFIGS = {}


def register_strategy(cls):
    name = cls.name
    assert name not in OPCONFIGS, "{} method exists".format(name)
    OPCONFIGS[name] = cls


for filename in glob.glob1(os.path.dirname(__file__), "*.py"):
    if filename in ["__init__.py"]:
        continue
    module_name = os.path.basename(filename).split(".py")[0]
    imported_module = __import__(
        "%s.%s" % (__package__, module_name),
        fromlist=[
            module_name,
        ],
    )
    idict = imported_module.__dict__

    for method_name in idict.get("__all__", idict.keys()):
        method = idict[method_name]
        register_strategy(method)
