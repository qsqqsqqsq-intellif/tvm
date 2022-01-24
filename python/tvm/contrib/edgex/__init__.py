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
"""Edgex APIs of TVM python package.
"""
from .edgex_runtime import *
from .edgex import *
from . import tir
from . import arith
from . import topi
from . import relay
from . import base
from . import uniform_schedule
from .config import get_cfg
from .base.edgexlog import EdgexLog as el

el.i("[TVM VERSION is {}]".format(get_cfg().TVM_VERSION))
