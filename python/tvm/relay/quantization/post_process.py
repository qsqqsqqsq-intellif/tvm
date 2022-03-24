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
# pylint: disable=unused-argument,unused-import,inconsistent-return-statements,bare-except
"""post process"""

import logging
from tvm import relay

# compatible with nnp300
try:
    from .post_processes import (
        eliminate_quantize_dequantize,
        eliminate_dequantize_quantize,
        extract_module,
    )
except:
    pass
from .post_processes import remove_input_quantize


LOGGER = logging.getLogger("quantize")


def post_process(cls):
    """post_process"""
    mod = cls.post_processed_mod
    if "adaquant_enable" in cls.config and cls.config["adaquant_enable"] == True:
        mod = eliminate_quantize_dequantize(mod)
        mod = eliminate_dequantize_quantize(mod)
        mod = relay.transform.ConvertQnnOp()(mod)
        mod = relay.transform.FoldConstant()(mod)

    mod = remove_input_quantize(mod, cls.net_in_dtype)
    LOGGER.info("[post_process]: ")
    if isinstance(mod, relay.Function):
        LOGGER.info(mod)
    else:
        LOGGER.info(mod["main"])
    cls.post_processed_mod = mod
