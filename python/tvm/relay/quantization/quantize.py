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

import os
import logging
from tvm import relay
from .analyze import analyze_graph
from .collect import collect_stats
from .calibrate import calibrate_params
from .realize import realize_graph
from .post_process import post_process
from .debug import compare_statistics

LOGGER = logging.getLogger("quantize")
logging.basicConfig(level=logging.INFO)


class Quantize:
    """quantize"""

    def __init__(self, cls, config):
        self.model_name = cls.model_name
        self.pre_processed_mod = cls.pre_processed_mod
        self.ctx = cls.ctx
        self.target = cls.target
        self.dataset = cls.dataset
        self.node_id = cls.node_id
        self.id_node = cls.id_node
        self.config = config
        self.net_in_dtype = cls.net_in_dtype
        self.opt_level = cls.opt_level

        LOGGER.info("pre_process finish...")
        LOGGER.debug("afert pre_process, output: ")
        if isinstance(self.pre_processed_mod, relay.Function):
            LOGGER.debug(self.pre_processed_mod)
        else:
            LOGGER.debug(self.pre_processed_mod["main"])
        analyze_graph(self)
        LOGGER.info("[collect] start...")
        LOGGER.info("[collect] the calibrate_num is %d", cls.calibrate_num)
        collect_stats(self)
        calibrate_params(self)
        realize_graph(self)

        if cls.root_path is not None:
            save_path = os.path.join(cls.root_path, cls.model_name)
            statistics_path = os.path.join(save_path, "statistics")
            if not os.path.exists(statistics_path):
                os.makedirs(statistics_path)
        else:
            statistics_path = None
        if cls.compare_statistics:
            self.similarity = compare_statistics(self, "cosine", statistics_path)
        else:
            self.similarity = None
        # post only move fp16 to UInt8
        post_process(self)
