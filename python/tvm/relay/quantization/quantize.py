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
# pylint: disable=unused-argument,inconsistent-return-statements,import-outside-toplevel
"""Automatic quantization toolkit."""

import os
import logging
import numpy as np
import tvm
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


def run_quantization(
    model_name: str, mod=None, params=None, config=None, fast_mode=True, use_gpu=False
):
    """Module to module quantization interface

    Parameters
    ----------
    model_name: str
        model name specification.

    mod : tvm.IRModule
        input relay module to run quantization on.

    params : dict[str, numpy.ndarray]
        input relay module params.

    config : dict
        quantization configurations.

    fast_mode : bool
        use small random datasets to get result rapidly.

    Returns
    -------
    (mod, params) pair of quantization result
    """
    # prepare mock data
    mod = relay.transform.InferType()(mod)
    if not fast_mode:
        raise ValueError("Do not know how to quantize when fast_mode == False")
    single_data = {}
    for param in mod["main"].params:
        input_name = param.name_hint
        dtype = param.checked_type.dtype
        shape = [int(_) for _ in param.checked_type.shape]
        if params is not None and input_name in params:
            continue  # skip model weight params
        data = np.random.randint(0, 64, shape).astype(dtype)
        single_data[input_name] = data
    dummy_mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    dummy_scale = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    def eval_nothing():
        return 0.0

    # call quantize search
    if use_gpu:
        ctx = tvm.gpu(0)
        target = "cuda"
    else:
        ctx = tvm.cpu()
        target = "llvm"
    quantize_search = relay.quantization.QuantizeSearch(
        model_name=model_name,
        mod=mod,
        params=params,
        dataset=lambda: iter([single_data]),
        calibrate_num=1,
        eval_func=eval_nothing,
        ctx=ctx,
        target=target,
        root_path=None,
        mean=dummy_mean,
        scale=dummy_scale,
        compare_statistics=False,
    )
    if config is None:
        config = quantize_search.get_default_config()
    quantize_search.quantize(config)
    quantized_mod = quantize_search.results[-1]["mod"]

    # test build and extract result
    relay.build(quantized_mod, target)
    from tvm.relay.quantization.post_processes.extract_module import ExtractParamsPass

    func, quantized_params = ExtractParamsPass().run(quantized_mod["main"])
    func = relay.frontend.common.infer_type(func)
    quantized_mod = tvm.ir.module.IRModule.from_expr(func)
    return quantized_mod, quantized_params
