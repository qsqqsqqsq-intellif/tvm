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
# pylint: disable=unused-argument,inconsistent-return-statements,import-outside-toplevel,bad-continuation
"""Automatic quantization toolkit."""

import logging
import warnings
from collections.abc import Iterator
import numpy as np
import tvm
from tvm import relay
from .analyze import analyze_graph
from .collect import collect_stats
from .calibrate import calibrate_params
from .realize import realize_graph
from .post_process import post_process
from .debug import compare_statistics, compare_statistics_api

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
        self.calibrate_num = cls.calibrate_num
        self.calibrate_batch = cls.calibrate_batch
        self.save_path = cls.save_path

        LOGGER.info("pre_process finish...")
        LOGGER.debug("afert pre_process, output: ")
        if isinstance(self.pre_processed_mod, relay.Function):
            LOGGER.info(self.pre_processed_mod)
        else:
            LOGGER.info(self.pre_processed_mod["main"])
        analyze_graph(self)
        LOGGER.info("[collect] start...")
        LOGGER.info("[collect] the calibrate_num is %d", self.calibrate_num)

        collect_stats(self)
        calibrate_params(self)
        realize_graph(self)
        post_process(self)

        if cls.compare_statistics:
            self.similarity = compare_statistics(self, "cosine")
        else:
            self.similarity = None


def run_quantization(
    model_name: str,
    mod=None,
    params=None,
    config=None,
    mean=0,
    std=1,
    axis=1,
    root_path=".",
    fast_mode=None,
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

    Returns
    -------
    (mod, params) pair of quantization result
    """
    # prepare mock data
    mod = relay.transform.InferType()(mod)
    single_data = {}
    for param in mod["main"].params:
        input_name = param.name_hint
        dtype = param.checked_type.dtype
        shape = [int(_) for _ in param.checked_type.shape]
        if params is not None and input_name in params:
            continue  # skip model weight params
        data = np.random.randint(0, 256, shape).astype(dtype)
        single_data[input_name] = data

    def eval_nothing():
        return 0.0

    quantize_search = relay.quantization.QuantizeSearch(
        model_name=model_name,
        mod=mod,
        params=params,
        dataset=lambda: iter([single_data]),
        calibrate_num=1,
        eval_func=eval_nothing,
        ctx=tvm.cpu(),
        target="llvm",
        root_path=root_path,
        norm={
            "input": {
                "mean": mean,
                "std": std,
                "axis": axis,
            },
        },
        quantize_config=config,
        compare_statistics=False,
        verbose=True,
    )
    config = quantize_search.get_default_config()
    quantize_search.quantize(config)
    quantized_mod = quantize_search.results[-1]["mod"]
    quantized_mod = relay.transform.FoldConstant()(quantized_mod)

    # test build and extract result
    relay.build(quantized_mod, "llvm")
    from tvm.relay.quantization.post_processes.extract_module import ExtractParamsPass

    func, quantized_params = ExtractParamsPass().run(quantized_mod["main"])
    func = relay.frontend.common.infer_type(func)
    quantized_mod = tvm.ir.module.IRModule.from_expr(func)
    return quantized_mod, quantized_params


def check_dataset(sym, dataset, channel_last):
    """check the dataset format"""
    if len(sym.params) in [1, 2]:
        if len(sym.params[0].type_annotation.shape) == 4:
            param_shape_imm = sym.params[0].type_annotation.shape
            param_shape = [meber.value for meber in param_shape_imm]
            if param_shape[0] != 1:
                assert callable(
                    dataset
                ), "input batch>1, The dataset must be callable, like lambda:iter([dict])!!"
                assert isinstance(
                    dataset(), Iterator
                ), "input batch>1, The dataset() must be iterator!!!"

            elif len(sym.params) == 2 and sym.params[1].name_hint != "im_info":
                assert callable(
                    dataset
                ), "the model is two input, The dataset must callable, like lambda:iter([dict])!!"
                assert isinstance(
                    dataset(), Iterator
                ), "the model is two input, The dataset() must be iterator!!!"
            elif (channel_last and param_shape[3] not in [1, 3]) or (
                not channel_last and param_shape[1] not in [1, 3]
            ):
                assert callable(
                    dataset
                ), "input-ch not in [1, 3], The dataset must callable, like lambda:iter([dict])!!"
                assert isinstance(
                    dataset(), Iterator
                ), "the model input-ch not in [1, 3], The dataset() must be iterator!!!"
        else:
            assert callable(
                dataset
            ), "input0 len(shape) !=4, The dataset must be callable, like lambda:iter([dict])!!"
            assert isinstance(
                dataset(), Iterator
            ), "input0 len(shape) !=4, The dataset() must be iterator!!!"
    else:
        assert callable(
            dataset
        ), "inpu_num > 2 The dataset must be callable, like lambda:iter([dict])!!"
        assert isinstance(dataset(), Iterator), "inpu_num > 2 The dataset() must be iterator!!!"


def quantize(
    sym,
    params=None,
    dataset=None,
    prof_img_num=0,
    rgb_en=1,
    norm=None,
    channel_last=False,
    quantize_config=None,
    debug_level=-1,
    similarity_dataset=None,
    similarity_img_num=1,
    save_dir=None,
    eval_func=None,
    verbose=False,
    sync_outdtype=True,
):

    """quantize api for customer"""
    is_irmodule = 0 if isinstance(sym, relay.Function) else 1
    sym = sym["main"] if is_irmodule else sym

    if params:
        name_dict = {}
        for arg in sym.params:
            name = arg.name_hint
            if name in name_dict:
                name_dict[name] = None
            else:
                name_dict[name] = arg
        bind_dict = {}
        for k, v in params.items():
            if k not in name_dict:
                continue
            arg = name_dict[k]
            if arg is None:
                raise ValueError("Multiple args in the function have name %s" % k)
            bind_dict[arg] = tvm.relay.expr.const(v)
        sym = tvm.relay.expr.bind(sym, bind_dict)

    if dataset is None:
        warnings.warn("the dataset is None!!!")
        single_data = {}
        sym = relay.frontend.common.infer_type(sym)
        for param in sym.params:
            input_name = param.name_hint
            dtype = param.checked_type.dtype
            shape = [int(_) for _ in param.checked_type.shape]
            if params is not None and input_name in params:
                continue  # skip model weight params
            data = np.random.randint(0, 256, shape).astype(dtype)
            single_data[input_name] = data
        dataset = lambda: iter([single_data])
    else:
        check_dataset(sym, dataset, channel_last)

    def filter_config(quantize_config):
        """filter config"""
        func_config_name = [
            "calib_method",
            "float_list",
            "target",
            "skip_conv_layers",
            "adaquant_enable",
            "net_in_dtype",
            "bias_method",
        ]
        func_config = {}

        for k in quantize_config.keys():
            if k in func_config_name and quantize_config[k] is not None:
                func_config[k] = quantize_config[k]

        return func_config

    # proxy to quantizesearch api. if dataset not iterator, dataset is imgae_path(str)
    func_config = filter_config(quantize_config)
    real_dataset, real_img_dir = (None, dataset) if isinstance(dataset, str) else (dataset, None)
    rgb_str = "rgb" if rgb_en else "bgr"
    if "dtype_net_input" in quantize_config:
        net_in_dtype = quantize_config["dtype_net_input"]
    else:
        net_in_dtype = "uint8"

    debug_level_dict = {-1: (0, 0, 0), 0: (1, 0, 0), 1: (1, 1, 0), 2: (1, 1, 1)}
    check_similarity, check_layer_similarity, display_result = debug_level_dict[debug_level]

    sym = tvm.IRModule.from_expr(sym) if is_irmodule else sym

    if "transform" in relay.__dict__:
        quantize_search = relay.quantization.QuantizeSearch(
            mod=sym,
            params=params,
            dataset=real_dataset,
            image_path=real_img_dir,
            calibrate_num=prof_img_num,
            eval_func=eval_func,
            rgb=rgb_str,
            norm=norm,
            root_path=save_dir,
            channel_last=channel_last,
            net_in_dtype=net_in_dtype,
            compare_statistics=False,
            quantize_config=func_config,
            verbose=verbose,
        )
    else:
        with relay.quantize.qconfig(**quantize_config):
            quantize_search = relay.quantization.QuantizeSearch(
                mod=sym,
                params=params,
                dataset=real_dataset,
                image_path=real_img_dir,
                calibrate_num=prof_img_num,
                eval_func=eval_func,
                rgb=rgb_str,
                norm=norm,
                root_path=save_dir,
                channel_last=channel_last,
                net_in_dtype=net_in_dtype,
                compare_statistics=False,
                quantize_config=func_config,
            )
    config = quantize_search.get_default_config()
    quantize_search.quantize(config)

    if similarity_dataset is not None:
        if isinstance(similarity_dataset, str):
            quantize_search.image_path = similarity_dataset
            quantize_search.calibrate_num = similarity_img_num
            quantize_search.dataset = relay.quantization.default_data(quantize_search)
        else:
            assert callable(
                similarity_dataset,
            ), "similarity_dataset must callable, like lambda: Iter([dict])"
            assert isinstance(
                similarity_dataset(), Iterator
            ), "similarity_dataset() must be Iterator"
            quantize_search.dataset = similarity_dataset
            quantize_search.calibrate_num = len(list(quantize_search.dataset()))
    else:
        quantize_search.calibrate_num = min(quantize_search.calibrate_num, similarity_img_num)
        if quantize_search.calibrate_num == 0:
            quantize_search.calibrate_num = 1

    if check_similarity:
        quantize_search.use_default_eval = True
        quantize_search.eval_func = relay.quantization.default_eval(quantize_search)
        quantize_search.evaluate("post_process", config)

    if check_layer_similarity:
        quantize_search.quantize_instance.dataset = quantize_search.dataset
        quantize_search.quantize_instance.calibrate_num = quantize_search.calibrate_num
        compare_statistics_api(
            quantize_search.quantize_instance,
            "cosine",
            display_result,
            save_dir,
            quantize_search.imgs,
        )

    if eval_func is not None:
        quantize_search.use_default_eval = False
        quantize_search.eval_func = eval_func
        quantize_search.evaluate("post_process", config)

    if "transform" in relay.__dict__:
        quantized_mod = quantize_search.results[-1]["mod"]
        quantized_mod = relay.transform.FoldConstant()(quantized_mod)

        # test build and extract result
        relay.build(quantized_mod, "llvm")
        from tvm.relay.quantization.post_processes.extract_module import ExtractParamsPass

        func, quantized_params = ExtractParamsPass().run(quantized_mod["main"])
        func = relay.frontend.common.infer_type(func)
        quantized_mod = tvm.ir.module.IRModule.from_expr(func)
        return quantized_mod, quantized_params

    return quantize_search.results[-1]["mod"], quantize_search.nnp300_pre_processed_mod