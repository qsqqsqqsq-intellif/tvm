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
# pylint: disable=unused-argument,inconsistent-return-statements,bare-except,bad-continuation
"""Automatic quantization toolkit."""

import os
import json
import time
import numpy
import pandas
import tvm
from tvm import relay
from .name import get_name
from .config import config_space
from .quantize import Quantize

# compatible with nnp300
try:
    from .pre_process import pre_process
except:
    pass
from .relay_viz import RelayVisualizer
from .default_process import default_data, default_eval
from .debug import cosine
from .threshold import Threshold


class QuantizeSearch:
    """quantization"""

    def __init__(
        self,
        model_name="opt",
        mod=None,
        params=None,
        dataset=None,
        calibrate_num=None,
        calibrate_batch=100,
        eval_func=None,
        ctx=tvm.cpu(),
        target="llvm",
        root_path=None,
        mean=None,
        scale=None,
        norm=None,
        image_path=None,
        image_size=None,
        channel_last=False,
        rgb="rgb",
        quantize_config=None,
        compare_statistics=False,
        net_in_dtype="uint8",
        opt_level=2,
        verbose=False,
    ):
        self.model_name = model_name
        self.calibrate_num = calibrate_num
        self.calibrate_batch = calibrate_batch
        self.root_path = root_path
        self.image_path = image_path
        self.image_size = image_size
        self.channel_last = channel_last
        self.rgb = rgb
        self.quantize_config = quantize_config
        self.compare_statistics = compare_statistics
        self.net_in_dtype = net_in_dtype
        self.opt_level = opt_level
        self.verbose = verbose

        if self.root_path:
            self.graph_path = os.path.join(self.root_path, self.model_name)
            if not os.path.exists(self.graph_path):
                os.makedirs(self.graph_path)
        else:
            self.graph_path = None

        if self.root_path and self.verbose:
            self.save_path = os.path.join(self.root_path, self.model_name)
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            self.ori_path = os.path.join(self.save_path, "origin_mod.json")
            self.pre_path = os.path.join(self.save_path, "pre_processed_mod.json")
        else:
            self.save_path = None
            self.ori_path = None
            self.pre_path = None

        if mod is not None and params is not None:
            # compatible with nnp300
            if params and not isinstance(mod, relay.Function):
                mod["main"] = self._bind_params(mod["main"], params)
            elif params:
                mod = self._bind_params(mod, params)

            # nnp300_prj
            if "optimize" in tvm.relay.quantize.__dict__:
                norm_en = 1
                if isinstance(mean, (float, int)):
                    mean = (mean,)
                if isinstance(scale, (float, int)):
                    scale = (scale,)
                if (
                    (numpy.all(numpy.array(mean) == 0.0) and numpy.all(numpy.array(scale) == 1.0))
                    or mean is None
                    or scale is None
                ):
                    norm_en = 0

                self.nnp300_pre_processed_mod = tvm.relay.quantize.optimize(
                    mod, params, norm_en, mean, scale
                )
                self.pre_processed_mod = (
                    tvm.relay.quantize.detvm_quantize_optimize.FuseConv2dBiasadd().run(
                        self.nnp300_pre_processed_mod
                    )
                )
                self.pre_processed_mod = tvm.relay.frontend.common.infer_type(
                    self.pre_processed_mod
                )
                # todo support origin float mode
                self.origin_mod = self.pre_processed_mod
            else:
                self.origin_mod = relay.transform.InferType()(mod)
                if self.ori_path:
                    with open(self.ori_path, "w") as f:
                        json.dump(tvm.ir.save_json(self.origin_mod), f)
        else:
            if self.ori_path:
                with open(self.ori_path, "r") as f:
                    self.origin_mod = tvm.ir.load_json(json.load(f))

        self.ctx = ctx
        self.target = target
        self.results = []

        if self.pre_path and os.path.exists(self.pre_path):
            with open(self.pre_path, "r") as f:
                self.pre_processed_mod = tvm.ir.load_json(json.load(f))
        elif "optimize" not in tvm.relay.quantize.__dict__:
            pre_process(self, norm)
            if self.pre_path:
                with open(self.pre_path, "w") as f:
                    json.dump(tvm.ir.save_json(self.pre_processed_mod), f)

        if dataset:
            self.dataset = dataset
        else:
            self.dataset = default_data(self)

        if eval_func:
            self.eval_func = eval_func
            self.use_default_eval = False
        else:
            self.eval_func = default_eval(self)
            self.use_default_eval = True

        get_name(self)
        config_space(self)

    def _bind_params(self, func, params):
        """Bind the params to the expression."""
        name_dict = {}
        for arg in func.params:
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
            bind_dict[arg] = relay.expr.const(v, v.dtype)
        return relay.expr.bind(func, bind_dict)

    def get_default_config(self):
        """get_default_config"""
        config = {}
        for name, v in self.config_space.items():
            default_config = v["default_config"]
            for one in default_config:
                if default_config[one] == {}:
                    continue

                new_arg = {"calibrate_num": self.calibrate_num}
                if isinstance(default_config[one]["threshold"], str):
                    assert default_config[one]["threshold"].startswith(
                        "percentile"
                    ), "if threshold is str, must be percentile_...!!!"
                    percentile_value = float(default_config[one]["threshold"].split("_")[1])
                    default_config[one]["threshold"] = Threshold.Percentile
                    args = default_config[one]["threshold"].args
                    for one_arg in args:
                        new_arg[one_arg["name"]] = percentile_value
                else:
                    args = default_config[one]["threshold"].args
                    for one_arg in args:
                        new_arg[one_arg["name"]] = one_arg["default"]

                default_config[one].update({"threshold_arg": new_arg})
            if "quantized" in v:
                default_config.update({"quantized": v["quantized"]})
            config[name] = default_config

        config["target"] = "nnp400"
        if isinstance(self.quantize_config, dict) and "target" in self.quantize_config:
            config["target"] = self.quantize_config["target"]

        config["adaquant_enable"] = False
        if isinstance(self.quantize_config, dict) and "adaquant_enable" in self.quantize_config:
            config["adaquant_enable"] = self.quantize_config["adaquant_enable"]

        return config

    def quantize(self, config):
        """quantize"""
        if self.save_path:
            quantized_path = os.path.join(self.save_path, "quantized")
            if not os.path.exists(quantized_path):
                os.makedirs(quantized_path)
        else:
            quantized_path = None

        if quantized_path:
            for tmp1 in os.listdir(quantized_path):
                tmp2 = os.path.join(quantized_path, tmp1)
                cond1 = os.path.exists(os.path.join(tmp2, "config"))
                cond2 = os.path.exists(os.path.join(tmp2, "post_processed_mod.json"))
                cond3 = os.path.exists(os.path.join(tmp2, "other"))
                if cond1 and cond2 and cond3:
                    with open(os.path.join(tmp2, "post_processed_mod.json"), "r") as f:
                        if "ir" in tvm.__dict__:
                            tmp_mod = tvm.ir.load_json(json.load(f))
                        else:
                            tmp_mod = tvm.load_json(json.load(f))
                    tmp3 = {
                        "mod": tmp_mod,
                        "config": pandas.read_pickle(os.path.join(tmp2, "config")),
                        "other": pandas.read_pickle(os.path.join(tmp2, "other")),
                    }
                    self.results.append(tmp3)
                    return

        quantize = Quantize(self, config)
        self.quantize_instance = quantize

        tmp = {
            "mod": quantize.post_processed_mod,
            "config": quantize.config,
            "other": {"similarity": quantize.similarity},
        }
        self.results.append(tmp)
        if quantized_path:
            time_path = os.path.join(quantized_path, "%s" % time.time())
            if not os.path.exists(time_path):
                os.makedirs(time_path)
            with open(os.path.join(time_path, "post_processed_mod.json"), "w") as f:
                if "ir" in tvm.__dict__:
                    json.dump(tvm.ir.save_json(tmp["mod"]), f)
                else:
                    json.dump(tvm.save_json(tmp["mod"]), f)
            pandas.to_pickle(tmp["config"], os.path.join(time_path, "config"))
            pandas.to_pickle(tmp["other"], os.path.join(time_path, "other"))

    def evaluate(self, name, config=None):
        """evaluate"""

        def tmp(mod):
            """tmp"""
            if "transform" in relay.__dict__:
                with tvm.transform.PassContext(opt_level=2):
                    graph, lib, params = relay.build(mod, self.target)
                runtime = tvm.contrib.graph_executor.create(graph, lib, self.ctx)
            else:
                with relay.build_config(opt_level=2):
                    graph, lib, params = relay.build(mod, target=self.target)
                runtime = tvm.contrib.graph_runtime.create(graph, lib, self.ctx)

            runtime.set_input(**params)
            return runtime

        if name == "origin":
            runtime = tmp(self.origin_mod)
            self.origin_performance = self.eval_func(runtime)
        elif name == "pre_process":
            runtime = tmp(self.pre_processed_mod)
            self.pre_processed_performance = self.eval_func(runtime)
        elif name == "post_process":
            if self.use_default_eval:
                runtime1 = tmp(self.pre_processed_mod)
                result1 = self.eval_func(runtime1)

                mod = None
                for quantize in self.results:
                    if quantize["config"] == config:
                        mod = quantize["mod"]
                if mod is None:
                    raise ValueError
                runtime2 = tmp(mod)
                result2 = self.eval_func(runtime2)

                tmp3 = []
                for tmp1, tmp2 in zip(result1, result2):
                    print("one image similairy:", cosine(tmp1, tmp2))
                    tmp3.append(cosine(tmp1, tmp2))
                tmp3 = numpy.array(tmp3).mean()
                self.post_processed_performance = tmp3
            else:
                mod = None
                for quantize in self.results:
                    if quantize["config"] == config:
                        mod = quantize["mod"]

                if mod is None:
                    raise ValueError
                runtime = tmp(mod)
                self.post_processed_performance = self.eval_func(runtime)
        else:
            raise ValueError

    def visualize(self, name, config=None):
        """visualize"""
        if name == "origin":
            tmp = RelayVisualizer(self.origin_mod)
            tmp.render("%s_origin.html" % self.model_name)

            with open("%s_origin.txt" % self.model_name, "w") as f:
                f.write(self.origin_mod["main"].__str__())
        elif name == "pre_process":
            tmp = RelayVisualizer(self.pre_processed_mod)
            tmp.render("%s_pre_processed.html" % self.model_name)

            with open("%s_pre_processed.txt" % self.model_name, "w") as f:
                f.write(self.pre_processed_mod["main"].__str__())
        elif name == "post_process":
            mod = None
            for quantize in self.results:
                if quantize["config"] == config:
                    mod = quantize["post_processed_mod"]
            if mod is None:
                raise ValueError

            tmp = RelayVisualizer(mod)
            tmp.render("%s_post_processed.html" % self.model_name)

            with open("%s_post_processed.txt" % self.model_name, "w") as f:
                f.write(mod["main"].__str__())
