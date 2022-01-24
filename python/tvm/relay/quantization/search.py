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
# pylint: disable=unused-argument,inconsistent-return-statements,bare-except
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


class QuantizeSearch:
    """quantization"""

    def __init__(
        self,
        model_name=None,
        mod=None,
        params=None,
        dataset=None,
        calibrate_num=None,
        eval_func=None,
        ctx=tvm.cpu(),
        target="llvm",
        root_path=None,
        mean=None,
        scale=None,
        image_path=None,
        image_size=None,
        channel_last=True,
        rgb="rgb",
        quantize_config=None,
        compare_statistics=False,
        net_in_dtype="uint8",
        opt_level=3,
    ):
        self.model_name = model_name
        self.calibrate_num = calibrate_num
        self.root_path = root_path
        self.image_path = image_path
        self.image_size = image_size
        self.channel_last = channel_last
        self.rgb = rgb
        self.quantize_config = quantize_config
        self.compare_statistics = compare_statistics
        self.net_in_dtype = net_in_dtype
        self.opt_level = opt_level

        if self.root_path is not None:
            save_path = os.path.join(self.root_path, self.model_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            self.ori_path = os.path.join(save_path, "origin_mod.json")
            self.pre_path = os.path.join(save_path, "pre_processed_mod.json")
        else:
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
            else:
                self.origin_mod = relay.transform.InferType()(mod)
                if self.ori_path is not None:
                    with open(self.ori_path, "w") as f:
                        json.dump(tvm.ir.save_json(self.origin_mod), f)
        else:
            with open(self.ori_path, "r") as f:
                self.origin_mod = tvm.ir.load_json(json.load(f))

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

        self.ctx = ctx
        self.target = target
        self.results = []

        if self.pre_path is not None and os.path.exists(self.pre_path):
            with open(self.pre_path, "r") as f:
                self.pre_processed_mod = tvm.ir.load_json(json.load(f))
        elif "optimize" not in tvm.relay.quantize.__dict__:
            pre_process(self, mean, scale)
            if self.pre_path is not None:
                with open(self.pre_path, "w") as f:
                    json.dump(tvm.ir.save_json(self.pre_processed_mod), f)

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
                args = default_config[one]["threshold"].args
                new_arg = {"calibrate_num": self.calibrate_num}
                for one_arg in args:
                    new_arg[one_arg["name"]] = one_arg["default"]
                default_config[one].update({"threshold_arg": new_arg})
            config[name] = default_config
        if isinstance(self.quantize_config, dict) and "skip_conv_layers" in self.quantize_config:
            config["skip_conv_layers"] = self.quantize_config["skip_conv_layers"]

        config["target"] = "nnp400"
        if isinstance(self.quantize_config, dict) and "target" in self.quantize_config:
            config["target"] = self.quantize_config["target"]
        return config

    def quantize(self, config):
        """quantize"""
        if self.root_path is not None:
            save_path = os.path.join(self.root_path, self.model_name, "quantized")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        else:
            save_path = None

        if save_path is not None:
            for tmp1 in os.listdir(save_path):
                tmp2 = os.path.join(save_path, tmp1)
                cond1 = os.path.exists(os.path.join(tmp2, "config"))
                cond2 = os.path.exists(os.path.join(tmp2, "post_processed_mod.json"))
                cond3 = os.path.exists(os.path.join(tmp2, "other"))
                if cond1 and cond2 and cond3:
                    with open(os.path.join(tmp2, "post_processed_mod.json"), "r") as f:
                        tmp_mod = tvm.ir.load_json(json.load(f))
                    tmp3 = {
                        "mod": tmp_mod,
                        "config": pandas.read_pickle(os.path.join(tmp2, "config")),
                        "other": pandas.read_pickle(os.path.join(tmp2, "other")),
                    }
                    self.results.append(tmp3)
                    return

        quantize = Quantize(self, config)

        self.quantized_func = quantize.post_processed_mod
        # with open("/home/yhh/Desktop/tmp/nnp400/mobilenet_edgeput.json", "w") as f:
        #     json.dump(tvm.ir.save_json(quantize.post_processed_mod), f)

        tmp = {
            "mod": quantize.post_processed_mod,
            "config": quantize.config,
            "other": {"similarity": quantize.similarity},
        }
        self.results.append(tmp)
        if save_path is not None:
            qtz_path = os.path.join(save_path, "%s" % time.time())
            if not os.path.exists(qtz_path):
                os.makedirs(qtz_path)
            with open(os.path.join(qtz_path, "post_processed_mod.json"), "w") as f:
                json.dump(tvm.ir.save_json(tmp["mod"]), f)
            pandas.to_pickle(tmp["config"], os.path.join(qtz_path, "config"))
            pandas.to_pickle(tmp["other"], os.path.join(qtz_path, "other"))

    def evaluate(self, name, config=None):
        """evaluate"""

        def tmp(mod):
            with tvm.transform.PassContext(opt_level=3):
                graph, lib, params = relay.build(mod, self.target)
            runtime = tvm.contrib.graph_executor.create(graph, lib, self.ctx)
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
                runtime1 = tmp(self.origin_mod)
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
                    tmp3.append(cosine(tmp1, tmp2))
                tmp3 = numpy.array(tmp3).mean()
                self.post_processed_performance = tmp3
            else:
                mod = None
                for quantize in self.results:
                    if quantize["config"] == config:
                        mod = quantize["mod"]

                # with open("/home/yhh/Desktop/tmp/yolov5s_ult/p9999_prof300.json", "r") as f:
                #     xx = json.load(f)
                #     mod = tvm.ir.load_json(xx)

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
