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
# encoding=utf-8
"""CI模型集成测试, 任务集合"""
import argparse
import sys
import os
import json
import onnx  # must precede `import tvm`
import numpy as np
import tvm


def load_module(mod, params):
    if isinstance(mod, str):
        if not os.path.exists(mod):
            raise IOError("Module path %s not exists." % mod)
        with open(mod, "r") as infile:
            mod = tvm.ir.load_json(json.load(infile))
    if isinstance(params, str):
        if not os.path.exists(params):
            raise IOError("Params path %s not exists." % params)
        with open(params, "rb") as infile:
            params = tvm.relay.load_param_dict(infile.read())
    return mod, params


def save_module(output_dir, model_name, mod, params):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, f"{model_name}.json"), "w") as outfile:
        json.dump(tvm.ir.save_json(mod), outfile)
    if params is not None:
        with open(os.path.join(output_dir, f"{model_name}.params"), "wb") as outfile:
            outfile.write(tvm.runtime.save_param_dict(params))


def get_config(model_name, model_config_file):
    with open(model_config_file, "r") as fp:
        config_jsons = json.load(fp)
    config = config_jsons[model_name]
    return config


def convert_frontend_model(args):
    from tvm.driver.tvmc.frontends import load_model

    config = get_config(args.model_name, args.model_config)
    shape_dict, dtype_dict = None, None
    input_names = config["input_names"]
    if config["input_shapes"]:
        shape_dict = {}
        for i, name in enumerate(input_names):
            shape_dict[name] = config["input_shapes"][i]
    if config["input_dtypes"]:
        dtype_dict = {}
        for i, name in enumerate(input_names):
            dtype_dict[name] = config["input_dtypes"][i]

    kwargs = {}
    if config.get("framework") == "onnx":
        kwargs["dtype"] = dtype_dict
    tvmc_model = load_model(
        args.input_file, model_format=config["framework"], shape_dict=shape_dict, **kwargs
    )
    mod, params = tvmc_model.mod, tvmc_model.params
    if args.output_dir is not None:
        save_module(args.output_dir, args.model_name, mod, params)
    return mod, params


def verify_frontend_model(args):
    from tvm.contrib.edgex.utils import verify_model_precision

    config = get_config(args.model_name, args.model_config)
    frontend_flat_config = config.get("frontend", {})
    for k in config:
        if k == "frontend":
            continue
        frontend_flat_config[k] = config[k]
    frontend_flat_config["model_file"] = args.input_file
    mod, params = load_module(args.json, args.params)
    opt_level = config.get("opt_level", 0)
    with tvm.ir.transform.PassContext(opt_level=opt_level):
        print(mod.astext(False))
        lib = tvm.relay.build(mod, params=params, target="llvm")
    executor = tvm.contrib.graph_executor.GraphModule(lib["default"](tvm.cpu()))
    verify_model_precision(frontend_flat_config, executor=executor)


def quantize_model(args):
    import tvm.relay.quantization

    model_name = args.model_name
    config = get_config(model_name, args.model_config)
    mod, params = load_module(args.json, args.params)
    quantized_mod, quantized_params = tvm.relay.quantization.run_quantization(
        model_name,
        mod,
        params,
        mean=config["quantization"]["mean"],
        std=config["quantization"]["std"],
        axis=config["quantization"]["axis"],
        root_path=args.output_dir,
    )
    if args.output_dir is not None:
        save_module(args.output_dir, args.model_name, quantized_mod, quantized_params)
    return quantized_mod, quantized_params


def compile_nnp_model(args):
    from tvm import relay
    from tvm.relay.build_module import bind_params_by_name
    from tvm.contrib.edgex import build_config_nnp
    from tvm.contrib.edgex.relay.transform import FusionStitch, ConvertDepthwiseConv2D
    from tvm.contrib.edgex.testing import (
        TempOpStrategy,
        RelayToTIRAnnotator,
        OnDeviceDetector,
        get_edgex_plan_device_config,
    )
    from tvm.contrib.edgex.relay.op.strategy import (
        fschedule_general_vu,
        SPECIFIED_FSCHEDULE_OPS,
    )

    mod, params = load_module(args.json, args.params)
    mod = FusionStitch()(mod)
    mod, params = ConvertDepthwiseConv2D()(mod, params=params)

    device_annotation_detector = OnDeviceDetector()
    device_annotation_detector.visit_function(mod["main"])

    mod = tvm.IRModule.from_expr(RelayToTIRAnnotator().visit(mod["main"]))
    pass_ctx = build_config_nnp()
    plan_device_cfg = get_edgex_plan_device_config(pass_ctx)
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.PlanDevices(plan_device_cfg)(mod)
    if params is not None:
        func_with_params = bind_params_by_name(mod["main"], params)
        mod = tvm.ir.IRModule.from_expr(func_with_params)

    if device_annotation_detector.has_cpu:
        target = {"edgex": tvm.target.edgex(), "cpu": tvm.target.Target("llvm")}
    else:
        target = tvm.target.edgex()

    override_ops = [x for x in tvm.ir.Op.list_op_names() if x not in SPECIFIED_FSCHEDULE_OPS]
    with TempOpStrategy(override_ops, "edgex", fschedule=fschedule_general_vu):
        with pass_ctx:
            lib = tvm.relay.build(mod, params=params, target=target)
    if args.output_dir is not None:
        lib.export_library(os.path.join(args.output_dir, args.model_name + ".so"))
    return lib


def run_and_check_iss(args):
    from tvm.contrib.edgex.testing import check_edgex_relay_build

    lib = tvm.runtime.load_module(args.input_file)
    mod, params = load_module(args.json, args.params)

    # weight normalization
    params = dict(params.items())
    for k in params:
        if k.find("round_right_shift") >= 0:
            norm = params[k].asnumpy()
            norm = np.minimum(norm, 24)
            norm = np.maximum(norm, 1)
            params[k] = tvm.nd.array(norm)
        elif k.find("multiply") >= 0:
            norm = params[k].asnumpy()
            norm = np.minimum(norm, 127)
            params[k] = tvm.nd.array(norm)
        elif k.find("bias_add") >= 0:
            norm = params[k].asnumpy()
            norm = np.maximum(np.minimum(norm, 2 ** 20), -(2 ** 20))
            params[k] = tvm.nd.array(norm)

    check_edgex_relay_build(mod, params, check_cpu=True, rmse=0.001, edgex_lib=lib)


def parse_args(cmdargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="运行任务名称")
    parser.add_argument("--json", "-j", type=str, help="输入relay json模型路径")
    parser.add_argument("--params", "-p", type=str, help="输入relay参数文件路径")
    parser.add_argument("--input-file", "-i", type=str, help="输入模型文件路径")
    parser.add_argument("--output-dir", "-o", type=str, help="输出文件夹路径")
    parser.add_argument("--model-name", "-n", type=str, required=True, help="模型名称")
    parser.add_argument("--model-config", type=str, help="模型json配置文件")
    return parser.parse_args(cmdargs)


def app_main(cmdargs):
    args = parse_args(cmdargs)
    if args.task == "convert_frontend_model":
        convert_frontend_model(args)
    elif args.task == "verify_frontend_model":
        verify_frontend_model(args)
    elif args.task == "quantize_model":
        quantize_model(args)
    elif args.task == "compile_nnp_model":
        compile_nnp_model(args)
    elif args.task == "run_and_check_iss":
        run_and_check_iss(args)
    else:
        raise ValueError(f"Unknown task name {args.task}")


if __name__ == "__main__":
    app_main(sys.argv[1:])
