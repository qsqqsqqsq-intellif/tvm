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
# pylint: disable=invalid-name, unused-argument, not-callable, import-outside-toplevel
""" verify frontend precision """
import os
import logging
import numpy as np
import cv2
import tvm
from tvm import relay
from tvm.contrib import graph_executor
import tvm.testing

from .preprocessing import get_preproc_method, PreProcessing
from . import framework_infer
from .load_relay_frontend import get_from_framework_frontend, frontend_args

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("frontend")


def get_framework_infer(name):
    mod = framework_infer
    infer_func = "".join([name.lower(), "_model_infer"])
    return getattr(mod, infer_func)


def get_tvm_outputs(
    mod, params, target, dev, baseline_inputs, add_optimize_before_quantize, add_mac_count
):
    """ tvm compilation and get outputs """
    if add_mac_count:
        compute_count = relay.analysis.get_total_mac_number(mod["main"])
        print(
            "---------\033[1;32;43m FLOPS = \033[0m-------: {} GFLOPS".format(
                compute_count * 2 / 1e9
            )
        )

    # todo: consider after realizing optimization
    # if add_optimize_before_quantize:
    #     mod = relay.quantize.optimize(mod, params)

    opt_level = 0  # default
    with tvm.transform.PassContext(opt_level=opt_level):
        lib = relay.build(mod, target, params=params)

    # create module
    module = graph_executor.GraphModule(lib["default"](dev))
    # set input and parameters
    module.set_input(**baseline_inputs)
    # run
    module.run()
    # get output
    num_outputs = module.get_num_outputs()
    tvm_compile_outputs = []
    for i in range(num_outputs):
        compiled_output = module.get_output(i).numpy()
        tvm_compile_outputs.append(compiled_output)
    return tvm_compile_outputs


def assert_shapes_match(actual, desired):
    """ match the shape of output """
    if actual.shape != desired.shape:
        msg = "Output shapes {} and {} don't match"
        raise AssertionError(msg.format(actual.shape, desired.shape))


def compare_results(baseline_outputs, tvm_outputs, rtol=1e-3, atol=1e-3):
    """float point compare with tvm"""
    assert len(baseline_outputs) == len(
        tvm_outputs
    ), "framework output node num = {},tvm output node num = {}".format(
        len(baseline_outputs), len(baseline_outputs)
    )
    for i, tvm_output in enumerate(tvm_outputs):
        assert_shapes_match(baseline_outputs[i], tvm_output)
        tvm.testing.assert_allclose(baseline_outputs[i], tvm_output, rtol, atol)


def model_verify(
    dataset_file,
    model_config,
    target="llvm",
    dev=tvm.cpu(),
    rtol=1e-3,
    atol=1e-3,
    add_optimize_before_quantize=True,
    add_mac_count=True,
):
    """ verify precision between tvm and framework with float point """
    input_names = model_config["input_names"]
    input_shapes = model_config["input_shapes"]
    input_dtypes = model_config["input_dtypes"]

    data_inputs = {}
    if dataset_file:
        assert os.path.exists(dataset_file), "{} is not accessible!".format(dataset_file)
        assert "layout" in model_config, "layout for model should be assign!"
        layout = model_config["layout"].upper()
        assert layout in (
            "NCHW",
            "NHWC",
            "CHW",
            "HWC",
        ), "supported layout: NCHW or NHWC or CHW or CHW!"
        channel_index = list(layout).index("C")
        channel_num = input_shapes[0][channel_index]  # note: default inputs[0]
        if channel_num == 1:
            img = cv2.imread(dataset_file, cv2.IMREAD_GRAYSCALE)
        else:
            assert channel_num == 3
            img = cv2.imread(dataset_file, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height_index = list(layout).index("H")
        width_index = list(layout).index("W")
        new_size = (input_shapes[0][height_index], input_shapes[0][width_index])
        img = cv2.resize(img, new_size)

        img = np.asarray(img)
        if "CHW" in layout:
            img = np.transpose(img, axes=[2, 0, 1])
        if len(layout) == 4:
            img = np.expand_dims(img, axis=0)
        data_inputs[input_names[0]] = img.astype(input_dtypes[0])  # todo: only consider port0
    else:
        for i, input_name in enumerate(input_names):
            data_gen = np.random.uniform(low=0, high=256, size=input_shapes[i])
            data_inputs[input_name] = data_gen.astype(input_dtypes[i])

    # preprocessing
    assert (
        model_config["preproc_method"] in PreProcessing.get_methods()
    ), "configured preproc_method is not in processing list!"
    baseline_inputs = get_preproc_method(model_config["preproc_method"])(model_config, data_inputs)

    # load framework and run
    f_model, baseline_outputs = get_framework_infer(model_config["framework"])(
        model_config["model_file"], baseline_inputs
    )
    logger.info("model_file = %s", model_config["model_file"])

    # tvm processing
    frontend_kwargs = frontend_args(
        input_names, input_shapes, input_dtypes, model_config["framework"]
    )
    mod, params = get_from_framework_frontend(model_config["framework"])(f_model, **frontend_kwargs)
    mod = relay.transform.InferType()(mod)
    logger.info("from %s", model_config["framework"])
    logger.info(mod)

    # get tvm outputs
    tvm_outputs = get_tvm_outputs(
        mod, params, target, dev, baseline_inputs, add_optimize_before_quantize, add_mac_count
    )

    # compare
    compare_results(baseline_outputs, tvm_outputs, rtol, atol)


def check_model_config(config):
    """ check model config. """
    assert "input_names" in config, "input names for model should be assign!"
    assert "input_shapes" in config, "input data shape should be assign!"
    assert "input_dtypes" in config, "input dtype should be assign!"

    assert len(config["input_names"]) == len(config["input_shapes"]), (
        "the number of input_names should be equal to the number of input_shapes,"
        "got {} vs {}".format(len(config["input_names"]), len(config["input_shapes"]))
    )
    assert len(config["input_shapes"]) == len(config["input_dtypes"]), (
        "the number of input_shapes should be equal to the number of input_dtypes,"
        "got {} vs {}".format(len(config["input_shapes"]), len(config["input_dtypes"]))
    )
    assert (
        "framework" in config
    ), "model framework should be assgin: 'onnx', 'pytorch', or 'tflite' etc."
    assert "model_file" in config, "model file should be assign!"


def verify_model_precision(
    model_config,
    test_repeats=1,
    target="llvm",
    dev=tvm.cpu(),
    rtol=1e-3,
    atol=1e-3,
    add_optimize_before_quantize=True,
    add_mac_count=True,
):
    """Assert that the output of a compiled model matches with that of its
        baseline from framework.
    ---------
    Parameters:
    model_config: dict,
                  the configs from model

    test_repeats: int,
                  the times for testing the case, which uses random input

    target: target for run

    rtol: float,
          Relative tolerance

    atol: float,
          Absolute tolerance

    add_optimize_before_quantize: bool,
                                  whether to add optimizations before quantization

    add_mac_count: bool,
                  whether to add mac compute

    Returns:
    ---------
    """
    assert isinstance(test_repeats, int)
    check_model_config(model_config)

    dataset_dir = model_config["dataset_dir"] if "dataset_dir" in model_config else None

    # todo: only support single input port for loading dataset(CV).
    if not dataset_dir is None and dataset_dir != "":
        assert os.path.exists(dataset_dir), "img_dir {} not exists".format(dataset_dir)
        file_lists = os.listdir(dataset_dir)
        for file_name in file_lists:
            dataset_file = dataset_dir + file_name
            model_verify(
                dataset_file,
                model_config,
                target,
                dev,
                rtol,
                atol,
                add_optimize_before_quantize,
                add_mac_count,
            )
    else:
        i = 0
        while i < test_repeats:
            model_verify(
                None,
                model_config,
                target,
                dev,
                rtol,
                atol,
                add_optimize_before_quantize,
                add_mac_count,
            )
            i += 1
    print("PRECISION VERIFY SUCCESS!!!")
