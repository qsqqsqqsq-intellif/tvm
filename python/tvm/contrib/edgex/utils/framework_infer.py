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
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel
# pylint: disable=c-extension-no-member, broad-except
""" framework inference """
import os
import onnx
import tflite
import onnxruntime


def onnx_model_infer(model_file, baseline_inputs):
    """
    infer a onnx model with given inputs
    """
    assert os.path.exists(model_file), "{} is not accessible!".format(model_file)
    onnx_model = onnx.load(model_file)

    # for debug:
    # import onnx_graphsurgeon as gs
    # graph = gs.import_onnx(onnx_model)
    # tensors = graph.tensors()
    # graph.inputs = [tensors["input:0"].to_variable(dtype=np.float32, shape=(1, 3, 224, 224))]
    # graph.outputs = [tensors["generator/G_MODEL/A/LeakyRelu:0"].to_variable(dtype=np.float32)]
    # graph.cleanup()
    # onnx.save(gs.export_onnx(graph), "/data/subgraph.onnx")
    # model_file = "/data/subgraph.onnx"
    # onnx_model = onnx.load(model_file)

    try:
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        import warnings

        # the checker is a bit violent about errors, so simply print warnings here
        warnings.warn(str(e))
    ort_session = onnxruntime.InferenceSession(model_file)
    ort_outs = ort_session.run(None, baseline_inputs)  # list

    # used for debug shape
    # inferred_onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    # onnx.save(inferred_onnx_model, "/path/check.onnx")

    return onnx_model, ort_outs


def tflite_model_infer(model_file, baseline_inputs):
    """
    infer a onnx model with given inputs
    """
    assert os.path.exists(model_file), "{} is not accessible!".format(model_file)
    from tensorflow.lite.python import interpreter as interpreter_wrapper

    interpreter = interpreter_wrapper.Interpreter(model_path=model_file)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    output_nums = len(output_details)

    interpreter.allocate_tensors()
    # set input
    for i, baseline_input in enumerate(baseline_inputs.values()):
        interpreter.set_tensor(input_details[i]["index"], baseline_input)

    # Run
    interpreter.invoke()

    tf_outs = []
    for i in range(output_nums):
        # get output
        baseline_output = interpreter.get_tensor(output_details[i]["index"])
        tf_outs.append(baseline_output)

    tflite_model_buf = open(model_file, "rb").read()
    # get TFLite model from buffer
    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)

    return tflite_model, tf_outs
