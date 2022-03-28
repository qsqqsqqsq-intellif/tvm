import os
import sys
import logging
import traceback
import onnx
import tvm.relay as relay
from tvm.relay.quantization import run_quantization

source_path = "/data/share/demodels-lfs/onnx"
target_path = "/home/zhaojinxi/Documents/onnx_result"

models = {}
for name in os.listdir(source_path):
    model_path = os.path.join(source_path, name)
    for file in os.listdir(model_path):
        if os.path.splitext(file)[1] == ".onnx":
            models[name] = {"file": file}

for name, v in models.items():
    save_path = os.path.join(target_path, name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    log_path = os.path.join(save_path, "log.txt")

    # logging.basicConfig(filename=log_path, filemode='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    log = open(log_path, "w")

    sys.stdout = log

    model_path = os.path.join(source_path, name)
    file_path = os.path.join(model_path, v["file"])
    model = onnx.load(file_path)
    model_input = model.graph.input

    shape_list = []
    for i, input in enumerate(model_input):
        dim = input.type.tensor_type.shape.dim
        dim[0].dim_value = 1
        shape = [j.dim_value for j in dim]
        # logging.debug(shape)
        log.write("input shape:\n" + str(shape) + "\n\n")

        shape_list.append(("input%s" % i, shape))

        for k, a in enumerate(dim):
            if a.dim_value == 3:
                models[name].update({"axis": k})

    try:
        mod, params = relay.frontend.from_onnx(model)
        run_quantization(name, mod, params, axis=models[name]["axis"], root_path=target_path)
    except Exception:
        # logging.debug(traceback.format_exc())
        log.write(traceback.format_exc())

    # logging.shutdown()
    log.close()
