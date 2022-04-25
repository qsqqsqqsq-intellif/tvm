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

import os
import tqdm
import numpy
import tensorflow
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tvm
from tvm import relay
from tvm.relay.frontend.tensorflow2 import from_tensorflow
import tvm.relay.quantization

tensorflow.random.set_seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if tvm.runtime.enabled("gpu"):
    ctx = tvm.gpu()
    target = "cuda"
else:
    ctx = tvm.cpu()
    target = "llvm"

batch_size = 1
calibrate_num = 500
num_workers = 8
model_name = "inception_resnet_v2"
performance = {"float": None, "int8": None}
root_path = "/data/zhaojinxi/Documents/quantize_result"
data_path = "/data/zhaojinxi/data/imagenet"

all_op = []


def data_loader():
    a = os.path.join(data_path, "val")
    for b in os.listdir(a):
        bb = os.path.join(a, b)
        for c in os.listdir(bb):
            cc = os.path.join(bb, c)

            # with open(cc, 'rb') as f:
            #     img = pil_image.open(io.BytesIO(f.read()))
            #     img = img.resize([299,299], 0)
            # x = numpy.asarray(img, dtype='float32')
            # if len(x.shape)==2:
            #     x=numpy.expand_dims(x,2)
            #     x=numpy.repeat(x,3,2)
            # x = numpy.expand_dims(x, 0)
            # x=(x-127.5)/127.5

            img = tensorflow.keras.preprocessing.image.load_img(cc, target_size=(299, 299))
            x = tensorflow.keras.preprocessing.image.img_to_array(img)
            x = numpy.expand_dims(x, axis=0)
            # x = tensorflow.keras.applications.inception_resnet_v2.preprocess_input(x)
            yield x, b

            # preds = model.predict(x)
            # if b==tensorflow.keras.applications.inception_resnet_v2.decode_predictions(preds, top=1)[0][0][0]:
            #     correct=correct+1
            # total=total+1
            # print(correct/total)


calibrate_data = []
for i, (image, label) in enumerate(data_loader()):
    if i >= (calibrate_num // batch_size):
        break
    image = image.astype(numpy.uint8)
    calibrate_data.append({"input": image})


def yield_calibrate_data():
    for i in calibrate_data:
        yield i


def evaluate(runtime):
    correct = 0
    total = 0

    t = tqdm.tqdm(data_loader(), total=50000)
    for image, label in t:
        image = image.astype(numpy.uint8)
        image = (image - 127.5) / 127.5
        data = {"input": image}
        runtime.set_input(**data)
        runtime.run()
        output = runtime.get_output(0).asnumpy()
        if (
            label
            == tensorflow.keras.applications.inception_resnet_v2.decode_predictions(output, top=1)[
                0
            ][0][0]
        ):
            correct = correct + 1
        total = total + 1
        acc = correct / total * 100
        t.set_postfix({"accuracy": "{:.4f}".format(acc)})


path = os.path.join(root_path, model_name, "origin_mod.json")
if os.path.exists(path):
    mod = None
    params = None
else:
    x = tensorflow.random.uniform([1, 299, 299, 3], -1, 1)
    model = tensorflow.keras.applications.inception_resnet_v2.InceptionResNetV2(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )
    model_func = tensorflow.function(lambda x: model(x))
    model_func = model_func.get_concrete_function(tensorflow.TensorSpec(x.shape, x.dtype))
    frozen_func = convert_variables_to_constants_v2(model_func)
    graph_def = frozen_func.graph.as_graph_def()
    shape_list = [("input", x.numpy().shape)]
    mod, params = from_tensorflow(graph_def, shape=shape_list)

quantize_search = relay.quantization.QuantizeSearch(
    model_name=model_name,
    mod=mod,
    params=params,
    dataset=yield_calibrate_data,
    calibrate_num=calibrate_num,
    eval_func=evaluate,
    ctx=ctx,
    target=target,
    root_path=root_path,
    norm={
        "input": {
            "mean": [127.5, 127.5, 127.5],
            "std": [127.5, 127.5, 127.5],
            "axis": 3,
        },
    },
    compare_statistics=False,
    verbose=True,
)

config = quantize_search.get_default_config()
quantize_search.quantize(config)
quantize_search.evaluate("post_process", config)
