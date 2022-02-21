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
import torch
import torchvision
import tvm
from tvm import relay
import tvm.relay.quantization


from tvm.relay.expr_functor import ExprMutator

torch.manual_seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if tvm.runtime.enabled("gpu"):
    ctx = tvm.cuda()
    target = "cuda"
else:
    ctx = tvm.cpu()
    target = "llvm"

ctx = tvm.cpu()
target = "llvm"

batch_size = 1
calibrate_num = 50
num_workers = 8
model_name = "mobilenet_v3_small"
performance = {"float": 67.668, "int8": 53.146}
root_path = os.path.join(os.path.expanduser("~"), "Documents/quantize_result")
data_path = "/data/zhaojinxi/data/imagenet"
# data_path = "/home/yhh/Desktop/dedatasets-lfs"

all_op = [
    "conv2d_bias_add",
    "hard_swish",
    "nn.relu",
    "nn.adaptive_avg_pool2d",
    "hard_sigmoid",
    "squeeze",
    "expand_dims",
    "multiply",
    "add",
    "reshape",
    "dense_bias_add",
]


def prepare_data_loaders(data_path, batch_size):
    dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_path, "val"),
        torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(
                #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                # ),
            ]
        ),
    )

    sampler = torch.utils.data.RandomSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler
    )
    return data_loader


data_loader = prepare_data_loaders(data_path, batch_size)

calibrate_data = []
for i, (image, label) in enumerate(data_loader):
    if i >= (calibrate_num // batch_size):
        break
    image = (image.numpy() * 255).astype(numpy.uint8)
    calibrate_data.append({"input": image})


def yield_calibrate_data():
    for i in calibrate_data:
        yield i


def evaluate(runtime):
    correct = 0
    total = 0

    t = tqdm.tqdm(data_loader)
    for image, label in t:
        image = (image.numpy() * 255).astype(numpy.uint8)
        data = {"input": image}
        label = label.numpy()
        runtime.set_input(**data)
        runtime.run()
        output = runtime.get_output(0).asnumpy()
        result = output.argmax(axis=1) == label
        correct = correct + result.astype(numpy.float32).sum()
        total = total + label.shape[0]
        acc = correct / total * 100
        t.set_postfix({"accuracy": "{:.4f}".format(acc)})
    return acc


path = os.path.join(root_path, model_name, "origin_mod.json")
if os.path.exists(path):
    mod = None
    params = None
else:
    model = torch.hub.load("pytorch/vision:v0.10.0", model_name, pretrained=True)
    x = torch.randn([1, 3, 224, 224])
    scripted_model = torch.jit.trace(model.eval(), x)
    shape_list = [("input", x.numpy().shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

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
            "mean": [123.675, 116.28, 103.53],
            "std": [58.395, 57.12, 57.375],
            "axis": 1,
        },
    },
    compare_statistics=False,
)

config = quantize_search.get_default_config()
quantize_search.quantize(config)
# quantize_search.visualize("post_process", config)

# from tvm.relay.quantization.post_process import extract_module

# extract_module(quantize_search.results[0]["mod"], "/home/wangjiuyang/", "mobilenet_v3", next(yield_calibrate_data()))

mod = quantize_search.results[-1]["mod"]

print(mod["main"])


class Xxxxx(ExprMutator):
    def __init__(self):
        super().__init__()
        self.out = []
        self.cnt = 0

    def visit_call(self, call):
        visited = super().visit_call(call)
        if isinstance(visited.op, relay.Function):
            name = getattr(visited.op.attrs, "Composite")
        else:
            name = visited.op.name

        # if self.cnt > 1:
        #     return visited

        if name == "nn.relu":
            self.out.append(visited)
            self.cnt = self.cnt + 1
        # else:
        #     if len(self.out) > 0:
        #         self.out.append(visited)

        return visited

    def visit_function(self, fn):
        super().visit_function(fn)

        params = fn.params

        return relay.Function(params, relay.Tuple(self.out))

    def run(self, mod):
        mod["main"] = self.visit(mod["main"])
        return relay.transform.InferType()(mod)


# mod = Xxxxx().run(mod)
# print(mod)

# with tvm.transform.PassContext(opt_level=3):
#     graph, lib, params = relay.build(mod, "llvm")
#     runtime = tvm.contrib.graph_executor.create(graph, lib, tvm.cpu())
#     runtime.set_input(**params)

# dataset = yield_calibrate_data()

# while True:
#     try:
#         batch = next(dataset)

#         runtime.set_input("input", batch["input"])
#         runtime.run()
#         outputs = []
#         output = runtime.get_output(1).asnumpy()
#         max_v = numpy.max(output)
#         min_v = numpy.min(output)
#         print("max:",max_v,"min:", min_v)

#         output = runtime.get_output(2).asnumpy()
#         max_v = numpy.max(output)
#         min_v = numpy.min(output)
#         print("max:",max_v,"min:", min_v)

#         output = runtime.get_output(3).asnumpy()
#         max_v = numpy.max(output)
#         min_v = numpy.min(output)
#         print("max:",max_v,"min:", min_v)

#         output = runtime.get_output(4).asnumpy()
#         max_v = numpy.max(output)
#         min_v = numpy.min(output)
#         print("max:",max_v,"min:", min_v)

#         output = runtime.get_output(5).asnumpy()
#         max_v = numpy.max(output)
#         min_v = numpy.min(output)
#         print("max:",max_v,"min:", min_v)

#         output = runtime.get_output(6).asnumpy()
#         max_v = numpy.max(output)
#         min_v = numpy.min(output)
#         print("max:",max_v,"min:", min_v)

#     except StopIteration:
#         break

# exit()


quantize_search.evaluate("post_process", config)
