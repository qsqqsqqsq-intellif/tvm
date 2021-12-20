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
import pytest
import tqdm
import numpy
import torch
import torchvision
import tvm
from tvm import relay
import tvm.relay.quantization


@pytest.mark.edgex_slow
def test_run():
    torch.manual_seed(0)

    ctx = tvm.cpu()
    target = "llvm"

    batch_size = 1
    calibrate_num = 1
    num_workers = 1
    model_name = "densenet121"
    performance = {"float": 74.434, "int8": None}
    root_path = None

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

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, sampler=None
        )
        return data_loader

    path = os.getenv("QUANT_DIR")
    data_loader = prepare_data_loaders(path + "/data/imagenet", 1)

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

    model = torch.jit.load(path + "/model/densenet121")
    shape_list = [("input", [1, 3, 224, 224])]
    mod, params = relay.frontend.from_pytorch(model, shape_list)

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
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        scale=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        opt_level=2,
        compare_statistics=True,
    )

    config = quantize_search.get_default_config()
    quantize_search.quantize(config)
    assert quantize_search.results[0]["other"]["similarity"][0][-1][1] >= 0.99


if __name__ == "__main__":
    test_run()
