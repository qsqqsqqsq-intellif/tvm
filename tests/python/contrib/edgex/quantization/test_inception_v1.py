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
import tvm
from tvm import relay


@pytest.mark.skip("mxnet in conflict with pytest")
def test_run():
    import mxnet
    import gluoncv

    mxnet.random.seed(0)

    ctx = tvm.cpu()
    target = "llvm"

    batch_size = 1
    calibrate_num = 1
    num_workers = 1
    model_name = "googlenet"
    performance = {"float": 72.882, "int8": 72.504, "乘右移": 71.962}
    root_path = None

    def prepare_data_loaders(data_path, batch_size):
        transform_test = mxnet.gluon.data.vision.transforms.Compose(
            [
                mxnet.gluon.data.vision.transforms.Resize(256, keep_ratio=True),
                mxnet.gluon.data.vision.transforms.CenterCrop(224),
                mxnet.gluon.data.vision.transforms.ToTensor(),
                # mxnet.gluon.data.vision.transforms.Normalize(
                #     [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                # ),
            ]
        )
        dataset = gluoncv.data.imagenet.classification.ImageNet(
            data_path, train=False
        ).transform_first(transform_test)

        data_loader = mxnet.gluon.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=None,
            num_workers=num_workers,
        )
        return data_loader

    path = os.getenv("QUANT_DIR")
    data_loader = prepare_data_loaders(path + "/data/imagenet", 1)

    calibrate_data = []
    for i, (image, label) in enumerate(data_loader):
        if i >= (calibrate_num // batch_size):
            break
        image = (image.asnumpy() * 255).astype(numpy.uint8)
        calibrate_data.append({"input": image})

    def yield_calibrate_data():
        for i in calibrate_data:
            yield i

    def evaluate(runtime):
        correct = 0
        total = 0

        t = tqdm.tqdm(data_loader)
        for image, label in t:
            image = (image.asnumpy() * 255).astype(numpy.uint8)
            data = {"input": image}
            label = label.asnumpy()
            runtime.set_input(**data)
            runtime.run()
            output = runtime.get_output(0).asnumpy()
            result = output.argmax(axis=1) == label
            correct = correct + result.astype(numpy.float32).sum()
            total = total + label.shape[0]
            acc = correct / total * 100
            t.set_postfix({"accuracy": "{:.4f}".format(acc)})
        return acc

    model = gluoncv.model_zoo.get_model(model_name, pretrained=False, classes=1000)
    model.hybridize()
    model.load_parameters(path + "/model/googlenet-c7c89366.params", cast_dtype=True)
    shape_dict = {"input": (1, 3, 224, 224)}
    mod, params = relay.frontend.from_mxnet(model, shape_dict)

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
                "mean": [0.485 * 255, 0.456 * 255, 0.406 * 255],
                "std": [0.229 * 255, 0.224 * 255, 0.225 * 255],
                "axis": 1,
            },
        },
        compare_statistics=True,
    )

    config = quantize_search.get_default_config()
    quantize_search.quantize(config)
    assert quantize_search.results[0]["other"]["similarity"][0][-1][1] >= 0.99


if __name__ == "__main__":
    test_run()
