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
# pylint: disable=unused-argument,inconsistent-return-statements
"""default data"""

import os
import random
from PIL import Image
import tqdm
import numpy


def traverse_image(total, path):
    """traverse_image"""
    for one in os.listdir(path):
        new_path = os.path.join(path, one)
        if os.path.isdir(new_path):
            traverse_image(total, new_path)
        else:
            total.append(new_path)


def process_image(path, image_size, channel_last, rgb):
    """process_image"""
    img = Image.open(path)
    img = img.resize(image_size, Image.BILINEAR)
    image = numpy.array(img)
    if len(image.shape) == 2:
        image = numpy.stack([image, image, image], 2)
    if rgb == "rgb":
        pass
    elif rgb == "bgr":
        rc1 = image[:, :, 0]
        gc1 = image[:, :, 0]
        bc1 = image[:, :, 0]
        image = numpy.stack([bc1, gc1, rc1], 2)
    else:
        raise NotImplementedError

    if channel_last:
        pass
    else:
        image = numpy.transpose(image, [2, 0, 1])
        image = numpy.expand_dims(image, 0)
    return image


def default_data(cls):
    """default_data"""
    total_path = []
    traverse_image(total_path, cls.image_path)
    random.shuffle(total_path)

    if not cls.calibrate_num:
        cls.calibrate_num = len(total_path)

    calibrate_data = []
    for i, path in enumerate(total_path):
        if i >= cls.calibrate_num:
            break
        image = process_image(path, cls.image_size, cls.channel_last, cls.rgb)
        calibrate_data.append({"input": image})

    def yield_calibrate_data():
        for i in calibrate_data:
            yield i

    return yield_calibrate_data


def default_eval(cls):
    """default_eval"""
    total_path = []
    traverse_image(total_path, cls.image_path)

    def evaluate(runtime):
        """evaluate"""
        total_output = []
        for path in tqdm.tqdm(total_path):
            image = process_image(path, cls.image_size, cls.channel_last, cls.rgb)
            data = {"input": image}
            runtime.set_input(**data)
            runtime.run()
            output = runtime.get_output(0)
            output = output.asnumpy()
            total_output.append(output)
        return total_output

    return evaluate
