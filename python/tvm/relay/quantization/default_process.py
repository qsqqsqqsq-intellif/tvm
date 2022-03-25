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
import cv2
import numpy as np
import tvm


def fiter_img(file):
    if file[-4:] in [".jpg", ".bmp", ".png"] or file[-5:] in [".JPEG", ".jpeg"]:
        return True
    return False


def traverse_image(total, path):
    """traverse_image"""
    for one in os.listdir(path):
        new_path = os.path.join(path, one)
        if os.path.isdir(new_path):
            traverse_image(total, new_path)
        elif fiter_img(new_path):
            total.append(new_path)


def process_image(path, net_size, channel_last, rgb, im_info=None):
    """process_image"""

    gray_img = 1 if len(net_size) == 3 and net_size[2] == 1 else 0
    # imread:BGR, HWC
    if gray_img == 0:
        img = cv2.imread(path)
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    im_h_size = img.shape[0]
    im_w_size = img.shape[1]

    wh_tuple = (net_size[1], net_size[0])
    # resize the img to target W/H, the params of resize must be (w, h)
    if img.shape[1::-1] != wh_tuple:
        img = cv2.resize(img, wh_tuple, interpolation=cv2.INTER_LINEAR)
    # convert BGR to RGB if enable
    if rgb == "rgb" and gray_img == 0:
        img = img[:, :, ::-1]
    if len(img.shape) == 2:
        img = np.expand_dims(img, -1)

    if not channel_last:
        img = np.swapaxes([img], 1, 3)
        img = np.swapaxes(img, 2, 3)
    else:
        img = [img]

    if im_info is not None:
        im_h_ratio = float(net_size[0]) / float(im_h_size)
        im_w_ratio = float(net_size[1]) / float(im_w_size)
        if im_info[1] == 3:
            im_scale_factors = np.array([[net_size[0], net_size[1], im_w_ratio]], dtype=np.float32)
        else:
            assert im_info[1] == 4, "when iminfo shape not [1 3], this function only support [1,4]"
            im_scale_factors = np.array(
                [[net_size[0], net_size[1], im_w_ratio, im_h_ratio]], dtype=np.float32
            )
        return [img, im_scale_factors]

    return img


def default_data(cls):
    """default_data"""
    total_path = []
    traverse_image(total_path, cls.image_path)
    random.shuffle(total_path)

    precess_mod = (
        cls.pre_processed_mod["main"]
        if not isinstance(cls.pre_processed_mod, tvm.relay.Function)
        else cls.pre_processed_mod
    )

    shape = [_.value for _ in precess_mod.params[0]._checked_type_.shape]
    model_size = (
        [shape[1], shape[2], shape[3]] if cls.channel_last else [shape[2], shape[3], shape[1]]
    )

    im_info = None
    if len(precess_mod.params) == 2 and precess_mod.params[1].name_hint == "im_info":
        im_info = [_.value for _ in precess_mod.params[1]._checked_type_.shape]

    if not cls.calibrate_num:
        cls.calibrate_num = len(total_path)

    if cls.calibrate_num > len(total_path):
        cls.calibrate_num = len(total_path)

    calibrate_data = []
    for i, path in enumerate(total_path):
        if i >= cls.calibrate_num:
            break
        image = process_image(path, model_size, cls.channel_last, cls.rgb, im_info)
        if im_info is not None:
            calibrate_data.append(
                {
                    precess_mod.params[0].name_hint: image[0],
                    precess_mod.params[1].name_hint: image[1],
                }
            )
        else:
            calibrate_data.append({precess_mod.params[0].name_hint: image})

    # def yield_calibrate_data():
    #     for i in calibrate_data:
    #         yield i

    return lambda: iter(calibrate_data)


def default_eval(cls):
    """default_eval"""

    def evaluate(runtime):
        """evaluate"""
        total_output = []
        idx_count = 1
        for data in cls.dataset():
            if idx_count > cls.calibrate_num:
                break
            idx_count = idx_count + 1

            runtime.set_input(**data)
            # print("num_count_" + str(idx_count), data["data"])
            # idx_count = idx_count + 1
            runtime.run()
            output = runtime.get_output(0)
            output = output.asnumpy()
            total_output.append(output)
        return total_output

    return evaluate
