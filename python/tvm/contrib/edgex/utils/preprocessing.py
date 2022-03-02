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
# pylint: disable=invalid-name, unused-argument
""" preprocessing for models"""

__all__ = ["PreProcessing", "get_preproc_method"]


def get_preproc_method(name):
    method = "".join(n.lower() for n in name.strip())
    return getattr(PreProcessing, method)


class PreProcessing:
    """ preprocessing """

    @classmethod
    def get_config(cls, config):
        """ get config and default values"""
        return config

    @classmethod
    def pass_through(cls, config, inputs):
        """ pass through all the inputs """
        return inputs

    @classmethod
    def mean_scale(cls, config, inputs):
        """ all the inputs using the same mean/scale """
        assert (
            "means" in config and "scales" in config and "layout" in config
        ), "means, scales and layout should be configured with mean_scale preprocessing!"
        means = config["means"]
        scales = config["scales"]
        layout = config["layout"].upper()
        assert layout in (
            "NCHW",
            "NHWC",
            "CHW",
            "HWC",
        ), "supported layout: NCHW or NHWC or CHW or CHW!"
        channel_axis = list(layout).index("C")
        outputs = {}
        for name, inp in inputs.items():
            c = inp.shape[channel_axis]
            assert (
                len(means) == c
            ), "input channel should be equal with means lengths, len(means) = {}".format(
                len(means)
            )
            assert (
                len(scales) == c
            ), "input channel should be equal with scales lengths, len(scales) = {}".format(
                len(scales)
            )

            for channel in range(c):
                inp[:, channel] -= means[channel]
                inp[:, channel] /= scales[channel]

            outputs[name] = inp

        return outputs

    @classmethod
    def mix_processing(cls, config, inputs):
        pass

    @classmethod
    def get_methods(cls):
        return list(
            filter(
                lambda m: not m.startswith("__")
                and not m.endswith("__")
                and callable(getattr(cls, m)),
                dir(cls),
            )
        )
