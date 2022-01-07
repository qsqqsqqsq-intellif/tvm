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
"""op"""

from ..threshold import Threshold
from ..method_dtype import Granularity, Method, DataType, _get_qmin_qmax

__all__ = ("BiasAdd",)

_valid_config = {
    "input0": {
        "threshold": (
            Threshold.MinMax,
            Threshold.Percentile,
            Threshold.MovingAverageMinMax,
            Threshold.L2Norm,
            Threshold.RelativeEntropy,
        ),
        "granularity": (Granularity.Tensor, Granularity.Channel),
        "method": (Method.Symmetry, Method.Asymmetry),
        "dtype": (DataType.Int32,),
    },
    "input1": {
        "threshold": (Threshold.MinMax, Threshold.Percentile),
        "granularity": (Granularity.Tensor, Granularity.Channel),
        "method": (Method.Symmetry, Method.Asymmetry),
        "dtype": (DataType.Int32,),
    },
    "output0": {
        "dtype": (DataType.Int32,),
    },
}

_default_config = {
    "input0": {
        "threshold": Threshold.L2Norm,
        "granularity": Granularity.Channel,
        "method": Method.Symmetry,
        "dtype": DataType.Int32,
    },
    "input1": {
        "threshold": Threshold.MinMax,
        "granularity": Granularity.Channel,
        "method": Method.Symmetry,
        "dtype": DataType.Int32,
    },
    "output0": {
        "dtype": DataType.Int32,
    },
}


class BiasAdd:
    """BiasAdd"""

    name = "nn.bias_add"

    def __init__(self, config, node):
        self._valid_config = _valid_config
        self._default_config = _default_config

    @classmethod
    def get_config(cls, config, call):
        return {"valid_config": _valid_config, "default_config": _default_config}
