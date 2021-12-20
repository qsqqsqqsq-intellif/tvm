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
"""Automatic quantization toolkit."""

import numpy
from tvm._ffi import runtime_ctypes


class DataType:
    """DataType"""

    Int4 = "int4"
    Int8 = "int8"
    UInt8 = "uint8"
    Int16 = "int16"
    Int32 = "int32"
    Float16 = "float16"
    Float32 = "float32"


def _get_dtype_info(dtype, qmin=None, qmax=None):
    """get_dtype_info"""
    dtype = runtime_ctypes.DataType(dtype)
    dtype_str = dtype.CODE2STR[dtype.type_code]
    dtype_bit = dtype.bits
    if qmin is None and qmax is None:
        if dtype_str == "int":
            int_range = 2 ** (dtype_bit - 1)
            qmin = -int_range
            qmax = int_range - 1
        elif dtype_str == "uint":
            int_range = 2 ** dtype_bit
            qmin = 0
            qmax = int_range - 1
        else:
            raise NotImplementedError
    assert qmin is not None and isinstance(qmin, int)
    assert qmax is not None and isinstance(qmax, int)
    return {"qmin": qmin, "qmax": qmax}


def symmetry(config):
    """symmetry method"""
    if "min" not in config or "max" not in config:
        y = config["threshold"].min_max
        config.update(y)
    max_val = numpy.maximum(numpy.abs(config["min"]), numpy.abs(config["max"]))
    scale = numpy.array(max_val / config["qmax"]).astype(numpy.float32)
    zero_point = numpy.zeros_like(scale).astype(numpy.int32)
    result = {"scale": scale, "zero_point": zero_point, "axis": config["threshold"].axis}
    return result


def asymmetry(config):
    """asymmetry method"""
    if "min" not in config or "max" not in config:
        y = config["threshold"].min_max
        config.update(y)
    min_val = numpy.minimum(config["min"], numpy.zeros_like(config["min"]))
    max_val = numpy.maximum(config["max"], numpy.zeros_like(config["max"]))
    # min_val = config["min"]
    # max_val = config["max"]
    scale = numpy.array((max_val - min_val) / float(config["qmax"] - config["qmin"])).astype(
        numpy.float32
    )
    # zero_point = result['qmin'] - numpy.round(min_val / scale)
    zero_point = config["qmax"] - numpy.round(max_val / scale)  # 添加对比重建误差
    zero_point = numpy.clip(zero_point, config["qmin"], config["qmax"]).astype(numpy.int32)
    result = {"scale": scale, "zero_point": zero_point}
    return result


class Method:
    """symmetry or asymmetry"""

    Symmetry = symmetry
    Asymmetry = asymmetry


class Granularity:
    """Granularity"""

    Tensor = "tensor"
    Channel = "channel"
