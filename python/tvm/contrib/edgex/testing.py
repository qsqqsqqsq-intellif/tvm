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
"""Edgex testing utilities"""
import inspect
import numpy as np
import tvm
from tvm.contrib.edgex import build_config_nnp
from tvm.ir.module import IRModule


def check_edgex_tir_build(
    name,
    prim_func,
    numpy_func=None,
    check_edgex=True,
    check_cpu=True,
    need_lower=True,
    data_range=None,
    input_data=None,
    rmse=None,
):
    """build and check edgex tir module
    Parameters
    ----------
    name : str
        The kernel name to use.

    primfunc : tir.PrimFunc
        The tir function to build and test.

    numpy_func : function
        The compatible computation logic in numpy to get the expect output.

    check_edgex : bool
        Whether run on edgex

    check_cpu : bool
        Whether run on cpu

    need_lower : bool
        Whether lower the input primfunc

    data_range :
        Testdata sample range.

    input_data : list[numpy.ndarray]
        Input data bindings, if not given, will sample data randomly.

    rmse : float
        If specified, check root-mean-square deviation between results and expects
        instead of close assertion.
    """
    arrs = []
    for idx, param in enumerate(prim_func.params):
        if input_data is not None and idx < len(input_data) and input_data[idx] is not None:
            arrs.append(input_data[idx])
            continue
        buffer = prim_func.buffer_map[param]
        shape = [int(x) for x in buffer.shape]
        dtype = buffer.dtype
        if data_range is None:
            data_range = (-64, 63)
        elif data_range == "full":
            if dtype.startswith("i") or dtype.startswith("u"):
                tinfo = np.iinfo(dtype)
            else:
                tinfo = np.finfo(dtype)
            data_range = (tinfo.min, tinfo.max)
        elif isinstance(data_range, int):
            data_range = (data_range, data_range + 1)
        elif isinstance(data_range, float):
            data_range = (data_range, data_range)
        if dtype.startswith("i") or dtype.startswith("u"):
            arrs.append(np.random.randint(data_range[0], data_range[1], size=shape).astype(dtype))
        else:
            arrs.append(np.random.uniform(data_range[0], data_range[1], size=shape).astype(dtype))

    expects = None
    if numpy_func is not None:
        n_args = len(inspect.getfullargspec(numpy_func).args)
        expects = numpy_func(*arrs[:n_args])
        if not isinstance(expects, (list, tuple)):
            expects = [expects]
        assert (
            len(expects) == len(arrs) - n_args
        ), """numpy func take %d input and return %d outputs,
           but edgex func take %d tensors total""" % (
            n_args,
            len(expects),
            len(arrs),
        )
        expects = arrs[:n_args] + expects

    if check_cpu:
        build_input = prim_func if need_lower else {"llvm": IRModule({name: prim_func})}
        ctx = tvm.cpu()
        cpu_mod = tvm.build(build_input, [], target="llvm", name=name)
        cpu_tensors = [tvm.nd.array(x, ctx) for x in arrs]
        cpu_mod(*cpu_tensors)
        cpu_results = [x.asnumpy() for x in cpu_tensors]
        if expects is None:
            expects = cpu_results
        else:
            for _, (expect, res) in enumerate(zip(expects, cpu_results)):
                check_numpy_result(res, expect, rmse=rmse)

    if check_edgex:
        build_input = prim_func if need_lower else {"edgex": IRModule({name: prim_func})}
        ctx = tvm.edgex()
        with build_config_nnp():
            edgex_mod = tvm.build(build_input, [], target="edgex", name=name)
        edgex_tensors = [tvm.nd.array(x, ctx) for x in arrs]
        edgex_mod(*edgex_tensors)
        edgex_results = [x.asnumpy() for x in edgex_tensors]
        if expects is not None:
            for _, (expect, res) in enumerate(zip(expects, edgex_results)):
                check_numpy_result(res, expect, rmse=rmse)
