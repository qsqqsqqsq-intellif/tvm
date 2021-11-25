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
"""DMA pattern detection utilities"""
from . import _ffi_api


def detect_fuse_split_seq(itervars, bindings, dom_map, respect_input_dom=True, verbose=False):
    """Detect fuse/split operation sequence from iter bindings.

    Parameters
    ----------
    itervars : List[tvm.tir.Var]
        The input iteration variables
    bindings : List[tvm.tir.PrimExpr]
        The output bindings
    dom_map : dict
        Iteration range dict
    respect_input_dom : bool
        Whether use input dom extent if conflict with detected iteration extent.

    Returns
    -------
    arr : (List[(str, int, int, int)], List[tvm.tir.PrimExpr], List[int])
        The first item is the operation sequence (op_type, fuse_id, outer_id, inner_id)
        where op_type should be either "fuse" or "split". The id [0, len(bindings))
        represent iterations for output bindings.
        The second item is expr binding for all iters indexed by id.
        The third item is iter extent inferred for all iters indexed by id.
    """
    c_ops, bindings, c_extents = _ffi_api.DetectFuseSplitSeq(
        itervars, bindings, dom_map, respect_input_dom, verbose
    )
    ops = []
    for typ, fuse_id, outer_id, inner_id in c_ops:
        ops.append((typ, fuse_id.value, outer_id.value, inner_id.value))
    extents = [x.value for x in c_extents]
    return ops, bindings, extents


def detect_reshape_transpose_seq(
    itervars, bindings, dom_map, respect_input_dom=True, verbose=False
):
    """Detect reshape/transpose operation sequence from iter bindings.

    Parameters
    ----------
    itervars : List[tvm.tir.Var]
        The input iteration variables
    bindings : List[tvm.tir.PrimExpr]
        The output bindings
    dom_map : dict
        Iteration range dict
    respect_input_dom : bool
        Whether use input dom extent if conflict with detected iteration extent.

    Returns
    -------
    tuple : List[(str, List[int])]
        The operation sequence (op_type, values)
        where op_type should be either "reshape" or "transpose".
    """
    c_results = _ffi_api.DetectReshapeTransposeSeq(
        itervars, bindings, dom_map, respect_input_dom, verbose
    )
    results = []
    for typ, vec in c_results:
        results.append((typ, [x.value for x in vec]))
    return results
