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
# pylint: disable=invalid-name, unused-argument, missing-docstring, unused-import
"""edgex Relay pass transformations."""
import pulp
from tvm.relay import ExprMutator
import tvm
from tvm import relay
from . import _ffi_api
from .convert_depthwise_conv2d import DepthwiseConv2DConvertor


def PostScheduleArgumentRewrite(is_legacy=True):
    """Rewrite relay arguments by transformation specified by edgex schedule.
    Parameters
    ----------
    is_legacy : bool
        TODO(bxq): remove legacy implementation

    Returns
    -------
    ret: tvm.transform.Pass
    """
    return _ffi_api.PostScheduleArgumentRewrite(is_legacy)


def FusionStitch(fuse_opt_level=-1, device_type=16):
    """Fuse operators in an expr to a larger operator according to performance.

    Parameters
    ----------
    fuse_opt_level : int
        The level of fuse optimization. -1 indicates that the level will be
        inferred from pass context.
    device_type : int
        The device type of hardware

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass for operator fusion.
    """
    return _ffi_api.FusionStitch(fuse_opt_level, device_type)


def EdgeXRelayToTIR(
    entry_name="main", renamer=None, post_schedule_rewrite=True, fold_constants=True
):
    """Relay to tir lowering pass

    Parameters
    ----------
    entry_name : str
        Entry relay function name, defautl to "main"

    renamer : function
        External renamer callback

    post_schedule_rewrite : bool
        Whether do schedule and rewrite, just for debug purpose

    fold_constants : bool
        Whether do a final constant folding, just for debug purpose

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass.
    """
    return _ffi_api.EdgeXRelayToTIR(entry_name, renamer, post_schedule_rewrite, fold_constants)


@tvm.register_func("edgex.util.pulp_compute", override=True)
def pulp_compute(pattern_gains, pattern_conflicts):
    """Solve the given Lp problem using pulp

    Parameters
    ----------
    Constraints
    """
    gains = [g.value for g in pattern_gains]
    conflicts = [[i.value for i in row] for row in pattern_conflicts]

    prob = pulp.LpProblem("myPro", pulp.LpMaximize)
    X = pulp.LpVariable.dicts("x", range(len(gains)), lowBound=0, upBound=1, cat=pulp.LpInteger)

    prob += sum(X[i] * gains[i] for i in range(len(gains)))

    for i in range(len(gains)):
        for j in range(len(gains)):
            if j > i and conflicts[i][j] == 1:
                prob += X[i] + X[j] <= 1

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if not status:
        raise RuntimeError("LP failed")

    ret = list()
    for _, x in enumerate(X.values()):
        value = pulp.value(x)
        if value is None:
            ret.append(tvm.tir.IntImm("int32", 1))
        else:
            ret.append(tvm.tir.IntImm("int32", int(value)))

    return ret


class ExtractParamsInOrderPass(ExprMutator):
    """Pass of extracting paramaters in order"""

    def __init__(self):
        super().__init__()
        self.params = {}
        self.func_vars = []
        self.count = 0

    def visit_call(self, call):
        visit = super().visit_call(call)
        return visit

    def visit_tuple_getitem(self, op):
        visit = super().visit_tuple_getitem(op)
        return visit

    def visit_tuple(self, tup):
        visit = super().visit_tuple(tup)
        return visit

    def visit_var(self, var):
        visit = super().visit_var(var)
        self.func_vars.append(visit)
        return visit

    def visit_constant(self, const):
        visit = super().visit_constant(const)
        if len(visit.data.shape) > 0:
            var_name = "arg_{}".format(self.count)
            self.count += 1
            var = relay.var(var_name, shape=visit.data.shape, dtype=visit.data.dtype)
            self.params[var_name] = visit.data
            self.func_vars.append(var)
            return var
        return visit

    def visit_function(self, fn):
        visited = super().visit_function(fn)
        func_vars = list()
        func_vars.extend(self.func_vars)
        return relay.Function(func_vars, visited.body, visited.ret_type, visited.type_params)

    def run(self, expr):
        visited = self.visit(expr)
        return visited, self.params


@tvm._ffi.register_func("tvm.edgex.replace_constants", override=True)
def replace_constants(expr):
    """Replace constants in given expr

    Parameters
    ----------
    expr
    """
    new_expr, _ = ExtractParamsInOrderPass().run(expr)
    return new_expr


def ConvertDepthwiseConv2D(mod, params=None):
    """Convert Depthwise Conv2d"""
    return DepthwiseConv2DConvertor(params=params).run(mod)
