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
import numpy as np
from tvm.ir.module import IRModule
from tvm.relay import ExprMutator
import tvm
from tvm import relay
from tvm.relay.op.strategy.generic import is_depthwise_conv2d
from . import _ffi_api


def PostScheduleArgumentRewrite():
    """Rewrite relay arguments by transformation specified by edgex schedule.

    Returns
    -------
    ret: tvm.transform.Pass
    """
    return _ffi_api.PostScheduleArgumentRewrite()


class ConvertDepthwiseConv2DToConv2D(ExprMutator):
    """Convert Depthwise Conv2d to Conv2d"""

    def __init__(self, params=None):
        super().__init__()
        self._params = dict(params) if params else dict()
        self._is_irmod = False
        self._name_var_map = dict()
        self._var_dict = dict()

    def visit_call(self, call):
        # TODO(someone): need double check if need super().visit_var(call) first,
        # if so, the new var won't be replaced completely, need refine.
        if call.op == tvm.relay.op.get("nn.conv2d"):
            data_layout = call.attrs.data_layout
            kernel_layout = call.attrs.kernel_layout
            data_shape = call.args[0].checked_type.shape
            if self._is_irmod:
                kernel_shape = [int(x) for x in call.args[1].checked_type.shape]
            else:
                weight_old = call.args[1].data.asnumpy()
                kernel_shape = list(weight_old.shape)
            groups = call.attrs.groups
            if data_layout == "NCHW" and is_depthwise_conv2d(
                data_shape, data_layout, kernel_shape, kernel_layout, groups
            ):
                if self._is_irmod:
                    name_hint = call.args[1].name_hint
                    weight_old = self._params.get(name_hint)
                    if weight_old is None:
                        raise ValueError("Can not get %s parameter" % name_hint)
                    weight_old = weight_old.asnumpy()
                kernel_shape[1] = groups
                weight_new = np.zeros(kernel_shape, dtype=weight_old.dtype)
                for i in range(groups):
                    weight_new[i, i] = weight_old[i, 0, :, :]
                # pylint: disable=no-else-return
                if self._is_irmod:
                    self._params[name_hint] = tvm.nd.array(weight_new)
                    conv2d_arg1 = relay.var(
                        name_hint, shape=kernel_shape, dtype=call.args[1].type_annotation.dtype
                    )
                    self._name_var_map[name_hint] = conv2d_arg1
                    # visit args here to create new var for specified name_hint
                    new_fn = self.visit(call.op)
                    new_args = [self.visit(arg) for arg in call.args]
                    new_attr_dict = dict()
                    for attr in call.attrs.keys():
                        attr_value = call.attrs[attr]
                        if isinstance(attr_value, tvm.ir.container.Array):
                            attr_value = tuple(attr_value)
                        new_attr_dict[str(attr)] = attr_value
                    new_attr_dict.update({"groups": 1})
                    attr_type = str(call.attrs).split("(")[0]
                    new_attrs = tvm.ir.make_node(attr_type, **new_attr_dict)
                    return relay.Call(new_fn, new_args, new_attrs, call.span)
                else:
                    conv2d_arg1 = relay.Constant(tvm.nd.array(weight_new))
                    return relay.nn.conv2d(
                        call.args[0],
                        conv2d_arg1,
                        strides=call.attrs.strides,
                        padding=call.attrs.padding,
                        dilation=call.attrs.dilation,
                        groups=1,
                        channels=call.attrs.channels,
                        kernel_size=call.attrs.kernel_size,
                        data_layout=data_layout,
                        kernel_layout=kernel_layout,
                        out_layout=call.attrs.out_layout,
                        out_dtype=call.attrs.out_dtype,
                    )
        return super().visit_call(call)

    def visit_function(self, fn):
        # 1.visit body to get the specified arg's name_hint in the visit_call,
        # 2.then visit the args in visit_call to create new var,
        # 3.bind the new var(free variables) in expr or function arguments.
        new_body = self.visit(fn.body)
        new_params = list()
        binds = dict()
        for param in fn.params:
            if param in self._var_dict:
                new_param = self._var_dict.get(param)
            else:
                new_param = self.visit(param)
            new_params.append(new_param)
            binds[param] = new_param
        new_body = relay.bind(new_body, binds)
        return relay.Function(new_params, new_body, fn.ret_type, fn.type_params, fn.attrs)

    def visit_var(self, var):
        if var.name_hint in self._name_var_map:
            demo_var = self._name_var_map.get(var.name_hint)
            new_var = self._var_dict.get(var)
            if new_var is None:
                new_var = relay.var(var.name_hint, type_annotation=demo_var.type_annotation)
                self._var_dict[var] = new_var
            return new_var
        return super().visit_var(var)

    def run(self, func):
        # pylint: disable=no-else-return
        if isinstance(func, IRModule):
            self._is_irmod = True
            if self._params is None:
                raise ValueError("Need initialize parameters, when the function type is IRModule.")
            func = self.visit(func["main"])
            return IRModule.from_expr(func), self._params
        else:
            return self.visit(func)
