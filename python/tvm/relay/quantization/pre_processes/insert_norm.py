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
"""insert norm"""
import numpy as np
from tvm import relay
from tvm.relay.expr_functor import ExprMutator


@relay.transform.function_pass(opt_level=3)
class InsertNorm(ExprMutator):
    """InsertNorm"""

    def __init__(self, norm):
        super().__init__()
        self.norm = norm

    def visit_var(self, var):
        new_var = super().visit_var(var)
        if self.norm is not None and new_var.name_hint in self.norm:
            shape = var.checked_type.shape
            name = new_var.name_hint
            axis = self.norm[name]["axis"]
            channels = shape[axis].value
            norm_info = self.norm[name]

            if "mean" not in norm_info:
                mean = [0 for _ in range(channels)]
            else:
                mean = norm_info["mean"]
                if mean is None:
                    mean = 0
                if isinstance(mean, (int, float)):
                    mean = [mean for _ in range(channels)]

            if "std" not in norm_info:
                std = [1 for _ in range(channels)]
            else:
                std = norm_info["std"]
                if std is None:
                    std = 1
                if isinstance(std, (int, float)):
                    std = [std for _ in range(channels)]
            std = [x ** 2 for x in std]

            mean_eq_0 = [x == 0 for x in mean]
            std_eq_1 = [x == 1 for x in std]
            if all(mean_eq_0) and all(std_eq_1):
                return new_var

            std = np.array(std, "float32")
            mean = np.array(mean, "float32")

            gamma = relay.const(np.ones((channels,), "float32"))
            beta = relay.const(np.zeros((channels,), "float32"))
            mean = relay.const(mean)
            std = relay.const(std)

            new_call = relay.nn.batch_norm(new_var, gamma, beta, mean, std, axis=axis)
            return new_call[0]

        return new_var

    def visit_function(self, fn):
        for x in fn.params:
            self.visit(x)
        new_body = self.visit(fn.body)
        return relay.Function(fn.params, new_body, fn.ret_type, fn.type_params, fn.attrs)

    def transform_function(self, func, mod, ctx):
        return self.visit(func)
