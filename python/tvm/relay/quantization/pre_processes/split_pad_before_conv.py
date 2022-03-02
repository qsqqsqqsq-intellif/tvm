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
# pylint: disable=unused-argument,inconsistent-return-statements,bad-continuation
"""split pad before conv if pad refrence count is greater than 1"""
from tvm import relay
from tvm.relay.expr_functor import ExprMutator, ExprVisitor


class GetExprRefCount(ExprVisitor):
    """GetExprRefCount"""

    def __init__(self):
        super().__init__()
        self.ret_ref_cnt = {}

    def visit_call(self, call):
        for arg in call.args:
            if arg in self.ret_ref_cnt:
                self.ret_ref_cnt[arg] = self.ret_ref_cnt[arg] + 1
            else:
                self.ret_ref_cnt[arg] = 1
            self.visit(arg)

    def visit_tuple(self, tup):
        for x in tup.fields:
            if x in self.ret_ref_cnt:
                self.ret_ref_cnt[x] = self.ret_ref_cnt[x] + 1
            else:
                self.ret_ref_cnt[x] = 1
            self.visit(x)

    def visit_tuple_getitem(self, t):
        if t in self.ret_ref_cnt:
            self.ret_ref_cnt[t] = self.ret_ref_cnt[t] + 1
        else:
            self.ret_ref_cnt[t] = 1
        self.visit(t.tuple_value)

    def run(self, func):
        self.visit(func)
        return self.ret_ref_cnt


@relay.transform.function_pass(opt_level=3)
class SplitPadBeforeConv(ExprMutator):
    """SplitPadBeforeConv"""

    def visit_call(self, call):
        visited = super().visit_call(call)
        if (
            visited.op.name in ["nn.conv1d", "nn.conv2d", "nn.conv3d"]
            and isinstance(visited.args[0], relay.Call)
            and visited.args[0].op.name == "nn.pad"
        ):
            pad_attrs = visited.args[0].attrs
            pad_value = visited.args[0].args[1].data.asnumpy()
            if pad_attrs.pad_mode == "constant" and pad_value == 0.0:

                conv_attr = visited.attrs
                new_pad = relay.nn.pad(visited.args[0].args[0], pad_attrs.pad_width, pad_value)
                return relay.Call(visited.op, [new_pad, visited.args[1]], conv_attr)

        return visited

    def transform_function(self, func, mod, ctx):
        self.ref_cnt = GetExprRefCount().run(func)
        return self.visit(func)
