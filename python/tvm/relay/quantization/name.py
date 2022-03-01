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
# pylint: disable=unused-argument,inconsistent-return-statements,arguments-differ
"""name"""

import logging
from tvm import relay
from tvm.relay.expr_functor import ExprVisitor

LOGGER = logging.getLogger("quantize")


class GetName(ExprVisitor):
    """get name"""

    def __init__(self, mod):
        super().__init__()
        self.id_count = 0
        self.node_id = {}
        self.id_node = {}
        self.op_num = {}
        self.tuple_num = 0
        LOGGER.info("  ")
        LOGGER.info("--model after pre_process structure:")
        if isinstance(mod, relay.Function):
            self.visit(mod)
        else:
            self.visit(mod["main"])

    def visit_call(self, call):
        for arg in call.args:
            self.visit(arg)

        if isinstance(call.op, relay.Function):
            name = getattr(call.op.attrs, "Composite")
            if not isinstance(name, str):
                name = name.value
        else:
            name = call.op.name
        LOGGER.info("  ")
        LOGGER.info("--id_count %d:", self.id_count)
        LOGGER.info(">> Call, %s, checked_type is:", name)
        LOGGER.info(call.checked_type)
        LOGGER.info(">> inputs:")
        for arg in call.args:
            arg_id = self.node_id[arg]
            LOGGER.info(arg_id)

        if name not in self.op_num:
            self.op_num.update({name: 1})
        else:
            self.op_num[name] = self.op_num[name] + 1
        log_value = ">> " + str(self.op_num[name]) + "th of all " + name
        LOGGER.info(log_value)

        tmp = str(self.id_count) + "_" + name

        self.id_count = self.id_count + 1
        self.node_id[call] = tmp
        self.id_node[tmp] = call

    def visit_var(self, var):
        super().visit_var(var)

        name = type(var).__name__
        tmp = str(self.id_count) + "_" + name
        LOGGER.info("  ")
        LOGGER.info("--id_count %d:", self.id_count)
        LOGGER.info(">>Var, name is %s, checked_type is", var.name_hint)
        LOGGER.info(var.checked_type)
        self.id_count = self.id_count + 1

        self.node_id[var] = tmp
        self.id_node[tmp] = var

    def visit_constant(self, const):
        super().visit_constant(const)

        name = type(const).__name__
        tmp = str(self.id_count) + "_" + name
        self.id_count = self.id_count + 1

        self.node_id[const] = tmp
        self.id_node[tmp] = const

    def visit_tuple(self, tup):
        super().visit_tuple(tup)

        name = type(tup).__name__
        tmp = str(self.id_count) + "_" + name
        LOGGER.info("  ")
        LOGGER.info("--id_count %d:", self.id_count)
        LOGGER.info(">> Tuple, checked_type is:")
        LOGGER.info(tup.checked_type)
        LOGGER.info(">> inputs:")
        for arg in tup.fields:
            arg_id = self.node_id[arg]
            LOGGER.info(arg_id)

        self.tuple_num = self.tuple_num + 1
        log_value = ">> " + str(self.tuple_num) + "th of all tuple"
        LOGGER.info(log_value)

        self.id_count = self.id_count + 1

        self.node_id[tup] = tmp
        self.id_node[tmp] = tup

    def visit_tuple_getitem(self, t):
        super().visit_tuple_getitem(t)

        name = type(t).__name__
        tmp = str(self.id_count) + "_" + name
        self.id_count = self.id_count + 1
        self.node_id[t] = tmp
        self.id_node[tmp] = t

    def visit_function(self, fn):
        super().visit_function(fn)

        name = type(fn).__name__
        tmp = str(self.id_count) + "_" + name
        self.id_count = self.id_count + 1
        self.node_id[fn] = tmp
        self.id_node[tmp] = fn


def get_name(cls):
    tmp = GetName(cls.pre_processed_mod)
    cls.node_id = tmp.node_id
    cls.id_node = tmp.id_node
