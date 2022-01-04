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
"""extract_module"""
import pickle
from collections import deque
import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprMutator


def is_call(the_expr, names):
    """is_call"""
    op_names = names if isinstance(names, list) else [names]
    return isinstance(the_expr, tvm.relay.expr.Call) and the_expr.op.name in op_names


def replace_arg_of_expr(expr, index, new_arg):
    """replace_arg_of_expr"""
    if isinstance(expr, tvm.relay.expr.Tuple):
        assert index < len(expr.fields)
        new_fields = list(expr.fields)
        new_fields[index] = new_arg
        new_expr = relay.Tuple(new_fields)
    elif isinstance(expr, tvm.relay.expr.TupleGetItem):
        assert index == 0, "The input index of TupleGetItem must be 0, cannot be {}".format(index)
        assert isinstance(new_arg, tvm.relay.expr.Tuple), "new_arg must be Tuple"
        new_expr = relay.TupleGetItem(new_arg, expr.index)
    elif isinstance(expr, tvm.relay.expr.Call):
        assert index < len(expr.args)
        new_args = list(expr.args)
        new_args[index] = new_arg
        new_expr = relay.Call(expr.op, new_args, expr.attrs, expr.type_args)
    else:
        raise RuntimeError("replace_arg_of_expr don't support {}".format(type(expr)))
    return new_expr


class ExtractParamsPass(ExprMutator):
    """ExtractParamsPass"""

    def __init__(self):
        super().__init__()
        self.params = {}
        self.func_vars = []
        self.count = 0

    def visit_call(self, call):
        """visit_call"""
        visit = super().visit_call(call)
        current_index = self.count
        self.count += 1
        for i in range(len(visit.args)):
            arg = visit.args[i]
            if isinstance(arg, tvm.relay.Constant) and arg.data.shape:
                var_name = "{}_{}_arg_{}".format(visit.op.name, current_index, i)
                var = relay.var(var_name, shape=arg.data.shape, dtype=arg.data.dtype)
                self.params[var_name] = arg.data
                self.func_vars.append(var)
                visit = replace_arg_of_expr(visit, i, var)
        return visit

    def visit_tuple_getitem(self, op):
        visit = super().visit_tuple_getitem(op)
        self.count += 1
        return visit

    def visit_tuple(self, tup):
        """visit_tuple"""
        visit = super().visit_tuple(tup)
        current_index = self.count
        self.count += 1
        for i in range(len(visit.fields)):
            arg = visit.fields[i]
            if isinstance(arg, tvm.relay.Constant):
                var_name = "tuple_{}_arg_{}".format(current_index, i)
                var = relay.var(var_name, shape=arg.data.shape, dtype=arg.data.dtype)
                self.params[var_name] = arg.data
                self.func_vars.append(var)
                visit = replace_arg_of_expr(visit, i, var)
        return visit

    def visit_function(self, fn):
        visited = super().visit_function(fn)
        fun_vars = list(visited.params)
        fun_vars.extend(self.func_vars)
        return relay.Function(fun_vars, visited.body, visited.ret_type, visited.type_params)

    def run(self, expr):
        visited = self.visit(expr)
        return visited, self.params


class GetOutput(ExprMutator):
    """GetOutput"""

    def __init__(self):
        super().__init__()
        self.node_list = []

    def visit_call(self, call):
        visited = super().visit_call(call)
        if is_call(visited, ["cast", "zero", "great_equal", "where"]):
            return visited
        self.node_list.append(visited)
        return visited

    def visit_function(self, fn):
        super().visit_function(fn)

        return relay.Function(fn.params, relay.Tuple(self.node_list))

    def run(self, graph):
        return self.visit(graph)


class FilterResult(ExprMutator):
    """FilterResult"""

    def __init__(self, result):
        super().__init__()
        self.result = deque(result)
        self.data_map = {}
        self.data_map["input"] = self.result.popleft()
        self.index = 0

    def visit_call(self, call):
        visited = super().visit_call(call)
        if not is_call(visited, ["cast", "zeros", "greater_equal", "where"]):
            self.data_map["node_{}_{}".format(self.index, visited.op.name)] = self.result.popleft()
        self.index = self.index + 1
        return visited

    def visit_tuple(self, tup):
        visited = super().visit_tuple(tup)
        self.index = self.index + 1
        return visited

    def run(self, graph):
        self.visit(graph)
        return self.data_map


class ShowMeta(ExprMutator):
    """Show Meta"""

    def visit_call(self, call):
        visited = super().visit_call(call)

        if len(call.args) > 1:
            if isinstance(call.args[1], relay.Constant) and not call.args[1].checked_type.shape:
                print(call.op.name, call.args[1].data.asnumpy())

        return visited

    def run(self, func):
        self.visit(func)


class CutGraph(ExprMutator):
    """cut graph"""

    def __init__(self):
        super().__init__()
        self.bias_num = 0
        self.end = False
        self.extra_node_num = 0

    def visit_call(self, call):
        visited = super().visit_call(call)

        if not self.end and self.extra_node_num <= 9:
            self.body = visited

        if visited.op.name == "nn.bias_add":
            self.bias_num = self.bias_num + 1
            if self.bias_num == 10:
                self.end = True

        if self.end:
            self.extra_node_num = self.extra_node_num + 1
            if self.extra_node_num >= 9:
                return self.body

        self.body = visited

        return visited

    def visit_function(self, fn):
        params = fn.params
        super().visit_function(fn)

        return relay.Function(params, self.body)

    def run(self, func):
        return self.visit(func)


def extract_module(mod, path, name, batch):
    """extract_module"""

    pre_func = CutGraph().run(mod["main"])

    func, params = ExtractParamsPass().run(pre_func)
    func = relay.frontend.common.infer_type(func)

    # ShowMeta().run(func)

    with open(path + name + "_int.txt", "w+") as f:
        f.write(func.__str__())

    for k, _ in params.items():
        params[k] = params[k].asnumpy()

    with open(path + name + "_params_int.pkl", "wb+") as f:
        pickle.dump(params, f)

    # import json
    # with open(path + name + ".json", "w+") as f:
    #     json.dump(tvm.ir.save_json(mod), f)

    # with open(path + name + "_params.params", "wb+") as f:
    #     f.write(tvm.runtime.save_param_dict(params))

    func = GetOutput().run(pre_func)
    new_mod = tvm.IRModule.from_expr(func)

    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(new_mod, "llvm")
        runtime = tvm.contrib.graph_executor.create(graph, lib, tvm.cpu())
        runtime.set_input(**params)
        result = []
        for key in batch.keys():
            runtime.set_input(key, batch[key])
            result.append(batch[key])

        runtime.run()
        num_outputs = runtime.get_num_outputs()
        for j in range(0, num_outputs):
            result.append(runtime.get_output(j).asnumpy())

        data_map = FilterResult(result).run(func)
        # with open(path + name + "_data_int.pkl", 'wb+') as f:
        #     pickle.dump(data_map, f)
        #     exit()

        j = 0
        xxx = {}
        max_num = 99999
        for key, value in data_map.items():
            xxx[key] = value
            if (j + 1) % max_num == 0 or (j + 1) == len(data_map):
                with open(path + name + "_data_int_{}.pkl".format(int(j / max_num)), "wb+") as f:
                    pickle.dump(xxx, f)
                xxx.clear()

            j = j + 1
