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
"""Edgex relay graph command line debugger tool."""
# pylint: disable=missing-class-docstring,missing-function-docstring,broad-except,import-outside-toplevel
import os
import sys
import json
import tempfile
import argparse
import subprocess
import shutil
import importlib
import inspect
import tvm
from tvm import relay
from tvm import IRModule
from tvm import runtime
from tvm.contrib.edgex.testing import TempOpStrategy, check_edgex_relay_build
from tvm.contrib.edgex.relay.op.strategy import (
    fschedule_general_vu,
    SPECIFIED_FSCHEDULE_OPS,
)


def print_info(msg):
    print("\x1b[32m[INFO] " + msg + "\x1b[0m")


def print_error(msg):
    print("\x1b[31m[ERROR] " + msg + "\x1b[0m")


def get_fused_functions(relay_module, params=None, need_optimize=False, target=None, namehint=None):
    """Get each of the fused ops group in input relay module as a separate relay function
    Parameters
    ----------
    relay_module : Union[relay.Function, relay.IRModule]
        The relay function or module already optimized with op fusing.

    need_optimize : bool
        Whether do optimize the input.

    params : dict
        The relay param dict.

    target : str
        The target specification.

    namehint : str
        Lowered function name hint.

    Returns
    ret : dict
        Dictionary map fused function name to (fused func, local params) pair
    """
    if isinstance(relay_module, relay.Function):
        relay_module = IRModule.from_expr(relay_module)
    if need_optimize:
        relay_module = relay.build_module.optimize(relay_module, target, params)
    functions = {}
    name_counter = {}

    class FusedNamer(relay.expr_functor.ExprVisitor):
        """relay fused function name getter"""

        def __init__(self):
            super().__init__()
            self.name = "fused"

        def __call__(self, func):
            self.visit_function(func)
            return self.name

        def visit_call(self, call):
            if not isinstance(call.op, tvm.ir.Op):
                raise ValueError("Only primitive call is supported in fused function")
            super().visit_call(call)
            self.name = self.name + "_" + call.op.name

    class FusedOpsVisitor(relay.expr_functor.ExprVisitor):
        """relay visitor to extract fused ops"""

        def __init__(self, global_function):
            super().__init__()
            self.global_function = global_function
            self.global_params = set(global_function.params)

        def run(self):
            self.visit_function(self.global_function)

        def visit_call(self, call):
            if not isinstance(call.op, relay.Function):
                raise ValueError(
                    "The graph should be fused before debug, optimize it or set need_optimize=True"
                )
            fused_func = call.op

            # get unique name
            if isinstance(namehint, bool):
                funcname = FusedNamer()(fused_func)
            elif isinstance(namehint, str):
                funcname = namehint + "_" + str(len(functions))
            else:
                funcname = "fused_" + str(len(functions))
            funcname = funcname.replace(".", "_")
            while True:
                if funcname in name_counter:
                    cnt = name_counter[funcname]
                    name_counter[funcname] += 1
                    funcname = funcname + "_" + str(cnt)
                else:
                    name_counter[funcname] = 1
                    break
            fused_func = fused_func.with_attr("LoweredFunctionNameHint", funcname)

            new_vars = []
            used_params = {}
            for i, t in enumerate(call.args):
                if t in self.global_params:
                    name = t.name_hint
                    if params is not None and name in params:
                        used_params[name] = params[name]
                else:
                    name = "arg%d" % i
                new_vars.append(relay.var(name, type_annotation=t.checked_type))
            single_func = relay.Function(
                new_vars, relay.Call(fused_func, new_vars), call.checked_type
            )
            for arg in call.args:
                super().visit(arg)
            functions[funcname] = (single_func, used_params)

    for _, func in relay_module.functions.items():
        FusedOpsVisitor(func).run()
    return functions


class RelayGraphDebugger:
    def __init__(
        self, mod=None, params=None, workspace_dir=None, fused_op_namehint=None, verbose=False
    ):
        if workspace_dir is None and mod is None:
            print_error("Relay module required")
            sys.exit(-1)
        self.mod_path = mod if isinstance(mod, str) else None
        self.params_path = params if isinstance(params, str) else None
        self.mod, self.params = self.load_module(mod, params)
        if workspace_dir is None:
            workspace_dir = tempfile.mkdtemp(prefix="/tmp/edgex_graph_debug_workspace_")
        if not os.path.exists(workspace_dir):
            os.makedirs(workspace_dir)
        self.workspace_dir = workspace_dir
        self.partitions = None
        self.fused_op_namehint = fused_op_namehint
        self.verbose = verbose
        self.state = None

    def initialize(self):
        self.partitions = get_fused_functions(
            self.mod, need_optimize=False, params=self.params, namehint=self.fused_op_namehint
        )
        self.state = {}
        self.state["last_perop_info"] = {}
        for key in self.partitions:
            self.state["last_perop_info"][key] = {"status": "not_run"}
        self.state["graph_json"] = (
            os.path.basename(self.mod_path) if self.mod_path is not None else "graph.json"
        )
        if self.params is not None:
            self.state["params"] = (
                os.path.basename(self.params_path)
                if self.params_path is not None
                else "graph.params"
            )

    @staticmethod
    def load_module(mod, params):
        if isinstance(mod, str):
            if not os.path.exists(mod):
                print_error("Module path %s not exists." % mod)
                sys.exit(-1)
            with open(mod, "r") as infile:
                mod = tvm.ir.load_json(json.load(infile))
        if isinstance(params, str):
            if not os.path.exists(params):
                print_error("Params path %s not exists." % params)
                sys.exit(-1)
            with open(params, "rb") as infile:
                params = relay.load_param_dict(infile.read())
        return mod, params

    @staticmethod
    def save_module(output_dir, model_name, mod, params):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, f"{model_name}.json"), "w") as outfile:
            json.dump(tvm.ir.save_json(mod), outfile)
        if params is not None:
            with open(os.path.join(output_dir, f"{model_name}.params"), "wb") as outfile:
                outfile.write(runtime.save_param_dict(params))

    def restore_state(self):
        state_file = os.path.join(self.workspace_dir, "state.json")
        if not os.path.exists(state_file):
            print_error("Missing state.json in workspace %s" % self.workspace_dir)
            sys.exit(-1)
        with open(state_file, "r") as infile:
            self.state = json.load(infile)
        return self.state

    def restore(self):
        print_info("Restore graph from %s" % os.path.abspath(self.workspace_dir))
        self.restore_state()
        ops_state = self.state.get("ops", dict())
        self.mod_path = os.path.join(self.workspace_dir, self.state.get("graph_json", "graph.json"))
        if "params" in self.state:
            self.params_path = os.path.join(self.workspace_dir, self.state["params"])
        else:
            self.params_path = None
        self.mod, self.params = self.load_module(self.mod_path, self.params_path)

        ops_dir = os.path.join(self.workspace_dir, "ops")
        self.partitions = {}
        if os.path.isdir(ops_dir):
            for fused_op_name in os.listdir(ops_dir):
                cur_dir = os.path.join(ops_dir, fused_op_name)
                cur_state = ops_state.get(fused_op_name, dict())
                if not os.path.isdir(cur_dir):
                    continue
                mod_path = os.path.join(cur_dir, cur_state.get("graph_json", "graph.json"))
                params_path = os.path.join(cur_dir, cur_state.get("params", "graph.params"))
                if not os.path.exists(params_path):
                    params_path = None
                self.partitions[fused_op_name] = self.load_module(mod_path, params_path)

    def dump_state(self):
        state_file = os.path.join(self.workspace_dir, "state.json")
        with open(state_file, "w") as infile:
            json.dump(self.state, infile)

    def dump(self, override=False):
        print_info("Dumping graph to %s" % os.path.abspath(self.workspace_dir))
        state_file = os.path.join(self.workspace_dir, "state.json")
        if os.path.exists(state_file):
            if override:
                shutil.rmtree(self.workspace_dir)
            else:
                print_error(
                    "Working directory %s already exists, specify --override/-f to remove old data"
                    % self.workspace_dir
                )
                sys.exit(-1)

        ops_dir = os.path.join(self.workspace_dir, "ops")
        if not os.path.exists(ops_dir):
            os.makedirs(ops_dir)
        for fused_op_name in self.partitions:
            mod, params = self.partitions[fused_op_name]
            cur_dir = os.path.join(ops_dir, fused_op_name)
            if not os.path.exists(cur_dir):
                os.makedirs(cur_dir)
            with open(os.path.join(cur_dir, "graph.json"), "w") as outfile:
                json.dump(tvm.ir.save_json(mod), outfile)
            with open(os.path.join(cur_dir, "astext.txt"), "w") as outfile:
                outfile.write(mod.astext(False))
            if params is not None:
                with open(os.path.join(cur_dir, "graph.params"), "wb") as outfile:
                    outfile.write(runtime.save_param_dict(params))
        with open(os.path.join(self.workspace_dir, self.state["graph_json"]), "w") as outfile:
            json.dump(tvm.ir.save_json(self.mod), outfile)
        if self.params is not None:
            with open(os.path.join(self.workspace_dir, self.state["params"]), "wb") as outfile:
                outfile.write(runtime.save_param_dict(self.params))
        self.dump_state()

    @staticmethod
    def simple_run(mod, params=None, result_json=None):
        with TempOpStrategy(
            [x for x in tvm.ir.Op.list_op_names() if x not in SPECIFIED_FSCHEDULE_OPS],
            "edgex",
            fschedule=fschedule_general_vu,
        ):
            check_result = check_edgex_relay_build(
                mod,
                params,
                check_cpu=True,
                test_fused=True,
                cpu_use_tir=False,
                rmse=0.01,
                nothrow=True,
            )
        if check_result:
            if result_json is not None:
                with open(result_json, "w") as outfile:
                    outfile.write(json.dumps({"rmse": str(check_result.rmse)}))
            if not check_result.success:
                sys.exit(-1)
        return check_result

    def show(self, fused_op_name=None):
        if fused_op_name is None:
            print(self.mod.astext(False))
        else:
            if fused_op_name not in self.partitions:
                print_error("Unknown fused function name: %s" % fused_op_name)
                sys.exit(-1)
            else:
                print(self.partitions[fused_op_name][0].astext(False))

    def run_single(self, fused_op_name):
        if self.partitions is None:
            print_error("Fused partitions are not initialized")
            sys.exit(-1)
        if fused_op_name not in self.partitions:
            print_error("Unknown fused function name: %s" % fused_op_name)
            sys.exit(-1)
        mod, params = self.partitions[fused_op_name]
        if params is None:
            params = self.params
        if self.verbose:
            print("Run single fused op: %s\n%s" % (fused_op_name, mod.astext(False)))
        last_run_info = self.state.setdefault("last_perop_info", {})
        info = last_run_info.setdefault(fused_op_name, {})
        info["status"] = "fail"
        self.dump_state()
        check_result = self.simple_run(mod, params)
        info["status"] = "success"
        if check_result is not None:
            info["rmse"] = str(check_result.rmse)
        self.dump_state()

    def run_end2end(self):
        check_edgex_relay_build(
            self.mod,
            params=self.params,
            check_cpu=True,
            test_fused=True,
            cpu_use_tir=False,
            rmse=0.1,
        )

    def run_perop(self, blacklist_ops=None, whitelist_ops=None, multiprocess=True, cont=False):
        if self.partitions is None:
            print_error("Fused partitions are not initialized")
            sys.exit(-1)
        count = 0
        err_count = 0
        skip_count = 0
        for fused_op_name in self.partitions:
            collect_info = self.state["last_perop_info"][fused_op_name]
            if collect_info.get("status", None) == "success" and cont:
                continue
            mod, params = self.partitions[fused_op_name]
            if params is None:
                params = self.params
            # skip if match blacklist/whitelist config
            if blacklist_ops or whitelist_ops:

                class OpFilter(relay.ExprVisitor):
                    def __call__(self, func):
                        self.black = False
                        self.white = False
                        self.visit_function(func)

                    def visit_call(self, call):
                        if isinstance(call.op, tvm.ir.Op):
                            if blacklist_ops and call.op.name in blacklist_ops:
                                self.black = True
                            if whitelist_ops and call.op.name in whitelist_ops:
                                self.white = True
                        super().visit_call(call)

                op_filter = OpFilter()
                op_filter(mod)
                if blacklist_ops and op_filter.black:
                    if self.verbose:
                        print("Skip fused op " + fused_op_name)
                    skip_count += 1
                    continue
                if whitelist_ops and not op_filter.white:
                    if self.verbose:
                        print("Skip fused op " + fused_op_name)
                    skip_count += 1
                    continue

            count += 1
            collect_info["status"] = "failed"
            self.dump_state()
            if self.verbose:
                print_info("Run single fused op: %s\n%s" % (fused_op_name, mod.astext(False)))

            cur_dir = os.path.join(self.workspace_dir, "ops", fused_op_name)
            err_log = os.path.abspath(os.path.join(cur_dir, "log.txt"))
            if multiprocess:
                params_path = os.path.join(cur_dir, "graph.params")
                result_path = os.path.join(cur_dir, "result.json")
                if not os.path.exists(params_path):
                    params_path = self.params_path
                with open(err_log, "w") as logf:
                    json_path = os.path.join(cur_dir, "graph.json")
                    new_envs = dict(os.environ)
                    new_envs["EDGEX_DEBUG_WORKING_DIR"] = cur_dir
                    exitcode = subprocess.call(
                        [
                            sys.executable,
                            __file__,
                            "simple_run",
                            "--json=%s" % json_path,
                            "--params=%s" % params_path,
                            "--output=%s" % result_path,
                        ],
                        stdout=logf,
                        stderr=logf,
                        env=new_envs,
                    )
                collect_info["status"] = "success"
                if os.path.exists(result_path):
                    with open(result_path, "r") as infile:
                        check_result = json.load(infile)
                        rmse = check_result.get("rmse", None)
                        if rmse:
                            collect_info["rmse"] = rmse
                if exitcode != 0:
                    print_error(
                        "Error occured in %s\nexitcode=%d log=%s"
                        % (fused_op_name, exitcode, err_log)
                    )
                    collect_info["status"] = "fail"
                    err_count += 1
            else:
                try:
                    check_result = self.simple_run(mod, params)
                    collect_info["status"] = "success"
                    if check_result is not None:
                        collect_info["rmse"] = check_result.rmse
                except Exception as exception:
                    collect_info["status"] = "fail"
                    err_count += 1
                    if not self.verbose:
                        with open(err_log, "w") as outfile:
                            outfile.write(str(exception))
                        print_error("Error occured in %s, check %s" % (fused_op_name, err_log))
                    else:
                        print(exception)

        print_info(
            "Graph %s pass %d/%d fused partitions, %d failures, %d skiped"
            % (self.mod_path, count - err_count, count, err_count, skip_count)
        )
        print_info("Detailed info recorded in %s" % self.workspace_dir)
        self.show_stats(self.state["last_perop_info"])
        self.dump_state()

    def run_all(self, mode, blacklist_ops=None, whitelist_ops=None, multiprocess=True, cont=False):
        if mode == "default":
            mode = "perop"
        if mode == "end2end":
            self.run_end2end()
        elif mode == "perop":
            self.run_perop(
                blacklist_ops=blacklist_ops,
                whitelist_ops=whitelist_ops,
                multiprocess=multiprocess,
                cont=cont,
            )

    def show_stats(self, collect_info_dict):
        keys = sorted(collect_info_dict.keys())
        import prettytable as pt  # pylint: disable=import-outside-toplevel

        table = pt.PrettyTable()
        table.field_names = ["Function", "Status", "RMSE"]
        for key in keys:
            info = collect_info_dict[key]
            table.add_row([key, info.get("status", "not_run"), info.get("rmse", "")])
        print(table)


def parse_shape_dict(data):
    result = {}
    specs = data.strip().split(";")
    for spec in specs:
        terms = spec.split("=")
        key = terms[0]
        value = terms[1].lstrip("[").rstrip("]")
        value = [int(_.strip()) for _ in value.split(",")]
        result[key] = value
    return result


def parse_args(cmd, extra_args, use_cache=True, require_relay_mod=False, require_main_arg=False):
    # config current workspace directory
    cached_workspace_infofile = os.path.join(
        os.environ.get("HOME", "/tmp"), ".my_graph_debugger_workspace_info"
    )
    cached_workspace_dir = None
    if os.path.exists(cached_workspace_infofile):
        with open(cached_workspace_infofile, "r") as infile:
            cached_workspace_dir = infile.read().strip()
    main_arg = None
    if len(extra_args) > 0 and not extra_args[0].startswith("-"):
        main_arg = extra_args[0]
        extra_args = extra_args[1:]
    require_json_arg = require_relay_mod and main_arg is None
    require_workspace_arg = not cached_workspace_dir and cmd in {"restore", "run"}

    # config argument parser
    parser = argparse.ArgumentParser(prog="relay_debug %s" % cmd)
    parser.add_argument(
        "--json", "-j", type=str, required=require_json_arg, help="Relay graph json file."
    )
    parser.add_argument("--params", "-p", type=str, help="Relay param dict file.")
    parser.add_argument("--output", "-o", type=str, help="Output information/result file.")
    parser.add_argument(
        "--workspace", "-d", type=str, required=require_workspace_arg, help="Debug workspace."
    )
    parser.add_argument(
        "--fused_op_namehint",
        type=lambda x: True if x == "auto" else x,
        default="auto",
        help='Fused op namehint prefix or specify as "auto".',
    )
    parser.add_argument("--input", "-i", type=str, help="Specify input name.")
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["default", "end2end", "perop"],
        default="default",
        help="Specify debug mode to run multiple ops.",
    )
    parser.add_argument(
        "--blacklist",
        type=lambda x: [_.strip() for _ in x.split(",") if _.strip() != ""],
        help="Specify blacklist op names to skip.",
    )
    parser.add_argument(
        "--whitelist",
        type=lambda x: [_.strip() for _ in x.split(",") if _.strip() != ""],
        help="Specify whitelist op names to run.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Print detailed error message to terminal.",
    )
    parser.add_argument(
        "--multiprocess",
        action="store_true",
        default=True,
        help="Whether use multiprocess to run test.",
    )
    parser.add_argument(
        "--override",
        "-f",
        action="store_true",
        default=False,
        help="Whether override existing workspace.",
    )
    parser.add_argument(
        "--continue",
        "-c",
        action="store_true",
        default=False,
        dest="cont",
        help="Continue from last run state.",
    )
    parser.add_argument(
        "--model-format",
        default=None,
        help="Frontend model format.",
    )
    parser.add_argument(
        "--input-shapes",
        default=None,
        type=parse_shape_dict,
        help="Frontend model shape dict.",
    )
    parser.add_argument(
        "--passes",
        type=lambda x: [_.strip() for _ in x.split(",") if _.strip() != ""],
        help="Specify transform passes to run.",
    )
    parser.add_argument(
        "--model-info-file",
        default=None,
        help="Model info Excel file, which contains shape dict, dtype dict, etc.",
    )
    args = parser.parse_args(extra_args)

    # update some parsed arguments
    if main_arg is not None:
        if require_relay_mod:
            if os.path.isdir(main_arg):
                model_name = os.path.basename(main_arg)
                if args.json is None:
                    args.json = os.path.join(main_arg, model_name + ".json")
                if args.params is None:
                    args.params = os.path.join(main_arg, model_name + ".params")
            elif os.path.exists(main_arg):
                if args.json is None:
                    args.json = main_arg
                if main_arg.endswith(".json") and os.path.exists(main_arg[:-5] + ".params"):
                    if args.params is None:
                        args.params = main_arg[:-5] + ".params"
            elif os.path.exists(main_arg + ".json"):
                if args.json is None:
                    args.json = main_arg + ".json"
                if os.path.exists(main_arg + ".params"):
                    if args.params is None:
                        args.params = main_arg + ".params"
            else:
                print_error("Unknown input graph path: " % main_arg)
                sys.exit(-1)
        elif require_main_arg:
            if args.input is None:
                args.input = main_arg
            if args.workspace is None and cmd == "workspace":
                args.workspace = main_arg
    if args.workspace is None:
        if cached_workspace_dir is not None:
            args.workspace = cached_workspace_dir
        else:
            args.workspace = tempfile.mkdtemp(prefix="/tmp/edgex_graph_debug_workspace_")

    # store current used workspace directory
    print_info("Current workspace directory: %s" % os.path.abspath(args.workspace))
    if use_cache:
        with open(cached_workspace_infofile, "w") as outfile:
            outfile.write(os.path.abspath(args.workspace) + "\n")
    return args


CMD_HANDLER_DICT = {}


class CmdHandler:
    def __init__(
        self, func, doc="", is_debugger_cmd=False, require_relay_mod=False, require_main_arg=False
    ):
        self.func = func
        self.doc = doc
        self.is_debugger_cmd = is_debugger_cmd
        self.require_relay_mod = require_relay_mod
        self.require_main_arg = require_main_arg


def show_arg_help():
    msg = f"Available commands are: \n\n"
    for cmd in CMD_HANDLER_DICT:
        handler = CMD_HANDLER_DICT[cmd]
        msg += "[" + cmd + "]\n"
        content = handler.doc.lstrip("\n")
        indent = " ".join(["" for _ in range(len(content) - len(content.lstrip()))])
        for line in content.split("\n"):
            line = "    " + indent + line + "\n"
            msg += line
    print(msg)


def register_cmd(
    cmd, doc="", is_debugger_cmd=False, require_relay_mod=False, require_main_arg=False
):
    """Rgister a command handler

    Parameters
    ----------
    cmd : str
        Command name, used as `relay_debug [cmd] [--arg...]`

    doc : str
        Help message for the command

    is_debugger_cmd : bool
        Create a graph debugger object with the handler

    require_relay_mod : bool
        The cmd use relay module as input

    require_main_arg : bool
        Command require a main argument, as `relay_debug [cmd] [main_arg] [--arg...]`

    """

    def func(handler):
        CMD_HANDLER_DICT[cmd] = CmdHandler(
            handler, doc, is_debugger_cmd, require_relay_mod, require_main_arg
        )
        return handler

    return func


def dispatch_cmd(cmd, extra_args):
    """execute the command"""
    if cmd not in CMD_HANDLER_DICT:
        print_error("Unknown command: " + cmd)
        show_arg_help()
        sys.exit(-1)

    handler = CMD_HANDLER_DICT[cmd]
    args = parse_args(
        cmd,
        extra_args,
        use_cache=handler.is_debugger_cmd or cmd == "workspace",
        require_relay_mod=handler.require_relay_mod,
        require_main_arg=handler.require_main_arg,
    )
    debugger = None
    if handler.is_debugger_cmd:
        debugger = RelayGraphDebugger(
            mod=args.json,
            params=args.params,
            workspace_dir=args.workspace,
            fused_op_namehint=args.fused_op_namehint,
            verbose=args.verbose,
        )
        handler.func(args=args, debugger=debugger)
    else:
        handler.func(args)


@register_cmd(
    "simple_run",
    require_relay_mod=True,
    doc="""
Compile and execute input relay module.
- relay_debug simple_run -j resnet50.json -p resnet50.params
- relay_debug simple_run resnet50 (if .json and .params take same prefix)
""",
)
def simple_run(args):
    mod, params = RelayGraphDebugger.load_module(args.json, args.params)
    RelayGraphDebugger.simple_run(mod, params, result_json=args.output)


@register_cmd(
    "show",
    require_main_arg=True,
    doc="""
Show current relay graph.
- relay_debug show resnet50.json
- relay_debug show (show curent graph)
- relay_debug show funcname (show single fused function)
""",
)
def show_debug_graph(args):
    if args.input is not None and os.path.exists(args.input):
        mod, _ = RelayGraphDebugger.load_module(args.input, args.params)
        print(mod.astext(False))
        return
    debugger = RelayGraphDebugger(
        mod=args.json,
        params=args.params,
        workspace_dir=args.workspace,
        fused_op_namehint=args.fused_op_namehint,
        verbose=args.verbose,
    )
    debugger.restore()
    debugger.show(args.input)


@register_cmd(
    "status",
    is_debugger_cmd=True,
    doc="""
Show current graph's debug status.
- relay_debug status
""",
)
def list_debug_status(args, debugger):  # pylint: disable=unused-argument
    state = debugger.restore_state()
    debugger.show_stats(state["last_perop_info"])


@register_cmd(
    "init",
    is_debugger_cmd=True,
    require_relay_mod=True,
    require_main_arg=True,
    doc="""
Initialize graph debugger workspace.
- relay_debug init -j resnet50.json -p resnet50.params
- relay_debug init resnet50 (if .json and .params take same prefix)
- relay_debug init resnet50 -d myworkspace/resnet50 (select workspace directory)
""",
)
def init_debug_status(args, debugger):
    debugger.initialize()
    debugger.dump(override=args.override)


@register_cmd(
    "workspace", is_debugger_cmd=False, require_main_arg=True, doc="Change debug workspace"
)
def change_workspace(args):  # pylint: disable=unused-argument
    pass


@register_cmd(
    "run",
    is_debugger_cmd=True,
    require_main_arg=True,
    doc="""
Compile and run current relay module in debugger workspace.
- relay_debug run (run all graph)
- relay_debug run funcname (run one fused function)
- relay_debug run -c (continue from last status)
""",
)
def run(args, debugger):
    debugger.restore()
    if args.input is not None:
        debugger.run_single(args.input)
    else:
        debugger.run_all(
            args.mode,
            blacklist_ops=args.blacklist,
            whitelist_ops=args.whitelist,
            multiprocess=args.multiprocess,
            cont=args.cont,
        )


def gen_tflite_model_info(excel_dir):
    import pandas as pd

    xls = pd.ExcelFile(excel_dir)
    data_frame = xls.parse(xls.sheet_names[0])

    models = {}
    for col in data_frame.to_dict("records"):
        info = {}
        shape_dict = {}
        dtype_dict = {}
        name_list = str(col["input_name"]).split(";")
        shape_list = col["input_shape"].split(";")

        for i, item in enumerate(shape_list):
            dtype, shape = item.split("[")
            shape = [int(x) for x in shape[:-1].split(",")]
            shape_dict[name_list[i]] = shape
            dtype_dict[name_list[i]] = dtype

        info["shape_dict"] = shape_dict
        info["dtype_dict"] = dtype_dict
        models[col["model"]] = info

    return models


@register_cmd(
    "convert",
    require_main_arg=True,
    doc="""
Convert relay module from frontend, dump result json and params to output directory.
- relay_debug convert resnet50.onnx -o mydir/resnet50
""",
)
def convert_frontend_model(args):
    if args.input is None:
        print_error("No model path specified")
        sys.exit(-1)
    from tvm.driver.tvmc.frontends import load_model

    model_format = args.input.rsplit(".")[-1]
    if model_format == "tflite":
        if args.model_info_file:
            file_name = args.model_info_file
        else:
            file_name = "/data/share/400tvm_models/nnp400_tflite.xlsx"
        name = os.path.basename(args.input)[:-7]
        model = gen_tflite_model_info(file_name).get(name)
        if model is None:
            print_error("Found none info for " % name)
            sys.exit(-1)
        print(name, model)
        tvmc_model = load_model(
            args.input,
            model_format=model_format,
            shape_dict=model["shape_dict"],
            dtype_dict=model["dtype_dict"],
        )
    else:
        tvmc_model = load_model(args.input, model_format=model_format, shape_dict=args.input_shapes)

    model_path = args.input
    model_name = os.path.basename(model_path)
    model_name = model_name[: model_name.rfind(".")]
    mod, params = tvmc_model.mod, tvmc_model.params
    if args.output:
        print_info(f"Dump converted relay model to {args.output}")
        RelayGraphDebugger.save_module(args.output, model_name, mod, params)
    else:
        print(mod.astext(False))


@register_cmd(
    "quantize",
    require_relay_mod=True,
    doc="""
Run edgex quantization on input relay module, dump result json and params to output directory.
- relay_debug quantize -j resnet50.json -p resnet50.params -o quantized/resnet50
- relay_debug quantize resnet50 -o quantized/resnet50 (if .json and .params take same prefix)
""",
)
def quantize(args):
    if args.output is None:
        raise ValueError("Must specify an output directory")
    mod, params = RelayGraphDebugger.load_module(args.json, args.params)
    model_name = os.path.basename(args.json)
    if model_name.endswith(".json"):
        model_name = model_name[:-5]
    quant_mod, quant_params = relay.quantization.run_quantization(
        model_name, mod, params, fast_mode=True, config=None
    )
    print_info(f"Dump quantized relay model to {args.output}")
    RelayGraphDebugger.save_module(args.output, model_name, quant_mod, quant_params)


@register_cmd(
    "fuse",
    require_relay_mod=True,
    doc="""
Run edgex fusion stitching pass on input relay module, dump result json and params to output directory.
- relay_debug fuse resnet50.json -o fused/resnet50
""",
)
def run_fusion_stitch(args):
    mod, params = RelayGraphDebugger.load_module(args.json, args.params)
    model_name = os.path.basename(args.json)
    if model_name.endswith(".json"):
        model_name = model_name[:-5]
    if args.output is None:
        raise ValueError("Must specify an output directory")
    mod = tvm.contrib.edgex.relay.transform.FusionStitch()(mod)
    RelayGraphDebugger.save_module(args.output, model_name, mod, params)


@register_cmd(
    "run_pass",
    require_relay_mod=True,
    doc="""
Run specified transform pass on input relay module, dump result json and params to output directory.
- relay_debug run_pass -j resnet50.json -p resnet50.params -o transformed/resnet50
- relay_debug run_pass resnet50 -o transformed/resnet50 (if .json and .params take same prefix)
""",
)
def run_relay_passes(args):
    mod, params = RelayGraphDebugger.load_module(args.json, args.params)
    model_name = os.path.basename(args.json)
    if model_name.endswith(".json"):
        model_name = model_name[:-5]
    if not isinstance(args.passes, list):
        raise ValueError("Require --passes argument")
    for pass_name in args.passes:
        idx = pass_name.rfind(".")
        modname, funcname = pass_name[:idx], pass_name[idx + 1 :]
        if not modname.startswith("tvm"):
            modname = "tvm." + modname
        pass_func = getattr(importlib.import_module(modname), funcname)()
        arg_spec = inspect.getfullargspec(pass_func)
        if len(arg_spec.args) == 1:
            # guess this is func(mod)
            mod = pass_func(mod)
        elif len(arg_spec.args) == 2:
            # guess this is func(mod, params)
            mod, params = pass_func(mod, params)
        else:
            raise ValueError(f"{pass_name} may not be a valid relay transformation")
    if args.output:
        RelayGraphDebugger.save_module(args.output, model_name, mod, params)
    else:
        print(mod.astext(False))


def main():
    """command line entry"""
    if len(sys.argv) < 2:
        print_error("Missing command line arguments")
        show_arg_help()
        sys.exit(-1)
    override_ops = [x for x in tvm.ir.Op.list_op_names() if x not in SPECIFIED_FSCHEDULE_OPS]
    with TempOpStrategy(override_ops, "edgex", fschedule=fschedule_general_vu):
        dispatch_cmd(sys.argv[1], sys.argv[2:])


if __name__ == "__main__":
    main()
