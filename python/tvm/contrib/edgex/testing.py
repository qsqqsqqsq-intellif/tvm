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
import json
import os
import numpy as np
import tvm
from tvm import relay
from tvm.contrib.edgex import build_config_nnp
from tvm.ir.module import IRModule
from tvm.contrib import graph_executor
from tvm.contrib.edgex.relay.transform import ConvertDepthwiseConv2D, PostScheduleArgumentRewrite
from tvm.contrib.edgex.relay.backend import ScheduleCache
from tvm.relay.build_module import bind_params_by_name


class UnsupportedDetector(relay.ExprVisitor):
    """Detect fused op currently not supported"""

    def __init__(self):
        super().__init__()
        self.has_conv = False

    def __call__(self, func):
        self.match = False
        self.visit_function(func)
        return self.match

    def visit_call(self, call):
        if isinstance(call.op, tvm.ir.Op):
            if (
                call.op.name == "cast"
                and isinstance(call.args[0], relay.Var)
                and call.args[0].type_annotation.dtype == "int32"
            ):
                self.match = True
        if isinstance(call.attrs, relay.op.op_attrs.Conv2DAttrs) and call.attrs.groups > 1:
            self.match = True
        super().visit_call(call)


class OffloadUnsupported(relay.ExprMutator):
    """Offload unsupported ops to cpu"""

    def __init__(self):
        super().__init__()
        self.offload_cpu = False
        self.detector = UnsupportedDetector()

    def visit_call(self, call):
        if isinstance(call.op, relay.Function):
            args = [self.visit(x) for x in call.args]
            if self.detector(call.op):
                new_f = relay.Function(call.op.params, call.op.body, call.op.ret_type)
                new_f = new_f.with_attr("Primitive", 1)
                new_f = new_f.with_attr("DeviceType", 1)
                return relay.annotation.on_device(relay.Call(new_f, args), "cpu")
            return relay.annotation.on_device(relay.Call(call.op, args), "edgex")
        return super().visit_call(call)


class EdgexFallbackOpStrategy:
    """A relay op strategy reuse origin strategy's compute"""

    def __init__(self, op_name):
        self.op_name = op_name
        origin_fstrategy = relay.op.get(op_name).get_attr("FTVMStrategy")
        assert origin_fstrategy is not None
        self.origin_func = origin_fstrategy.get_packed_func()
        if self.origin_func is None:
            with tvm.target.Target("llvm"):
                self.origin_func = origin_fstrategy.get_packed_func()

    def __call__(self, attrs, inputs, out_type, target):
        strategy = relay.op.OpStrategy()
        default_strategy = self.origin_func(attrs, inputs, out_type, "llvm")
        for spec in default_strategy.specializations:

            def wrap_schedule(fsched):
                def _call_llvm_fsched(attrs, outputs, _):
                    return fsched(attrs, outputs, tvm.target.Target("llvm"))

                return _call_llvm_fsched

            for default_impl in spec.implementations:
                strategy.add_implementation(
                    default_impl.compute,
                    wrap_schedule(default_impl.schedule),
                    name=default_impl.name,
                )
        return strategy


class TempOpStrategy(object):
    """Context manager for a temporary registered op strategy
    We can create temporary strategy registry as below:
     (1) with TempOpStrategy("nn.conv2d", "llvm", cpu_strategy):
             lib = relay.build(...)

     (2) with TempOpStrategy("nn.conv2d", ["llvm", "edgex"], general_strategy):
             lib = relay.build(...)
    """

    def __init__(self, op_name, target, fstrategy=None, fschedule=None):
        op_names = op_name if isinstance(op_name, list) else [op_name]
        targets = target if isinstance(target, list) else [target]
        self.origin_fstrategies = {}  # opname -> tgt_key -> strategy
        self.origin_fschedules = {}  # opname -> tgt_key -> fschedule
        for name in op_names:
            cur_fstrategy = fstrategy
            if cur_fstrategy is None and fschedule is None:
                cur_fstrategy = EdgexFallbackOpStrategy(name)
            if cur_fstrategy is not None:
                origin_fstrategies = {}
                generic_fstrategy = relay.op.get(name).get_attr("FTVMStrategy")
                if generic_fstrategy is None:
                    generic_fstrategy = tvm.target.get_native_generic_func(name + "_strategy")
                for tgt in targets:
                    with tvm.target.Target(tgt) as target_obj:
                        origin_func = generic_fstrategy.get_packed_func()
                        for tgt_key in target_obj.keys:
                            origin_fstrategies[tgt_key] = origin_func
                            generic_fstrategy.register(cur_fstrategy, tgt_key, allow_override=True)
                self.origin_fstrategies[name] = origin_fstrategies

            if fschedule is not None:
                origin_fschedules = {}
                generic_fschedule = None
                if relay.op.get(name).has_attr("FEdgeXSchedule"):
                    generic_fschedule = relay.op.get(name).get_attr("FEdgeXSchedule")
                if generic_fschedule is None:
                    generic_fschedule = tvm.target.get_native_generic_func(
                        name + "_edgex_fschedule"
                    )
                    tvm.ir.register_op_attr(name, "FEdgeXSchedule", generic_fschedule)
                for tgt in targets:
                    with tvm.target.Target(tgt) as target_obj:
                        origin_func = generic_fschedule.get_packed_func()
                        for tgt_key in target_obj.keys:
                            origin_fschedules[tgt_key] = origin_func
                            generic_fschedule.register(fschedule, tgt_key, allow_override=True)
                self.origin_fschedules[name] = origin_fschedules

    def __enter__(self):
        return self

    def __exit__(self, typ, value, traceback):
        for name in self.origin_fstrategies:
            generic_fstrategies = relay.op.get(name).get_attr("FTVMStrategy")
            origin_fstrategies = self.origin_fstrategies[name]
            for tgt in origin_fstrategies:
                generic_fstrategies.register(origin_fstrategies[tgt], tgt, allow_override=True)
        for name in self.origin_fschedules:
            generic_fschedule = relay.op.get(name).get_attr("FEdgeXSchedule")
            origin_fschedules = self.origin_fschedules[name]
            for tgt in origin_fschedules:
                generic_fschedule.register(origin_fschedules[tgt], tgt, allow_override=True)


class TempRegisterFunc:
    """Context manager for a temporary registered function"""

    def __init__(self, func_name, function):
        self.origin_func = tvm.get_global_func(func_name, allow_missing=True)
        self.func_name = func_name
        self.function = function

    def __enter__(self):
        tvm.register_func(self.func_name, self.function, override=True)
        return self

    def __exit__(self, typ, value, traceback):
        tvm.register_func(self.func_name, self.origin_func, override=True)


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
            fused_func = fused_func.with_attr("PrimFuncName", funcname)

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


class CheckResult:
    """Record the data check results"""

    def __init__(self, inputs, outputs, expects, rmse, success):
        self.inputs = inputs
        self.outputs = outputs
        self.expects = expects
        self.rmse = rmse
        self.success = success


class RelayToTIRAnnotator(relay.ExprMutator):
    """Mark all fused functions with Compiler=edgex"""

    def visit_call(self, call):
        call = super().visit_call(call)
        if isinstance(call.op, relay.Function) and "Primitive" in call.op.attrs:
            func = call.op.with_attr("Compiler", "edgex")
            return relay.Call(func, call.args, call.attrs, call.type_args, call.span)
        return call


def check_numpy_result(
    result, expect, nothrow=False, ref_inputs=None, rmse=None, rtol=1e-5, atol=1e-5
):
    """Helper function to compare result and expect tensor
    Parameters
    ----------
    result : numpy.ndarray
        result tensor.

    expect : numpy.ndarray
        expect tensor.

    nothrow : bool
        Do not raise error if check result failed.

    ref_inputs : list
        Input data list as a reference.

    rmse : float
        If specified, check root-mean-square deviation between results and expects
        instead of close assertion.

    rtol : float
        Relative tolerance.

    atol : float, optional
        Absolute tolerance.
    """
    ndim = len(result.shape)
    assert ndim == len(expect.shape) and all(
        [result.shape[i] == expect.shape[i] for i in range(ndim)]
    ), f"Tensor mismatch: result is {result.shape}, expect {expect.shape}"
    actual_rmse = np.sqrt(np.mean((result - expect).astype("float32") ** 2))

    def do_check():
        if rmse is not None:
            assert not np.isnan(actual_rmse) and actual_rmse < rmse, (
                "RMSE out of bound: %f" % actual_rmse
            )
        else:
            tvm.testing.assert_allclose(result, expect, rtol=rtol, atol=atol)

    success = True
    if nothrow:
        try:
            do_check()
        except BaseException as exception:
            success = False
            print(exception)
    else:
        do_check()
    return CheckResult(ref_inputs, [result], [expect], actual_rmse, success)


def get_edgex_plan_device_config(pass_ctx):
    """Get [cpu + edgex] config for relay device plan pass"""
    cpu = tvm.device("cpu")
    cpu_target = tvm.target.Target("llvm")
    edgex_dev = tvm.edgex()
    edgex_target = tvm.target.edgex()
    return tvm.target.make_compilation_config(
        pass_ctx,
        {
            tvm.tir.IntImm("int32", cpu.device_type): cpu_target,
            tvm.tir.IntImm("int32", edgex_dev.device_type): edgex_target,
        },
        cpu_target,
    )


def check_edgex_relay_build(
    function,
    params=None,
    numpy_func=None,
    check_edgex=True,
    check_cpu=True,
    cpu_use_tir=True,
    data_range=None,
    input_data=None,
    test_fused=False,
    rmse=None,
    nothrow=False,
    legacy_lower=False,
):
    """build and check edgex from relay
    Parameters
    ----------
    function : relay.Function
        The relay function to build and test.

    params : dict
        The relay param dict.

    numpy_func : function
        The compatible computation logic in numpy to get the expect output.

    check_edgex : bool
        Whether run on edgex

    check_cpu : bool
        Whether run on cpu

    cpu_use_tir : bool
        Enable relay.backend.use_meta_schedule option on cpu build

    data_range :
        Testdata sample range.

    input_data : list[numpy.ndarray] or dict
        Input data bindings, if not given, will sample data randomly.

    test_fused : bool
        Test fused edgex relay graph

    rmse : float
        If specified, check root-mean-square deviation between results and expects
        instead of close assertion.

    nothrow : bool
        Do not raise error if check result failed.

    legacy_lower : bool
        legacy relay to tir lowering, the legacy behavior is reserved for debug purpose
        until relay_to_tir functionality is totally checked.
    """
    arrs = {}
    if isinstance(function, IRModule):
        function = function.functions.items()[0][1]
    for idx, arg in enumerate(function.params):
        name = arg.name_hint
        if isinstance(input_data, list):
            if idx < len(input_data) and input_data[idx] is not None:
                arrs[name] = input_data[idx]
                continue
        elif isinstance(input_data, dict):
            if name in input_data:
                arrs[name] = input_data[name]
                continue
        if params and name in params:
            data = params[name]
            if isinstance(data, tvm.nd.NDArray):
                data = data.numpy()
            arrs[name] = data
            continue
        dtype = arg.type_annotation.dtype
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
        shape = [int(x) for x in arg.type_annotation.shape]
        if dtype.startswith("i") or dtype.startswith("u"):
            arrs[name] = np.random.randint(data_range[0], data_range[1], size=shape).astype(dtype)
        else:
            arrs[name] = np.random.uniform(-data_range[0], data_range[1], size=shape).astype(dtype)

    expect = None
    if numpy_func is not None:
        data_list = [arrs[p.name_hint] for p in function.params]
        expect = numpy_func(*data_list)

    def get_relay_output(relay_mod, target, ctx):
        if not isinstance(ctx, (tuple, list)):
            ctx = [ctx]
        lib = relay.build(relay_mod, target=target, params=params)
        m = graph_executor.GraphModule(lib["default"](*ctx))
        for name, arr in arrs.items():
            # param is bind already
            if params and name in params:
                continue
            m.set_input(name, arr)
        m.run()
        return m.get_output(0)

    class OnDeviceDetector(relay.ExprVisitor):
        """Detect on device annotations"""

        def __init__(self):
            super().__init__()
            self.has_cpu = False
            self.has_edgex = False
            self.has_annotations = False

        def visit_call(self, call):
            if isinstance(call.attrs, relay.op.op_attrs.OnDeviceAttrs):
                self.has_annotations = True
                if call.attrs.virtual_device.device_type_int == tvm.cpu().device_type:
                    self.has_cpu = True
                elif call.attrs.virtual_device.device_type_int == tvm.edgex().device_type:
                    self.has_edgex = True
            super().visit_call(call)

    class OnDeviceCleaner(relay.ExprMutator):
        """Clean on device annotation"""

        def visit_call(self, call):
            if isinstance(call.attrs, relay.op.op_attrs.OnDeviceAttrs):
                return self.visit(call.args[0])
            return super().visit_call(call)

    # detect heterogeneous graph
    device_annotation_detector = OnDeviceDetector()
    device_annotation_detector.visit_function(function)

    if check_cpu:
        with tvm.ir.transform.PassContext(config={"relay.backend.use_meta_schedule": cpu_use_tir}):
            if device_annotation_detector.has_annotations:
                # clear on_device annotations for pure cpu run
                cpu_mod = IRModule.from_expr(OnDeviceCleaner().visit_function(function))
            else:
                cpu_mod = IRModule.from_expr(function)
            cpu_result = get_relay_output(cpu_mod, "llvm", tvm.cpu())
        if expect is None:
            expect = cpu_result.numpy()
        else:
            check_numpy_result(cpu_result.numpy(), expect, rmse=rmse)

    if check_edgex:
        pass_ctx = build_config_nnp()
        edgex_target = tvm.target.edgex()
        cpu_target = tvm.target.Target("llvm")
        plan_device_cfg = get_edgex_plan_device_config(pass_ctx)
        edgex_mod = IRModule.from_expr(function)
        edgex_mod = relay.transform.InferType()(edgex_mod)
        edgex_mod = relay.transform.PlanDevices(plan_device_cfg)(edgex_mod)
        if params is not None:
            func_with_params = bind_params_by_name(edgex_mod["main"], params)
            edgex_mod = tvm.ir.IRModule.from_expr(func_with_params)
        if legacy_lower:
            with ScheduleCache():
                with pass_ctx:
                    if test_fused:
                        edgex_mod = PostScheduleArgumentRewrite()(edgex_mod)
                        edgex_mod = relay.transform.FoldConstant()(edgex_mod)
                    if device_annotation_detector.has_cpu:
                        targets = {"edgex": edgex_target, "cpu": cpu_target}
                        ctxs = [tvm.edgex(), tvm.cpu()]
                        edgex_result = get_relay_output(edgex_mod, targets, ctxs)
                    else:
                        edgex_result = get_relay_output(edgex_mod, edgex_target, tvm.edgex())
        else:
            # annotate Compiler=edgex to all fused functions
            if test_fused:
                function = RelayToTIRAnnotator().visit(edgex_mod["main"])
                edgex_mod = IRModule.from_expr(function)
            with pass_ctx:
                if device_annotation_detector.has_cpu:
                    targets = {"edgex": edgex_target, "cpu": cpu_target}
                    ctxs = [tvm.edgex(), tvm.cpu()]
                    edgex_result = get_relay_output(edgex_mod, targets, ctxs)
                else:
                    edgex_result = get_relay_output(edgex_mod, edgex_target, tvm.edgex())
        if expect is not None:
            return check_numpy_result(edgex_result.numpy(), expect, rmse=rmse, nothrow=nothrow)
    return None


def check_edgex_tir_build(
    name,
    prim_func,
    cpu_prim_func=None,
    numpy_func=None,
    edgex_fschedule=None,
    check_edgex=True,
    check_cpu=True,
    need_lower=True,
    data_range=None,
    input_data=None,
    rmse=None,
    output_idx=None,
):
    """build and check edgex tir module
    Parameters
    ----------
    name : str
        The kernel name to use.

    primfunc : tir.PrimFunc
        The tir function to build and test.

    cpu_prim_func : tir.PrimFunc
        The cpu tir function to build and test.

    edgex_fschedule : func
        Schedule before lowering, can be used if it changes argument layouts on device.
        Must be of signature (attrs, primfunc, target) -> primfunc

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

    output_idx : int or List[int]
        If specified, compare only the tensor at these indices.
    """
    # prepare test data
    arrs = []
    if output_idx is not None and not isinstance(output_idx, (set, list)):
        output_idx = {output_idx}
    for idx, param in enumerate(prim_func.params):
        if input_data is not None and idx < len(input_data) and input_data[idx] is not None:
            arrs.append(input_data[idx])
            continue
        buffer = prim_func.buffer_map[param]
        shape = [int(x) for x in buffer.shape]
        dtype = buffer.dtype
        if output_idx is not None and idx in output_idx:
            arrs.append(np.zeros(shape).astype(dtype))
            continue
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
            arrs.append(
                np.random.randint(
                    data_range[0], max(data_range[0] + 1, data_range[1]), size=shape
                ).astype(dtype)
            )
        else:
            arrs.append(np.random.uniform(data_range[0], data_range[1], size=shape).astype(dtype))

    # schedule on device differently if optional edgex_fschedule specified
    if cpu_prim_func is None:
        cpu_prim_func = prim_func
    cpu_arrs = arrs
    relay_arg_rewrite_mod = None
    if edgex_fschedule is not None:
        prim_func = edgex_fschedule({}, prim_func, tvm.target.edgex())
        if prim_func.attrs is not None:
            for k, v in prim_func.attrs.items():
                if k == "post_schedule_argument_rewrite":
                    relay_arg_rewrite_mod = tvm.ir.load_json(v)
                    executor = relay.create_executor(kind="vm")
                    rewritten_args = executor.evaluate(relay_arg_rewrite_mod["forward"])(*arrs)
                    arrs = [_.numpy() for _ in rewritten_args]
                    break

    # compute a expect result by numpy logic if optional numpy_func specified
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

    # cpu execution
    if check_cpu:
        build_input = cpu_prim_func if need_lower else {"llvm": IRModule({name: cpu_prim_func})}
        ctx = tvm.cpu()
        cpu_mod = tvm.build(build_input, [], target="llvm", name=name)
        cpu_tensors = [tvm.nd.array(x, ctx) for x in cpu_arrs]
        cpu_mod(*cpu_tensors)
        cpu_results = [x.numpy() for x in cpu_tensors]
        if expects is None:
            expects = cpu_results
        else:
            for i, (expect, res) in enumerate(zip(expects, cpu_results)):
                if output_idx is not None and not i in output_idx:
                    continue
                check_numpy_result(res, expect, rmse=rmse)

    # device execution
    if check_edgex:
        build_input = prim_func if need_lower else {"edgex": IRModule({name: prim_func})}
        ctx = tvm.edgex()
        with build_config_nnp():
            edgex_mod = tvm.build(build_input, [], target="edgex", name=name)
        edgex_tensors = [tvm.nd.array(x, ctx) for x in arrs]
        edgex_mod(*edgex_tensors)
        if relay_arg_rewrite_mod is not None:
            executor = relay.create_executor(kind="vm")
            edgex_tensors = executor.evaluate(relay_arg_rewrite_mod["backward"])(*edgex_tensors)
        edgex_results = [x.numpy() for x in edgex_tensors]
        if expects is not None:
            for i, (expect, res) in enumerate(zip(expects, edgex_results)):
                if output_idx is not None and not i in output_idx:
                    continue
                check_numpy_result(res, expect, rmse=rmse)


def get_fs_fused_workload(net, fix_norm=False):
    """helper to get network"""
    model_dir = os.getenv("EDGEX_MODELS_DIR", "/tmp")
    mod_file = model_dir + "/pytorch/%s/quantized/%s.json" % (net, net)
    params_file = model_dir + "/pytorch/%s/quantized/%s.params" % (net, net)
    assert os.path.exists(mod_file) and os.path.exists(params_file)
    with open(mod_file, "r") as file:
        mod = tvm.ir.load_json(json.load(file))
    with open(params_file, "rb") as file:
        params = relay.load_param_dict(file.read())

    if fix_norm:
        # fix norm value range temporarily
        params = dict(params.items())
        for k in params:
            if k.find("round_right_shift") >= 0:
                norm = params[k].asnumpy()
                norm = np.minimum(norm, 24)
                norm = np.maximum(norm, 1)
                params[k] = tvm.nd.array(norm)
            elif k.find("multiply") >= 0:
                norm = params[k].asnumpy()
                norm = np.minimum(norm, 127)
                params[k] = tvm.nd.array(norm)
            elif k.find("bias_add") >= 0:
                norm = params[k].asnumpy()
                norm = np.maximum(np.minimum(norm, 2 ** 20), -(2 ** 20))
                params[k] = tvm.nd.array(norm)
        mod, params = ConvertDepthwiseConv2D()(mod, params)

    return mod, params


def get_graph_runtime_output(lib, data, ctx):
    m = graph_executor.GraphModule(lib["default"](*ctx))
    m.set_input("input", data)
    m.run()
    return m.get_output(0).numpy()
