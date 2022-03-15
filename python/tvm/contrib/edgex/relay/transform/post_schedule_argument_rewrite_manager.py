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
# pylint: disable=wildcard-import, redefined-builtin, invalid-name
"""The edgex post schedule argument rewrite manager."""
import json
import tvm
from tvm import tir
from tvm import relay
from tvm.ir import IRModule
from tvm.contrib.edgex.tir.schedule.edgex_schedule import EdgexSchedule


class PostScheduleArgumentRewriteManager:
    """Manage post schedule relay arguments rewrite operations in schedule phase"""

    def __init__(self, schedule: EdgexSchedule, funcname="main"):
        self.funcname = funcname
        self.schedule = schedule
        self.cur_params = []
        self.updates = []
        self.update_func_info()
        self.origin_params = self.cur_params

    def trace_update(self, origin_buf, new_buf, forward_transform, backward_transform):
        """Trace relay rewrite operation to apply to function arguments
        Parameters
        ----------
        origin_buf : Union[tir.Buffer, List[tir.Buffer]]
            The original function buffers before rewrite

        new_buf : Union[tir.Buffer, List[tir.Buffer]]
            The updated function buffers

        forward_transform : Function
            Relay transform callback from origin params to new params

        backward_transform : Function
            Relay transform callback from origin params to new params
        """
        if isinstance(origin_buf, tir.Buffer):
            origin_buf = [origin_buf]
        if isinstance(new_buf, tir.Buffer):
            new_buf = [new_buf]
        origin_local_num = 0
        origin_idxs = []
        for buf in origin_buf:
            idx = self.__get_origin_param_idx(buf)
            if idx < 0:
                origin_local_num += 1
            origin_idxs.append(idx)
        new_local_num = 0
        new_idxs = []
        for buf in new_buf:
            idx = self.__get_new_param_idx(buf)
            if idx < 0:
                new_local_num += 1
            new_idxs.append(idx)
        if new_local_num > 0 or origin_local_num > 0:
            assert (
                len(origin_idxs) == origin_local_num and len(new_idxs) == new_local_num
            ), "Do not support local <-> function param buffer conversions"
            return
        self.updates.append((origin_idxs, new_idxs, forward_transform, backward_transform))
        self.update_func_info()

    def has_argument_rewrite(self):
        return len(self.updates) > 0

    def create_annotated_func(self):
        """Annotate prim function with relay rewrite information."""
        func = self.schedule.mod[self.funcname]
        if not self.has_argument_rewrite():
            return func

        # create relay function perform forward transformation from origin arguments
        origin_relay_args = []
        for i, (param, buf) in enumerate(self.origin_params):
            if buf is not None:
                origin_relay_args.append(relay.var("arg%d" % i, dtype=buf.dtype, shape=buf.shape))
            else:
                origin_relay_args.append(relay.var("arg%d" % i, dtype=param.dtype))
        cur_values = origin_relay_args
        for origin_idxs, new_idxs, forward_transform, _ in self.updates:
            origin_idxs_set = frozenset(origin_idxs)
            new_idxs = sorted(new_idxs)
            inputs = [cur_values[k] for k in origin_idxs]
            outputs = forward_transform(*inputs)
            if isinstance(outputs, relay.Expr):
                outputs = [outputs]
            assert len(outputs) == len(new_idxs), "Illegal relay transformations for arguments"
            i, j, k = 0, 0, 0
            new_values = []
            while True:
                if i < len(new_idxs) and k == new_idxs[i]:
                    new_values.append(outputs[i])
                    i += 1
                elif j < len(cur_values):
                    if j in origin_idxs_set:
                        j += 1
                        continue
                    new_values.append(cur_values[j])
                    j += 1
                else:
                    break
                k += 1
            cur_values = new_values
        relay_forward_func = relay.Function(origin_relay_args, relay.Tuple(cur_values))
        try:
            relay_forward_func = relay.transform.InferType()(
                IRModule.from_expr(relay_forward_func)
            )["main"]
        except tvm.TVMError as e:
            raise RuntimeError(
                "Infer type failed for forward transform:\n"
                + relay_forward_func.astext()
                + "\n"
                + str(e)
            )

        # create relay function perform backward transformation from new arguments
        new_relay_args = []
        for i, ty in enumerate(relay_forward_func.ret_type.fields):
            new_relay_args.append(relay.var("arg%d" % i, type_annotation=ty))
        cur_values = new_relay_args
        for new_idxs, origin_idxs, _, backward_transform in reversed(self.updates):
            origin_idxs_set = frozenset(origin_idxs)
            new_idxs = sorted(new_idxs)
            inputs = [cur_values[k] for k in origin_idxs]
            outputs = backward_transform(*inputs)
            if isinstance(outputs, relay.Expr):
                outputs = [outputs]
            assert len(outputs) == len(new_idxs), "Illegal relay transformations for arguments"
            i, j, k = 0, 0, 0
            new_values = []
            while True:
                if i < len(new_idxs) and k == new_idxs[i]:
                    new_values.append(outputs[i])
                    i += 1
                elif j < len(cur_values):
                    if j in origin_idxs_set:
                        j += 1
                        continue
                    new_values.append(cur_values[j])
                    j += 1
                else:
                    break
                k += 1
            cur_values = new_values
        relay_backward_func = relay.Function(new_relay_args, relay.Tuple(cur_values))
        try:
            relay_backward_func = relay.transform.InferType()(
                IRModule.from_expr(relay_backward_func)
            )["main"]
        except tvm.TVMError as e:
            raise RuntimeError(
                "Infer type failed for backward transform:\n"
                + relay_backward_func.astext()
                + "\n"
                + str(e)
            )

        # transform compatibility check
        self.__check_transform_types(relay_forward_func, self.cur_params, "forward")
        self.__check_transform_types(relay_backward_func, self.origin_params, "backward")

        if func.attrs is None:
            attrs = {}
        else:
            attrs = dict(func.attrs._dict().items())
        rewrite_mod = IRModule({"forward": relay_forward_func, "backward": relay_backward_func})
        json_str = tvm.ir.save_json(rewrite_mod)
        json_str = json.dumps(json.loads(json_str), separators=(",", ":"))
        attrs["post_schedule_argument_rewrite"] = json_str

        dict_attr = tvm.ir.make_node("DictAttrs", **attrs)
        return tir.PrimFunc(
            func.params,
            func.body,
            func.ret_type,
            func.buffer_map,
            func.preflattened_buffer_map,
            dict_attr,
            func.span,
        )

    def update_func_info(self):
        func = self.schedule.mod[self.funcname]
        self.cur_params = [
            (p, func.buffer_map[p] if p in func.buffer_map else None) for p in func.params
        ]

    def __get_new_param_idx(self, buffer):
        """Get parameter index of a function's buffer"""
        func = self.schedule.mod[self.funcname]
        for idx, param in enumerate(func.params):
            if param in func.buffer_map and func.buffer_map[param] == buffer:
                return idx
        return -1

    def __get_origin_param_idx(self, buffer):
        """Get parameter index of a function's buffer"""
        for idx, (_, origin_buffer) in enumerate(self.cur_params):
            if origin_buffer == buffer:
                return idx
        return -1

    def __check_transform_types(self, relay_func, tir_params, desc):
        relay_types = relay_func.ret_type.fields
        if len(relay_types) != len(tir_params):
            raise RuntimeError(
                "%s transformation incompatible: expect %d rewrite outputs but get %d"
                % (desc, len(tir_params), len(relay_types))
            )
        n = len(relay_types)
        for i in range(n):
            param, buffer = tir_params[i]
            if buffer is None:
                expect_ty = tvm.ir.PrimType(param.dtype)
            else:
                expect_ty = tvm.ir.TensorType(buffer.shape, buffer.dtype)
            if expect_ty != relay_types[i]:
                shape = "" if buffer is None else str(buffer.shape)
                raise RuntimeError(
                    (
                        "%dth arg of %s transformation incompatible: expect %s's shape=%s, "
                        + "dtype=%s, but get %s\nrelay func:\n%s"
                    )
                    % (
                        i,
                        desc,
                        param.name,
                        shape,
                        buffer.dtype,
                        relay_types[i],
                        relay_func.astext(False),
                    )
                )
