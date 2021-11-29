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
# pylint: disable=unused-import, missing-function-docstring, disable=no-member
"""The edgex schedule class"""

from __future__ import annotations

from typing import List, Union, Optional

from tvm import tir
from tvm._ffi import register_object as _register_object
from tvm.ir import IRModule
from tvm.runtime import Object, PackedFunc
from tvm.tir import PrimFunc
from tvm.tir.schedule import Schedule, LoopRV
from tvm.tir.schedule.schedule import BlockRV, ExprRV
from . import _ffi_api_schedule


@_register_object("tir.edgex.BufferAxisRV")
class BufferAxisRV(Object):
    """A random variable that refers to a buffer axis"""


@_register_object("tir.edgex.EdgexSchedule")
class EdgexSchedule(Schedule):
    """The schedule node for edgex TIR"""

    def __init__(  # pylint: disable=super-init-not-called
        self,
        func_or_mod: Union[PrimFunc, IRModule],
        debug_mode: Union[bool, int] = False,
    ):
        if isinstance(debug_mode, bool):
            if debug_mode:
                debug_mode = -1
            else:
                debug_mode = 0
        assert isinstance(debug_mode, int)
        self.__init_handle_by_constructor__(
            _ffi_api_schedule.EdgexSchedule,
            func_or_mod,
            -1,  # seed
            debug_mode,
        )

    def loop_partition(self, loops: Union[LoopRV, List[LoopRV]], lazy: bool = True):
        if isinstance(loops, LoopRV):
            loops = [loops]
        return _ffi_api_schedule.ScheduleLoopPartition(self, loops, lazy)

    def pragma(  # pylint: disable=arguments-differ
        self, target: Union[LoopRV, BlockRV], pragma_type: str, pragma_value
    ) -> None:
        if isinstance(pragma_value, int):
            pragma_value = tir.IntImm("int32", pragma_value)
        elif isinstance(pragma_value, float):
            pragma_value = tir.FloatImm("float32", pragma_value)
        elif isinstance(pragma_value, str):
            pragma_value = tir.StringImm(pragma_value)
        if isinstance(target, LoopRV):
            return _ffi_api_schedule.SchedulePragma(self, target, pragma_type, pragma_value)
        return self.annotate_block(target, "pragma_" + pragma_type, pragma_value)

    def annotate_block(self, block: BlockRV, attr_key: str, annotation):
        if isinstance(annotation, int):
            annotation = tir.IntImm("int32", annotation)
        elif isinstance(annotation, float):
            annotation = tir.FloatImm("float32", annotation)
        elif isinstance(annotation, str):
            annotation = tir.StringImm(annotation)
        return _ffi_api_schedule.ScheduleAnnotateBlock(self, block, attr_key, annotation)

    def get_read_buffer_axes(self, block: BlockRV, buffer_idx: int):
        return _ffi_api_schedule.ScheduleGetBlockReadBufferAxes(self, block, buffer_idx)

    def get_write_buffer_axes(self, block: BlockRV, buffer_idx: int):
        return _ffi_api_schedule.ScheduleGetBlockWriteBufferAxes(self, block, buffer_idx)

    def fuse_buffer(self, *axes: List[BufferAxisRV]) -> LoopRV:
        return _ffi_api_schedule.ScheduleFuseBuffer(self, axes)

    def split_buffer(
        self,
        buffer_axis: BufferAxisRV,
        *,
        nparts: Optional[ExprRV] = None,
        factor: Optional[ExprRV] = None,
        factors: Optional[List[ExprRV]] = None,
    ) -> List[BufferAxisRV]:
        if factors is not None:
            if (nparts is not None) or (factor is not None):
                raise ValueError("`nparts`/`factor` are not allowed when `factors` is specified")
        elif (nparts is None) and (factor is None):
            raise ValueError("None of the `nparts`, `factor` and `factors` are specified")
        elif (nparts is not None) and (factor is not None):
            raise ValueError("Only one of the `nparts`, `factor` are allowed to be specified")
        else:
            factors = [nparts, factor]
        return _ffi_api_schedule.ScheduleSplitBuffer(self, buffer_axis, factors)

    def reorder_buffer(self, *axes: List[BufferAxisRV]) -> None:
        _ffi_api_schedule.ScheduleReorderBuffer(self, axes)

    def stack_buffer(self, axis0: BufferAxisRV, axis1: BufferAxisRV) -> None:
        _ffi_api_schedule.ScheduleStackBuffer(self, axis0, axis1)

    def get_buffer_of(self, axis: BufferAxisRV) -> tir.Buffer:
        return _ffi_api_schedule.ScheduleGetBufferOf(self, axis)

    def replace_buffer(
        self,
        block: BlockRV,
        origin_buffer: tir.Buffer,
        new_buffer: tir.Buffer,
        load_rewrite: PackedFunc = None,
        store_rewrite: PackedFunc = None,
        region_rewrite: PackedFunc = None,
    ) -> None:
        _ffi_api_schedule.ScheduleReplaceBuffer(
            self, block, origin_buffer, new_buffer, load_rewrite, store_rewrite, region_rewrite
        )
