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
"""Tiling analysis utilities"""
from typing import Dict, List, Union
import tvm
from tvm import tir
from tvm.ir.expr import PrimExpr
from tvm.tir.schedule import BlockRV, StmtSRef
from . import _ffi_api


@tvm._ffi.register_object("edgex.uniform_schedule.TileRelation")
class TileRelation(tvm.runtime.Object):
    def get_spec(self, write_axis_idx, read_axis_idx):
        return _ffi_api.TileRelationGetSpec(self, write_axis_idx, read_axis_idx)


def get_tile_relations(s, block: Union[BlockRV, StmtSRef]) -> Dict[tir.Buffer, List[TileRelation]]:
    if isinstance(block, BlockRV):
        return _ffi_api.GetTileRelationsByRV(s, block)
    return _ffi_api.GetTileRelations(s, block)


def estimate_tile_shape(
    buffer: tir.Buffer, relations: List[TileRelation], placeholders: List[PrimExpr]
) -> List[PrimExpr]:
    return _ffi_api.EstimateTileSizes(buffer, relations, placeholders)


def compose_tile_relations(
    rel1: List[TileRelation], rel2: List[TileRelation]
) -> List[TileRelation]:
    return _ffi_api.ComposeTileRelations(rel1, rel2)


def get_root_tile_relations(buffer: tir.Buffer) -> List[TileRelation]:
    return _ffi_api.GetRootTileRelations(buffer)


def get_irrelevant_tile_relations(buffer: tir.Buffer) -> List[TileRelation]:
    return _ffi_api.GetIrrelevantTileRelations(buffer)


def get_expr_repr(expr: tir.PrimExpr) -> str:
    return _ffi_api.GetExprRepr(expr)
