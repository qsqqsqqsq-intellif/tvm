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
"""inject calculated isa tir pass tests."""

import tvm
from tvm import tir
from tvm.script import tir as T
from tvm.contrib.edgex.tir.transform import *


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func)
    mod = InjectCalculatedIsa()(mod)
    tvm.ir.assert_structural_equal(mod["main"], transformed, True)


@T.prim_func
def without_inject_isa_func(a: T.handle, c: T.handle) -> None:
    X_dm = T.alloc_buffer(
        [1, 3, 224, 224], dtype="uint8", elem_offset=0, scope="dm", align=128, offset_factor=1
    )
    W_dm = T.alloc_buffer(
        [16, 3, 7, 7],
        dtype="int8",
        elem_offset=0,
        scope="dm",
        align=128,
        offset_factor=1,
    )
    X_dm_iobuf = T.alloc_buffer(
        [1, 3, 224, 224],
        dtype="int8",
        elem_offset=0,
        scope="iobuf",
        align=128,
        offset_factor=1,
    )
    W_dm_wbuf = T.alloc_buffer(
        [16, 3, 7, 7],
        dtype="int8",
        elem_offset=0,
        scope="wbuf",
        align=128,
        offset_factor=1,
    )
    Y_iobuf = T.alloc_buffer(
        [1, 16, 112, 112],
        dtype="int32",
        elem_offset=0,
        scope="cube",
        align=128,
        offset_factor=1,
    )
    Y_dm = T.alloc_buffer(
        [1, 16, 112, 112], dtype="int8", elem_offset=0, scope="dm", align=128, offset_factor=1
    )
    T.attr(Y_dm, "pragma_nnp_num_co", 16)
    T.evaluate(
        T.nnp_idma_load(
            "uint8",
            T.tvm_access_ptr(
                T.type_annotation(dtype="uint8"),
                X_dm_iobuf.data,
                0,
                3 * 224 * 224,
                "w",
                dtype="handle",
            ),
            T.tvm_access_ptr(
                T.type_annotation(dtype="uint8"),
                X_dm.data,
                0,
                3 * 224 * 224,
                "r",
                dtype="handle",
            ),
            "sparsity_en_idma=0",
            "num_ci_group_idma=3",
            "op_idma=0",
            "wino_en_idma=0",
            "para_mode_idma=0",
            "co_w_idma=112",
            "co_h_idma=112",
            "co_d_idma=1",
            "cube_enable_idma=2",
            "B_dim2_idma=0",
            "data_type_idma=1",
            dtype="",
        )
    )
    T.evaluate(
        T.nnp_wdma_load(
            "int8",
            T.tvm_access_ptr(
                T.type_annotation(dtype="int8"),
                W_dm_wbuf.data,
                0,
                16 * 3 * 7 * 7,
                "w",
                dtype="handle",
            ),
            T.tvm_access_ptr(
                T.type_annotation(dtype="int8"),
                W_dm.data,
                0,
                16 * 3 * 7 * 7,
                "r",
                dtype="handle",
            ),
            "A_dim1_wdma=0",
            "A_dim2_wdma=0",
            "k_size_wdma=49",
            "A_transpose_wdma=0",
            "data_type_wdma=0",
            "bubble_insert_en_wdma=0",
            "wt_st_addr1_wdma=0x0",
            "wt_end_addr1_wdma=0x30ff",
            dtype="",
        )
    )
    T.evaluate(
        T.nnp_cube(
            "int32",
            T.tvm_access_ptr(
                T.type_annotation(dtype="int32"),
                Y_iobuf.data,
                0,
                16 * 112 * 112,
                "w",
                dtype="handle",
            ),
            T.tvm_access_ptr(
                T.type_annotation(dtype="uint8"),
                X_dm_iobuf.data,
                0,
                3 * 224 * 224,
                "r",
                dtype="handle",
            ),
            T.tvm_access_ptr(
                T.type_annotation(dtype="int8"),
                W_dm_wbuf.data,
                0,
                16 * 3 * 7 * 7,
                "r",
                dtype="handle",
            ),
            dtype="",
        )
    )

    T.evaluate(
        T.nnp_odma_store(
            "int8",
            T.tvm_access_ptr(
                T.type_annotation(dtype="int8"),
                Y_dm.data,
                0,
                16 * 112 * 112,
                "w",
                dtype="handle",
            ),
            T.tvm_access_ptr(
                T.type_annotation(dtype="int8"),
                Y_iobuf.data,
                0,
                16 * 112 * 112,
                "r",
                dtype="handle",
            ),
            "extract_2to1_odma=0",
            "num_group_odma=1",
            "data_type_odma=4",
            "psum_out_en_odma=1",
            "int_type_odma=0",
            "co_w_odma=112",
            "co_ch_offset_odma=802816",
            dtype="",
        )
    )


@T.prim_func
def inject_isa_transformed(a: T.handle, c: T.handle) -> None:
    X_dm = T.alloc_buffer(
        [1, 3, 224, 224], dtype="uint8", elem_offset=0, scope="dm", align=128, offset_factor=1
    )
    W_dm = T.alloc_buffer(
        [16, 3, 7, 7],
        dtype="int8",
        elem_offset=0,
        scope="dm",
        align=128,
        offset_factor=1,
    )
    X_dm_iobuf = T.alloc_buffer(
        [1, 3, 224, 224],
        dtype="int8",
        elem_offset=0,
        scope="iobuf",
        align=128,
        offset_factor=1,
    )
    W_dm_wbuf = T.alloc_buffer(
        [16, 3, 7, 7],
        dtype="int8",
        elem_offset=0,
        scope="wbuf",
        align=128,
        offset_factor=1,
    )
    Y_iobuf = T.alloc_buffer(
        [1, 16, 112, 112],
        dtype="int32",
        elem_offset=0,
        scope="cube",
        align=128,
        offset_factor=1,
    )
    Y_dm = T.alloc_buffer(
        [1, 16, 112, 112], dtype="int8", elem_offset=0, scope="dm", align=128, offset_factor=1
    )
    T.evaluate(
        T.nnp_idma_load(
            "uint8",
            T.tvm_access_ptr(
                T.type_annotation(dtype="uint8"),
                X_dm_iobuf.data,
                0,
                3 * 224 * 224,
                "w",
                dtype="handle",
            ),
            T.tvm_access_ptr(
                T.type_annotation(dtype="uint8"),
                X_dm.data,
                0,
                3 * 224 * 224,
                "r",
                dtype="handle",
            ),
            "sparsity_en_idma=0",
            "num_ci_group_idma=3",
            "op_idma=0",
            "wino_en_idma=0",
            "para_mode_idma=0",
            "co_w_idma=112",
            "co_h_idma=112",
            "co_d_idma=1",
            "cube_enable_idma=2",
            "B_dim2_idma=0",
            "data_type_idma=1",
            "epsilon_idma=0x31",
            "delta_idma=0x1",
            "zeta_idma=0x10",
            "dense_idma=0x1",
            "epsilon_times_idma=0x1",
            "delta_times_idma=0x1",
            "zeta_times_idma=0x11",
            "dense_times_idma=0x1",
            "last_epsilon_idma=0x31",
            "last_delta_idma=0x1",
            "last_zeta_idma=0x6",
            "last_dense_idma=0x1",
            "last_zeta_width_idma=0x10",
            "eps_ci_times_idma=0x1",
            "last_eps_ci_times_idma=0x1",
            "ub_ci_num_idma=0x10",
            "wo_ci_num_idma=0x31",
            "wo_d_num_idma=0x1",
            "wo_h_num_idma=0x10",
            dtype="",
        )
    )
    T.evaluate(
        T.nnp_wdma_load(
            "int8",
            T.tvm_access_ptr(
                T.type_annotation(dtype="int8"),
                W_dm_wbuf.data,
                0,
                16 * 3 * 7 * 7,
                "w",
                dtype="handle",
            ),
            T.tvm_access_ptr(
                T.type_annotation(dtype="int8"),
                W_dm.data,
                0,
                16 * 3 * 7 * 7,
                "r",
                dtype="handle",
            ),
            "A_dim1_wdma=0",
            "A_dim2_wdma=0",
            "k_size_wdma=49",
            "A_transpose_wdma=0",
            "data_type_wdma=0",
            "bubble_insert_en_wdma=0",
            "wt_st_addr1_wdma=0x0",
            "wt_end_addr1_wdma=0x30ff",
            "epsilon_wdma=0x31",
            "delta_wdma=0x1",
            "zeta_wdma=0x10",
            "dense_wdma=0x1",
            "epsilon_times_wdma=0x1",
            "delta_times_wdma=0x1",
            "zeta_times_wdma=0x11",
            "dense_times_wdma=0x1",
            "last_epsilon_wdma=0x31",
            "last_delta_wdma=0x1",
            "last_zeta_wdma=0x6",
            "last_dense_wdma=0x1",
            "epsilon_times_rewrite_dm_wdma=0x0",
            "delta_inc_addr_wdma=0x0",
            "ksize_inc_addr_wdma=0x6100",
            "epstimes_inc_addr_wdma=0x3100",
            "delta_times_inc_addr_wdma=0x0",
            "mat_row_offset_wdma=0x0",
            dtype="",
        )
    )

    T.evaluate(
        T.nnp_cube(
            "int32",
            T.tvm_access_ptr(
                T.type_annotation(dtype="int32"),
                Y_iobuf.data,
                0,
                16 * 112 * 112,
                "w",
                dtype="handle",
            ),
            T.tvm_access_ptr(
                T.type_annotation(dtype="uint8"),
                X_dm_iobuf.data,
                0,
                3 * 224 * 224,
                "r",
                dtype="handle",
            ),
            T.tvm_access_ptr(
                T.type_annotation(dtype="int8"),
                W_dm_wbuf.data,
                0,
                16 * 3 * 7 * 7,
                "r",
                dtype="handle",
            ),
            "epsilon_cube=0x31",
            "delta_cube=0x1",
            "zeta_cube=0x10",
            "dense_cube=0x1",
            "epsilon_times_cube=0x1",
            "delta_times_cube=0x1",
            "zeta_times_cube=0x11",
            "dense_times_cube=0x1",
            "last_epsilon_cube=0x31",
            "last_delta_cube=0x1",
            "last_zeta_cube=0x6",
            "last_dense_cube=0x1",
            "last_beta_remind_cube=0x3",
            "burst_size_pipe_num_cube=0x3226",
            dtype="",
        )
    )

    T.evaluate(
        T.nnp_odma_store(
            "int8",
            T.tvm_access_ptr(
                T.type_annotation(dtype="int8"),
                Y_dm.data,
                0,
                16 * 112 * 112,
                "w",
                dtype="handle",
            ),
            T.tvm_access_ptr(
                T.type_annotation(dtype="int8"),
                Y_iobuf.data,
                0,
                16 * 112 * 112,
                "r",
                dtype="handle",
            ),
            "extract_2to1_odma=0",
            "num_group_odma=1",
            "data_type_odma=4",
            "psum_out_en_odma=1",
            "int_type_odma=0",
            "co_w_odma=112",
            "co_ch_offset_odma=802816",
            "delta_odma=0x1",
            "zeta_odma=0x10",
            "dense_odma=0x1",
            "delta_times_odma=0x1",
            "zeta_times_odma=0x11",
            "dense_times_odma=0x1",
            "last_delta_odma=0x1",
            "last_zeta_odma=0x6",
            "last_dense_odma=0x1",
            "last_zeta_width_odma=0x10",
            "last_delta_co_odma=0x10",
            "zeta_offset_odma=0xc0",
            "wino_zeta_add_en_odma=0x0",
            "init_xbar_wr_byte_odma=0x7f",
            "last_xbar_co_times_odma=0x0",
            "last_xbar_pixel_times_odma=0x7",
            "last_xbar_wr_byte_odma=0x7f",
            "last_xbar_cube_times_odma=0x0",
            "delta_times_transfer_co_odma=0x10",
            "delta_ch_offset_odma=0xc4000",
            dtype="",
        )
    )


def test_inject_isa_transformed():
    _check(without_inject_isa_func, inject_isa_transformed)


if __name__ == "__main__":
    test_inject_isa_transformed()
