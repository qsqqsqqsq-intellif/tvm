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
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
from tvm.contrib.edgex.tir.transform import InjectDmaIntrin


# fmt: off
@T.prim_func
def edma_access_pattern(data: T.handle, result: T.handle) -> None:
    X = T.match_buffer(data, [1, 64, 224, 224], dtype="int32", scope="global")
    Y = T.match_buffer(result, [1, 64, 112, 112], dtype="int32", scope="global")
    dm = T.allocate([4096], dtype="int32", scope="dm")
    for c_o_o, h_o, w_o in T.grid(2, 14, 14):
        if h_o == 0 and w_o == 0:  # outer var should get fixed
            with T.attr("", "pragma_nnp_dma_scope", "eidma"):
                for c_o_i, h_i, w_i in T.grid(4, 17, 17):
                    if 1 <= 0*16 + h_i and 1 <= 0*16 + w_i:
                        for c_i in T.serial(0, 8):
                            dm[c_o_i*2312 + h_i*136 + w_i*8 + c_i] = T.load(
                                "int32", X.data,
                                c_o_o*1605632 + c_o_i*401408 + c_i*50176 + h_o*3584 + h_i*224 + w_o*16 + w_i - 225)
            with T.attr("", "pragma_nnp_dma_scope", "eodma"):
                for c_i, h_i, w_i in T.grid(32, 8, 8):
                    Y.data[c_o_o*401408 + c_i*12544 + h_o*896 + h_i*112 + w_o*8 + w_i] = T.load(
                        "int32", dm,
                        T.floordiv(c_i, 8)*512 + h_i*64 + w_i*8 + T.floormod(c_i, 8))

@T.prim_func
def edma_rewritten(data: T.handle, result: T.handle) -> None:
    X = T.match_buffer(data, [1, 64, 224, 224], dtype="int32")
    Y = T.match_buffer(result, [1, 64, 112, 112], dtype="int32")
    dm = T.allocate([4096], "int32", "dm")
    for c_o_o, h_o, w_o in T.grid(2, 14, 14):
        if ((h_o == 0) and (w_o == 0)):
            T.evaluate(T.nnp_eidma_load("int32",
                T.tvm_access_ptr(T.type_annotation(dtype="int32"), dm, 144, 9104, "w", dtype="handle"),
                T.tvm_access_ptr(T.type_annotation(dtype="int32"), X.data,
                    c_o_o*1605632 + h_o*3584 + w_o*16, 1558832, "r", dtype="handle"),
                "ei_start_addr_in_en=1", "ei_start_addr_out_en=1", "ei_first_state_en=1",
                "ei_state_num=1","ei_dtype=4", "ei_mode=2",
                "ei_j0_loop_num=4", "ei_j1_loop_num=16", "ei_j2_loop_num=16", "ei_j3_loop_num=8",
                "ei_j0_loop_sel=3", "ei_j1_loop_sel=1", "ei_j2_loop_sel=0", "ei_j3_loop_sel=2",
                "ei_j0_stridein=1605632", "ei_j1_stridein=200704", "ei_j2_stridein=896",
                "ei_j0_strideout=9248", "ei_j1_strideout=544", "ei_j2_strideout=32", dtype=""
            ))
            T.evaluate(T.nnp_eodma_store("int32",
                T.tvm_access_ptr(T.type_annotation(dtype="int32"), Y.data,
                    c_o_o*401408 + h_o*896 + w_o*8, 389656, "w", dtype="handle"),
                T.tvm_access_ptr(T.type_annotation(dtype="int32"), dm, 0, 2048, "r", dtype="handle"),
                "eo_start_addr_in_en=1", "eo_start_addr_out_en=1", "eo_first_state_en=1",
                "eo_state_num=3", "eo_dtype=4", "eo_mode=1",
                "eo_j0_loop_num=4", "eo_j1_loop_num=8", "eo_j2_loop_num=8", "eo_j3_loop_num=8",
                "eo_j0_loop_sel=3", "eo_j1_loop_sel=0", "eo_j2_loop_sel=2", "eo_j3_loop_sel=1",
                "eo_stride_in_j0=2048", "eo_stride_in_j1=256", "eo_stride_in_j2=32",
                "eo_j0_strideout=401408", "eo_j1_strideout=50176", "eo_j2_strideout=448", dtype=""
            ))


@T.prim_func
def eidma_1D_broadcast_pattern(data: T.handle) -> None:
    X = T.match_buffer(data, [16], dtype="int32", scope="global")
    dm = T.allocate([16], dtype="int32", scope="dm")
    with T.attr("", "pragma_nnp_dma_scope", "eidma"):
        for i in T.serial(0, 16):
            T.store(dm, i, T.load("int32", X.data, 0))

@T.prim_func
def eidma_1D_broadcast_pattern_rewritten(data: T.handle) -> None:
    X = T.match_buffer(data, [16], dtype="int32")
    dm = T.allocate([16], "int32", "dm")
    T.evaluate(T.nnp_eidma_load("int32",
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), dm, 0, 16, "w", dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), X.data, 0, 1, "r", dtype="handle"),
        "ei_start_addr_in_en=1", "ei_start_addr_out_en=1", "ei_first_state_en=1", "ei_state_num=1",
        "ei_dtype=4", "ei_mode=2", "ei_j0_loop_num=1", "ei_j1_loop_num=1", "ei_j2_loop_num=16", "ei_j3_loop_num=1",
        "ei_j0_loop_sel=3", "ei_j1_loop_sel=2", "ei_j2_loop_sel=1", "ei_j3_loop_sel=0",
        "ei_j0_stridein=0", "ei_j1_stridein=0", "ei_j2_stridein=0",
        "ei_j0_strideout=64", "ei_j1_strideout=64", "ei_j2_strideout=4", dtype=""))


@T.prim_func
def eidma_multi_branch_under_scope(data: T.handle) -> None:
    X = T.match_buffer(data, [256], dtype="int32", scope="global")
    dm = T.allocate([256], dtype="int32", scope="dm")
    with T.attr("", "pragma_nnp_dma_scope", "eidma"):
        # i in [0, 4)
        for i in range(4):
            # j in [0, 4)
            for j in range(4):
                T.store(dm, i * 16 + j, T.load("int32", X.data, i * 16 + j))
            # j in [4, 16)
            for j in range(12):
                T.store(dm, i * 16 + j + 4, T.load("int32", X.data, i * 16 + j + 4))
        # i in [4, 16)
        for i in T.serial(0, 12):
            # j == 0
            T.store(dm, i * 16 + 64, T.load("int32", X.data, i * 16 + 64))
            # j in [1, 15)
            for j in range(14):
                T.store(dm, i * 16 + j + 65, T.load("int32", X.data, i * 16 + j + 65))
            # j == 15
            T.store(dm, i * 16 + 79, T.load("int32", X.data, i * 16 + 79))

@T.prim_func
def eidma_multi_branch_under_scope_rewritten(X: T.Buffer[(256,), "int32"]) -> None:
    dm = T.allocate([256], "int32", "dm")
    T.evaluate(T.nnp_eidma_load("int32",
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), dm, 0, 52, "w", dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), X.data, 0, 52, "r", dtype="handle"),
        "ei_start_addr_in_en=1", "ei_start_addr_out_en=1", "ei_first_state_en=1", "ei_state_num=1", "ei_dtype=4", "ei_mode=2",
        "ei_j0_loop_num=1", "ei_j1_loop_num=1", "ei_j2_loop_num=4", "ei_j3_loop_num=4",
        "ei_j0_loop_sel=3", "ei_j1_loop_sel=2", "ei_j2_loop_sel=1", "ei_j3_loop_sel=0",
        "ei_j0_stridein=256", "ei_j1_stridein=256", "ei_j2_stridein=64",
        "ei_j0_strideout=256", "ei_j1_strideout=256", "ei_j2_strideout=64", dtype=""))
    T.evaluate(T.nnp_eidma_load("int32",
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), dm, 4, 60, "w", dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), X.data, 4, 60, "r", dtype="handle"),
        "ei_start_addr_in_en=1", "ei_start_addr_out_en=1", "ei_first_state_en=1", "ei_state_num=1", "ei_dtype=4", "ei_mode=2",
        "ei_j0_loop_num=1", "ei_j1_loop_num=1", "ei_j2_loop_num=4", "ei_j3_loop_num=12",
        "ei_j0_loop_sel=3", "ei_j1_loop_sel=2", "ei_j2_loop_sel=1", "ei_j3_loop_sel=0",
        "ei_j0_stridein=256", "ei_j1_stridein=256", "ei_j2_stridein=64",
        "ei_j0_strideout=256", "ei_j1_strideout=256", "ei_j2_strideout=64", dtype=""))
    T.evaluate(T.nnp_eidma_load("int32",
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), dm, 64, 177, "w", dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), X.data, 64, 177, "r", dtype="handle"),
        "ei_start_addr_in_en=1", "ei_start_addr_out_en=1", "ei_first_state_en=1", "ei_state_num=1", "ei_dtype=4", "ei_mode=2",
        "ei_j0_loop_num=1", "ei_j1_loop_num=1", "ei_j2_loop_num=12", "ei_j3_loop_num=1",
        "ei_j0_loop_sel=3", "ei_j1_loop_sel=2", "ei_j2_loop_sel=1", "ei_j3_loop_sel=0",
        "ei_j0_stridein=768", "ei_j1_stridein=768", "ei_j2_stridein=64",
        "ei_j0_strideout=768", "ei_j1_strideout=768", "ei_j2_strideout=64", dtype=""))
    T.evaluate(T.nnp_eidma_load("int32",
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), dm, 65, 190, "w", dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), X.data, 65, 190, "r", dtype="handle"),
        "ei_start_addr_in_en=1", "ei_start_addr_out_en=1", "ei_first_state_en=1", "ei_state_num=1", "ei_dtype=4", "ei_mode=2",
        "ei_j0_loop_num=1", "ei_j1_loop_num=1", "ei_j2_loop_num=12", "ei_j3_loop_num=14",
        "ei_j0_loop_sel=3", "ei_j1_loop_sel=2", "ei_j2_loop_sel=1", "ei_j3_loop_sel=0",
        "ei_j0_stridein=768", "ei_j1_stridein=768", "ei_j2_stridein=64",
        "ei_j0_strideout=768", "ei_j1_strideout=768", "ei_j2_strideout=64", dtype=""))
    T.evaluate(T.nnp_eidma_load("int32",
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), dm, 79, 177, "w", dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), X.data, 79, 177, "r", dtype="handle"),
        "ei_start_addr_in_en=1", "ei_start_addr_out_en=1", "ei_first_state_en=1", "ei_state_num=1", "ei_dtype=4", "ei_mode=2",
        "ei_j0_loop_num=1", "ei_j1_loop_num=1", "ei_j2_loop_num=12", "ei_j3_loop_num=1",
        "ei_j0_loop_sel=3", "ei_j1_loop_sel=2", "ei_j2_loop_sel=1", "ei_j3_loop_sel=0",
        "ei_j0_stridein=768", "ei_j1_stridein=768", "ei_j2_stridein=64",
        "ei_j0_strideout=768", "ei_j1_strideout=768", "ei_j2_strideout=64", dtype=""))

# fmt: on


def do_inject_test(func, expect, verbose=False):
    mod = IRModule.from_expr(func)
    config = {"tir.edgex.InjectDmaIntrin.verbose": verbose}
    with tvm.transform.PassContext(config=config):
        mod = InjectDmaIntrin()(mod)
        mod = tvm.tir.transform.RemoveNoOp()(mod)  # flatten seq
    tvm.ir.assert_structural_equal(mod["main"], expect, True)


def test_rewrite_eidma():
    do_inject_test(edma_access_pattern, edma_rewritten)


def test_eidma_1D_broadcast_pattern():
    do_inject_test(eidma_1D_broadcast_pattern, eidma_1D_broadcast_pattern_rewritten)


def test_eidma_multi_branch_under_scope():
    do_inject_test(eidma_multi_branch_under_scope, eidma_multi_branch_under_scope_rewritten)


if __name__ == "__main__":
    test_rewrite_eidma()
    test_eidma_1D_broadcast_pattern()
    test_eidma_multi_branch_under_scope()
