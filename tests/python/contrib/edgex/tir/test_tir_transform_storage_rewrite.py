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
from tvm.script import tir as T
from tvm.contrib.edgex.tir.transform import StorageRewriteNNP400


# fmt: off
@T.prim_func
def vu_example(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128], dtype="int32")
    B = T.match_buffer(b, [128], dtype="int32")
    C = T.match_buffer(c, [128], dtype="int32")
    A_dm = T.allocate([128], "int32", "dm")
    B_dm = T.allocate([128], "int32", "dm")
    C_dm = T.allocate([128], "int32", "dm")
    A_vm = T.allocate([64], "int32", "vm")
    B_vm = T.allocate([64], "int32", "vm")
    C_vm = T.allocate([64], "int32", "vm")
    T.evaluate(T.nnp_eidma_load(T.type_annotation(dtype="int32"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), A_dm.data, 0, 128, 2, dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), A.data, 0, 128, 1, dtype="handle"),
        dtype="handle"))
    T.evaluate(T.nnp_eidma_load(T.type_annotation(dtype="int32"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), B_dm.data, 0, 128, 2, dtype="handle"),
        T.tvm_access_ptr( T.type_annotation(dtype="int32"), B.data, 0, 128, 1, dtype="handle"),
        dtype="handle"))
    for i in T.serial(2, 1):
        T.evaluate(T.nnp_vidma_load(T.type_annotation(dtype="int32"),
            T.tvm_access_ptr(T.type_annotation(dtype="int32"), A_vm.data, 0, 64, 2, dtype="handle"),
            T.tvm_access_ptr(T.type_annotation(dtype="int32"), A_dm.data, i * 64, 64, 1, dtype="handle"),
            dtype="handle"))
        T.evaluate(T.nnp_vidma_load(T.type_annotation(dtype="int32"),
            T.tvm_access_ptr(T.type_annotation(dtype="int32"), B_vm.data, 0, 64, 2, dtype="handle"),
            T.tvm_access_ptr(T.type_annotation(dtype="int32"), B_dm.data, i * 64, 64, 1, dtype="handle"),
            dtype="handle"))
        for j in T.serial(64, 1):
            C_vm[j] = A_vm[j] + B_vm[j]
        T.evaluate(T.nnp_vodma_store(T.type_annotation(dtype="int32"),
            T.tvm_access_ptr(T.type_annotation(dtype="int32"), C_dm.data, i * 64, 64, 2, dtype="handle"),
            T.tvm_access_ptr(T.type_annotation(dtype="int32"), C_vm.data, 0, 64, 1, dtype="handle"),
            dtype="handle"))
    T.evaluate(T.nnp_eodma_store(T.type_annotation(dtype="int32"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), C.data, 0, 128, 2, dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), C_dm.data, 0, 128, 1, dtype="handle"),
        dtype="handle"))


@T.prim_func
def vu_with_static_dma_offset(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128], dtype="int32")
    B = T.match_buffer(b, [128], dtype="int32")
    C = T.match_buffer(c, [128], dtype="int32")
    A_dm = T.allocate([384], "int32", "dm")
    A_vm = T.allocate([192], "int32", "vm")
    C_vm = T.buffer_decl([64], dtype="int32", data=A_vm.data, elem_offset=128, scope="vm")
    A_vm_1 = T.buffer_decl([64], dtype="int32", data=A_vm.data, scope="vm")
    B_vm = T.buffer_decl([64], dtype="int32", data=A_vm.data, elem_offset=64, scope="vm")
    T.evaluate(T.nnp_eidma_load(T.type_annotation(dtype="int32"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), A_dm.data, 0, 128, 2, dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), A.data, 0, 128, 1, dtype="handle"),
        "ei_start_addr1=0x0", "ei_end_addr1=0x1ff", "ei_start_addr2=0x0", "ei_end_addr2=0x1ff", dtype="handle"))
    T.evaluate(T.nnp_eidma_load(T.type_annotation(dtype="int32"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), A_dm.data, 128, 128, 2, dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), B.data, 0, 128, 1, dtype="handle"),
        "ei_start_addr1=0x200", "ei_end_addr1=0x3ff", "ei_start_addr2=0x200", "ei_end_addr2=0x3ff", dtype="handle"))
    for i in T.serial(2, 1):
        T.evaluate(T.nnp_vidma_load(T.type_annotation(dtype="int32"),
            T.tvm_access_ptr(T.type_annotation(dtype="int32"), A_vm.data, 0, 64, 2, dtype="handle"),
            T.tvm_access_ptr(T.type_annotation(dtype="int32"), A_dm.data, (i*64), 64, 1, dtype="handle"),
            "cb_buf_start_addr_vm_vidma=0x0", "cb_buf_end_addr_vm_vidma=0xff", dtype="handle"))
        T.evaluate(T.nnp_vidma_load(T.type_annotation(dtype="int32"),
            T.tvm_access_ptr(T.type_annotation(dtype="int32"), A_vm.data, 64, 64, 2, dtype="handle"),
            T.tvm_access_ptr(T.type_annotation(dtype="int32"), A_dm.data, (128 + (i*64)), 64, 1, dtype="handle"),
            "cb_buf_start_addr_vm_vidma=0x100", "cb_buf_end_addr_vm_vidma=0x1ff", dtype="handle"))
        for j in T.serial(64, 1):
            C_vm[j] = A_vm_1[j] + B_vm[j]
        T.evaluate(T.nnp_vodma_store(T.type_annotation(dtype="int32"),
            T.tvm_access_ptr(T.type_annotation(dtype="int32"), A_dm.data, (256 + (i*64)), 64, 2, dtype="handle"),
            T.tvm_access_ptr(T.type_annotation(dtype="int32"), A_vm.data, 128, 64, 1, dtype="handle"),
            "cb_buf_start_addr_vm_vodma=0x200", "cb_buf_end_addr_vm_vodma=0x2ff", dtype="handle"))
    T.evaluate(T.nnp_eodma_store(T.type_annotation(dtype="int32"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), C.data, 0, 128, 2, dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), A_dm.data, 256, 128, 1, dtype="handle"),
        "eo_start_addr1=0x400", "eo_end_addr1=0x5ff", "eo_start_addr2=0x400", "eo_end_addr2=0x5ff", dtype="handle"))


@T.prim_func
def multiple_dm_buffer_with_alignment(a: T.handle) -> None:
    A = T.match_buffer(a, [128], dtype="int32")
    DM1 = T.allocate([256], "int32", "dm")
    T.attr(DM1.data, "storage_alignment", 16)
    DM2 = T.allocate([256], "int32", "dm")
    T.attr(DM2.data, "storage_alignment", 80)
    DM3 = T.allocate([256], "int32", "dm")
    T.attr(DM3.data, "storage_alignment", 2048)
    T.evaluate(T.nnp_eidma_load(T.type_annotation(dtype="int32"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), DM1.data, 0, 256, 2, dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), A.data, 0, 256, 1, dtype="handle"),
        dtype="handle"))
    T.evaluate(T.nnp_eidma_load(T.type_annotation(dtype="int32"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), DM2.data, 0, 256, 2, dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), A.data, 0, 256, 1, dtype="handle"),
        dtype="handle"))
    T.evaluate(T.nnp_eidma_load(T.type_annotation(dtype="int32"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), DM3.data, 0, 256, 2, dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), A.data, 0, 256, 1, dtype="handle"),
        dtype="handle"))
    # simulate use buffer
    DM3[0] = DM1[0] + DM2[0]
    

@T.prim_func
def multiple_dm_buffer_with_alignment_expect(a: T.handle) -> None:
    A = T.match_buffer(a, [128], dtype="int32")
    DM1 = T.allocate([1280], "int32", "dm")
    DM3 = T.buffer_decl([256], dtype="int32", data=DM1.data, elem_offset=1024, scope="dm")
    DM1_1 = T.buffer_decl([256], dtype="int32", data=DM1.data, scope="dm")
    DM2 = T.buffer_decl([256], dtype="int32", data=DM1.data, elem_offset=260, scope="dm")
    T.evaluate(T.nnp_eidma_load(T.type_annotation(dtype="int32"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), DM1.data, 0, 256, 2, dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), A.data, 0, 256, 1, dtype="handle"),
        "ei_start_addr1=0x0", "ei_end_addr1=0x3ff", "ei_start_addr2=0x0", "ei_end_addr2=0x3ff", dtype="handle"))
    T.evaluate(T.nnp_eidma_load(T.type_annotation(dtype="int32"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), DM1.data, 260, 256, 2, dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), A.data, 0, 256, 1, dtype="handle"),
        "ei_start_addr1=0x410", "ei_end_addr1=0x80f", "ei_start_addr2=0x410", "ei_end_addr2=0x80f", dtype="handle"))
    T.evaluate(T.nnp_eidma_load(T.type_annotation(dtype="int32"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), DM1.data, 1024, 256, 2, dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), A.data, 0, 256, 1, dtype="handle"),
        "ei_start_addr1=0x1000", "ei_end_addr1=0x13ff", "ei_start_addr2=0x1000", "ei_end_addr2=0x13ff", dtype="handle"))
    # simulate use buffer
    DM3[0] = DM1_1[0] + DM2[0]  


@T.prim_func
def simple_cube_dma() -> None:
    in_dm = T.allocate([1, 16, 224, 224], dtype="int8", scope="dm")
    in_buf = T.allocate([1, 16, 224, 224], dtype="int8", scope="iobuf")
    out_dm = T.allocate([1, 64, 112, 112], dtype="int8", scope="dm")
    out_buf = T.allocate([1, 64, 112, 112], dtype="int8", scope="iobuf")
    weight_dm = T.allocate([64, 16, 7, 7], dtype="int8", scope="dm")
    weight_buf = T.allocate([64, 16, 7, 7], dtype="int8", scope="wbuf")
    bias_dm = T.allocate([64], dtype="int32", scope="dm")
    bias_buf = T.allocate([64], dtype="int32", scope="bbuf")
    T.evaluate(T.nnp_idma_load(T.type_annotation(dtype="int32"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), in_buf.data, 0, 16*224*224, 2, dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), in_dm.data, 0, 16*224*224, 1, dtype="handle"), dtype=""))
    T.evaluate(T.nnp_wdma_load(T.type_annotation(dtype="int32"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), weight_buf.data, 0, 16*64*7*7, 2, dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), weight_dm.data, 0, 16*64*7*7, 1, dtype="handle"), dtype=""))
    T.evaluate(T.nnp_bdma_load(T.type_annotation(dtype="int32"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), bias_buf.data, 0, 64, 2, dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), bias_dm.data, 0, 64, 1, dtype="handle"), dtype=""))
    T.evaluate(T.nnp_cube(dtype=""))
    T.evaluate(T.nnp_odma_store(T.type_annotation(dtype="int32"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), out_dm.data, 0, 64*112*112, 2, dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), out_buf.data, 0, 64*112*112, 1, dtype="handle"), dtype=""))


@T.prim_func
def simple_cube_dma_expect() -> None:
    in_buf = T.allocate([802816], dtype="int8", scope="iobuf")
    in_dm = T.allocate([1656064],  dtype="int8", scope="dm")
    weight_buf = T.allocate([50176],  dtype="int8", scope="wbuf")
    bias_buf = T.allocate([64],  dtype="int32", scope="bbuf")
    out_buf = T.allocate([802816],  dtype="int8", scope="iobuf")
    T.evaluate(T.nnp_idma_load(T.type_annotation(dtype="int32"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), in_buf.data, 0, 16*224*224, 2, dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), in_dm.data, 0, 16*224*224, 1, dtype="handle"),
        "feat_st_addr1_idma=0x0", "feat_end_addr1_idma=0x30ffff",
        "feat_st_addr2_idma=0x0", "feat_end_addr2_idma=0x30ffff", dtype=""))
    T.evaluate(T.nnp_wdma_load(T.type_annotation(dtype="int32"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), weight_buf.data, 0, 16*64*7*7, 2, dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), in_dm.data, 200704, 16*64*7*7, 1, dtype="handle"),
        "wt_st_addr1_wdma=0xc4000", "wt_end_addr1_wdma=0xf4fff",
        "wt_st_addr2_wdma=0xc4000", "wt_end_addr2_wdma=0xf4fff", dtype=""))
    T.evaluate(T.nnp_bdma_load(T.type_annotation(dtype="int32"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), bias_buf.data, 0, 64, 2, dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), in_dm.data, 213248, 64, 1, dtype="handle"),
        "st_addr1_bdma=0xd0400", "end_addr1_bdma=0xd04ff",
        "st_addr2_bdma=0xd0400", "end_addr2_bdma=0xd04ff", dtype=""))
    T.evaluate(T.nnp_cube(dtype=""))
    T.evaluate(T.nnp_odma_store(T.type_annotation(dtype="int32"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), in_dm.data, 213312, 64*112*112, 2, dtype="handle"),
        T.tvm_access_ptr(T.type_annotation(dtype="int32"), out_buf.data, 0, 64*112*112, 1, dtype="handle"),
        "rslt_st_addr1_odma=0xd0500", "rslt_end_addr1_odma=0x3e04ff",
        "rslt_st_addr2_odma=0xd0500", "rslt_end_addr2_odma=0x3e04ff", dtype=""))


@T.prim_func
def mixed_datatype() -> None:
    vm0 = T.allocate([1024], dtype="int8", scope="vm")
    vm1 = T.allocate([256], dtype="int32", scope="vm")
    vm2 = T.allocate([512], dtype="int16", scope="vm")
    vm3 = T.allocate([256], dtype="float32", scope="vm")
    T.evaluate(vm0[0])
    T.evaluate(vm1[0])
    T.evaluate(vm2[0])
    T.evaluate(vm3[0])
    T.evaluate(vm0[0])
    T.evaluate(vm1[0])
    T.evaluate(vm2[0])
    T.evaluate(vm3[0])


@T.prim_func
def mixed_datatype_expect() -> None:
    vm0 = T.allocate([4096], "int8", "vm")
    vm0_1 = T.buffer_decl([1024], dtype="int8", data=vm0.data, scope="vm")
    vm1 = T.buffer_decl([256], dtype="int32", data=vm0.data, elem_offset=256, scope="vm")
    vm2 = T.buffer_decl([512], dtype="int16", data=vm0.data, elem_offset=1024, scope="vm")
    vm3 = T.buffer_decl([256], dtype="float32", data=vm0.data, elem_offset=768, scope="vm")
    T.evaluate(vm0_1[0])
    T.evaluate(vm1[0])
    T.evaluate(vm2[0])
    T.evaluate(vm3[0])
    T.evaluate(vm0_1[0])
    T.evaluate(vm1[0])
    T.evaluate(vm2[0])
    T.evaluate(vm3[0])
# fmt: on


def do_test_storage_rewrite(func, expect):
    mod = tvm.IRModule.from_expr(func)
    mod = StorageRewriteNNP400()(mod)
    tvm.ir.assert_structural_equal(mod["main"], expect, True)


def test_storage_rewrite_vu_simple():
    do_test_storage_rewrite(vu_example, vu_with_static_dma_offset)


def test_alignment_requirement():
    do_test_storage_rewrite(
        multiple_dm_buffer_with_alignment, multiple_dm_buffer_with_alignment_expect
    )


def test_cube_dma_lifetime():
    do_test_storage_rewrite(simple_cube_dma, simple_cube_dma_expect)


def test_mixed_datatype():
    do_test_storage_rewrite(mixed_datatype, mixed_datatype_expect)


if __name__ == "__main__":
    test_storage_rewrite_vu_simple()
    test_alignment_requirement()
    test_cube_dma_lifetime()
    test_mixed_datatype()
