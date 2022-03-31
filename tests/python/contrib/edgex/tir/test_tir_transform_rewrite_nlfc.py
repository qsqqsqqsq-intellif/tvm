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
from tvm.contrib.edgex.tir.transform import RewriteNlfc


@T.prim_func
def vu_nlfc_sigmoid_example() -> None:
    nlfc_table = T.allocate([1024], "int8", "dm")
    X = T.allocate([256], "float16", "vm")
    Y = T.allocate([256], "float16", "vm")
    for i in range(8):
        Y[T.ramp(i * 32, 1, 32)] = T.nnp_nlfc_sigmoid(
            X[T.ramp(i * 32, 1, 32)], nlfc_table.data, dtype="float16x32"
        )


@T.prim_func
def vu_nlfc_sigmoid_example_rewritten() -> None:
    nlfc_mem = T.allocate([1024], "int8", "nlfcmem")
    nlfc_table = T.allocate([1024], "int8", "dm")
    X = T.allocate([256], "float16", "vm")
    Y = T.allocate([256], "float16", "vm")
    nlfc_res = T.var("float16x32")
    for i in T.serial(8):
        T.evaluate(
            T.nnp_vidma_load_nlfc(
                "int8",
                T.tvm_access_ptr(
                    T.type_annotation(dtype="int8"), nlfc_mem.data, 0, 1024, "w", dtype="handle"
                ),
                T.tvm_access_ptr(
                    T.type_annotation(dtype="int8"), nlfc_table.data, 0, 1024, "r", dtype="handle"
                ),
                "dtype_vidma=0",
                "start_addr_in_en_vidma=1",
                "start_addr_out_en_vidma=1",
                "cb_buf_vm_vidma=1",
                "cb_buf_dm_vidma=1",
                "nlfc_mem_en_vidma=1",
                "j0_loop_sel_vidma=3",
                "j1_loop_sel_vidma=2",
                "j2_loop_sel_vidma=1",
                "j3_loop_sel_vidma=0",
                "j0_loop_num_vidma=1",
                "j1_loop_num_vidma=2",
                "j2_loop_num_vidma=128",
                "j3_loop_num_vidma=4",
                "j0_stridein_vidma=1024",
                "j1_stridein_vidma=512",
                "j2_stridein_vidma=4",
                "j0_strideout_vidma=1024",
                "j1_strideout_vidma=512",
                "j2_strideout_vidma=4",
                "wo_data_size_vm_vidma=256",
                "wo_data_size_dm_vidma=256",
                "ub_data_size_vm_vidma=256",
                "ub_data_size_dm_vidma=256",
                dtype="",
            )
        )
        with T.let(
            nlfc_res,
            T.nnp_inline_asm_vcu(
                "=&{vv},{vv}",
                "some inline asms",
                16,
                0,
                1,
                X[T.ramp(i * 32, 1, 32)],
                1,
                nlfc_table.data,
                dtype="float16x32",
            ),
        ):
            Y[T.ramp(i * 32, 1, 32)] = nlfc_res


def test_vu_nlfc_sigmoid_example():
    rewritten = RewriteNlfc()(tvm.IRModule.from_expr(vu_nlfc_sigmoid_example))["main"]

    def post_order(obj):
        """Work around for non-supported string imm with newline"""
        if isinstance(obj, tvm.tir.StringImm):
            if obj.value.find("\n") >= 0:
                return tvm.tir.StringImm("some inline asms")
        return obj

    body = tvm.tir.stmt_functor.ir_transform(rewritten.body, None, post_order)

    tvm.ir.assert_structural_equal(rewritten.params, vu_nlfc_sigmoid_example_rewritten.params)
    tvm.ir.assert_structural_equal(body, vu_nlfc_sigmoid_example_rewritten.body)


if __name__ == "__main__":
    test_vu_nlfc_sigmoid_example()
