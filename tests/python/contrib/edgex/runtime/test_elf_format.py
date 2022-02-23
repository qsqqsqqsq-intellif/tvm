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
import subprocess
import os
import tempfile
import time
import tvm
from tvm.contrib.edgex.runtime import create_relocatable_object
from tvm.contrib.edgex.runtime.elf_utils import merge_relocatable_object, wrap_obj_as_single_kernel
from tvm.contrib.edgex.runtime.edgex_runtime import create_llvm_module, edgex_invoke_assembler
from tvm.contrib.edgex.testing import check_edgex_tir_build


def get_simple_lowered_add(funcname, rhs) -> tvm.IRModule:
    x = tvm.te.placeholder([128], "int32")
    y = tvm.te.compute([128], lambda i: x[i] + rhs)
    func = tvm.te.create_prim_func([x, y])
    func = tvm.tir.PrimFunc(func.params, func.body, func.ret_type, func.buffer_map).with_attr(
        "global_symbol", funcname
    )
    s = tvm.contrib.edgex.tir.schedule.EdgexSchedule(func)
    block = s.get_child_blocks(s.get_block("root"))[0]
    x_dm = s.cache_read(block, 0, "dm")
    x_vm = s.cache_read(block, 0, "vm")
    y_vm = s.cache_write(block, 0, "vm")
    y_dm = s.cache_write(y_vm, 0, "dm")
    s.vectorize(s.get_loops(block)[0])
    s.pragma(s.get_loops(x_dm)[-1], "nnp_dma_scope", "eidma")
    s.pragma(s.get_loops(x_vm)[-1], "nnp_dma_scope", "vidma")
    s.pragma(s.get_loops(y_vm)[-1], "nnp_dma_scope", "vodma")
    s.pragma(s.get_loops(y_dm)[-1], "nnp_dma_scope", "eodma")
    with tvm.contrib.edgex.build_config_nnp():
        return tvm.lower(s.mod, name=funcname)


def test_merge_kernel_objects():
    dcl_root_dir = os.environ.get("EDGEX_ROOT_DIR", "./")
    llvm_bin_dir = os.environ.get("EDGEX_LLVM_TOOLCHAIN_DIR")
    assert llvm_bin_dir, "EDGEX_LLVM_TOOLCHAIN_DIR not configured"
    dir_prefix = os.path.join(tempfile.gettempdir(), "test_merge_obj_" + str(int(time.time())))
    os.makedirs(dir_prefix)

    obj1 = dir_prefix + "/1.obj"
    obj2 = dir_prefix + "/2.obj"
    libs = [dir_prefix + "/lib1.obj", dir_prefix + "/lib2.obj"]
    obj1_linked = dir_prefix + "/1.linked.obj"
    obj2_linked = dir_prefix + "/2.linked.obj"
    obj1_fused = dir_prefix + "/1.singlefunc.obj"
    obj2_fused = dir_prefix + "/2.singlefunc.obj"
    obj_final = dir_prefix + "/final.obj"

    for i, lib_name in enumerate(["get_mbx_lock", "cfg_vu_desp"]):
        out_dir = edgex_invoke_assembler(
            lib_name, os.path.join(dcl_root_dir, "ass", f"{lib_name}.asm"), 0, dir_prefix, True
        )
        create_relocatable_object(
            out_dir + f"/{lib_name}.bin",
            out_dir + f"/{lib_name}_cpp.lst",
            libs[i],
        )

    add1 = get_simple_lowered_add("add1", 1)
    mod = tvm.IRModule({"add1": add1["main"]})
    llvm_mod = create_llvm_module(mod, tvm.target.edgex())
    with tempfile.NamedTemporaryFile("w") as ll_file:
        ll_file.write(llvm_mod)
        ll_file.flush()
        subprocess.call(
            [f"{llvm_bin_dir}/llc", ll_file.name, "--filetype=obj", "-mtriple=nnp", "-o", obj1]
        )
    merge_relocatable_object([obj1], libs, obj1_linked)
    wrap_obj_as_single_kernel(obj1_linked, "add1_kernel0", obj1_fused)

    add2 = get_simple_lowered_add("add2", 1)
    mod = tvm.IRModule({"add2": add2["main"]})
    llvm_mod = create_llvm_module(mod, tvm.target.edgex())
    with tempfile.NamedTemporaryFile("w") as ll_file:
        ll_file.write(llvm_mod)
        ll_file.flush()
        subprocess.call(
            [f"{llvm_bin_dir}/llc", ll_file.name, "--filetype=obj", "-mtriple=nnp", "-o", obj2]
        )
    merge_relocatable_object([obj2], libs, obj2_linked)
    wrap_obj_as_single_kernel(obj2_linked, "add2_kernel0", obj2_fused)

    merge_relocatable_object([obj1_fused, obj2_fused], [], obj_final)
    status = subprocess.call([f"{llvm_bin_dir}/llvm-readelf", "-a", obj_final])
    assert status == 0


if __name__ == "__main__":
    test_merge_kernel_objects()
