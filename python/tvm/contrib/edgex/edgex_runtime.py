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
"""TVM EdgeX"""
# pylint: disable-msg=C0103
import os
import sys
import tempfile
import subprocess
import threading
import logging
import shutil
import struct
import numpy as np
import tvm
from tvm import te


def matmul(x, y):
    """Compute add using edgex

    Parameters
    ----------
    x : tvm.te.Tensor
        The input tensor

    y : tvm.te.Tensor
        The input tensor

    Returns
    -------
    ret : tvm.te.Tensor
        The result tensor
    """
    return te.extern(
        (x.shape[0], y.shape[1]),
        [x, y],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.edgex.matmul.forward", ins[0], ins[1], outs[0]
        ),
        name="C",
    )


def add(x):
    """Compute add using edgex

    Parameters
    ----------
    x : tvm.te.Tensor
        The input tensor

    Returns
    -------
    ret : tvm.te.Tensor
        The result tensor
    """
    return te.extern(
        (x.shape[0] // 2,),
        [x],
        lambda ins, outs: tvm.tir.call_packed("tvm.contrib.edgex.add.forward", ins[0], outs[0]),
        name="B",
    )


@tvm._ffi.register_func("tvm.edgex.invoke_assembler")
def edgex_invoke_assembler(
    op_name,
    asm,
    start_pc=-1,
    output_dir=None,
    keep_tmp_dir=True,
):
    """Invoke edgex ass tool chain

    Parameters
    ----------
    asm : str
        The asm file path or asm content.

    op_name : str
        Name prefix of the result files.

    start_pc : int
        Start pc parameter for ass.

    output_dir : str
        Specify directory of output files. If not specified,
        a temp directory is created and deleted after exit.

    keep_tmp_dir : bool
        Whether to keep temporary dir for assembler, Default to True.

    Returns
    -------
    ret : str
        Directory path to store ass results. It contains
        output/${op_name}.bin and output/${op_name}_cpp.lst
    """
    tmp_file_dir = None
    if output_dir is None or output_dir.strip() == "":
        tmp_file_dir = tempfile.mkdtemp(prefix="/tmp/edgex_assembler_workspace_")
        output_dir = tmp_file_dir
    else:
        output_dir = os.path.join(output_dir, op_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    asm_name = "%s.asm" % op_name
    asm_path = os.path.join(output_dir, asm_name)
    if os.path.exists(asm):
        # asm is a file path
        subprocess.call("cp %s %s" % (asm, asm_path))
    elif os.environ.get("EDGEX_DEBUG_USE_EXISTING_ASM", "").strip() == "":
        with open(asm_path, "w") as asm_file:
            asm_file.write(asm)

    bin_dir = os.path.abspath(os.path.join(os.environ.get("EDGEX_ROOT_DIR", "./"), "ass"))
    macro_header_path = os.path.join(bin_dir, "nnp400_main.h")
    status = subprocess.call(["cp", macro_header_path, output_dir])
    if status != 0:
        raise RuntimeError("Copy macro header failed")

    ass_path = os.path.join(bin_dir, "nnp400t_ass.py")
    if not os.path.exists(ass_path):
        raise IOError("%s not exists, check EDGEX_ROOT_DIR" % ass_path)
    hex2bin_path = os.path.join(bin_dir, "hex2bin.py")
    if not os.path.exists(hex2bin_path):
        raise IOError("%s not exists, check EDGEX_ROOT_DIR" % hex2bin_path)

    if start_pc < 0:
        start_pc = tvm.get_global_func("tvm.edgex.get_iss_start_pc")()
    status = subprocess.call(
        ["python3", ass_path, asm_name, "-cpp", "-start_pc", str(start_pc)], cwd=output_dir
    )
    if status != 0:
        raise RuntimeError("Invoke assembler failed")
    hex_path = os.path.join(output_dir, "output", "%s.hex" % op_name)
    bin_path = os.path.join(output_dir, "output", "%s.bin" % op_name)
    status = subprocess.call(["python3", hex2bin_path, hex_path, bin_path])
    if status != 0:
        raise RuntimeError("Invoke hex2bin failed")
    if not keep_tmp_dir and tmp_file_dir is not None:
        shutil.rmtree(tmp_file_dir)
    return os.path.join(output_dir, "output")


@tvm._ffi.register_func("tvm.edgex.launch_iss")
def edgex_launch_iss(
    bin_data,
    lst_data,
    tensors,
    interactive=False,
    op_name=None,
    output_dir=None,
    nnp_main_dir=None,
    keep_tmp_dir=True,
):
    """Invoke edgex nnp iss tool

    Parameters
    ----------
    bin_data : str
        The bin file path or bin content.

    lst_data : str
        The lst file path or lst content.

    tensors : list of ndarray
        Input tensors.

    interactive : bool
        Whether launch iss in interactive mode. Default to False.

    op_name : str
        Optional kernel name for debug simplicity.

    output_dir : str
        Directory for output files.

    nnp_main_dir : str
        Directory of nnp main programs.

    keep_tmp_dir : bool
        Whether to keep temporary dir for iss tool, Default to True.
    """
    server_dir = os.path.join(os.environ.get("EDGEX_SERVER_DIR", "./"))
    iss_dir = os.path.join(server_dir, "iss")
    if not os.path.isdir(iss_dir):
        raise IOError("Can not find iss: %s" % iss_dir)

    if nnp_main_dir is None:
        nnp_main_dir = os.path.join(server_dir, "bin/main")

    tmp_file_dir = None
    if output_dir is None or output_dir.strip() == "":
        tmp_file_dir = tempfile.mkdtemp(prefix="/tmp/edgex_iss_workspace_")
        output_dir = tmp_file_dir
    else:
        if op_name is not None:
            output_dir = os.path.join(output_dir, op_name, "iss")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # bin2hex util
    def bin2hex(data, hex_path):
        hex_lines = 0
        if isinstance(data, str):
            data = str.encode(data)
        length = len(data) // 8
        items = struct.unpack(str(length) + "Q", data)
        assert len(items) % 4 == 0
        with open(hex_path, "w") as hex_file:
            for lineno in range(len(items) // 4):
                v0 = hex(items[lineno * 4])
                v1 = hex(items[lineno * 4 + 1])
                v2 = hex(items[lineno * 4 + 2])
                v3 = hex(items[lineno * 4 + 3])
                line = ""
                line += v3[2:].rjust(16, "0")
                line += v2[2:].rjust(16, "0")
                line += v1[2:].rjust(16, "0")
                line += v0[2:].rjust(16, "0")
                hex_file.write(line + "\n")
                hex_lines += 1
        return hex_lines

    hex_path = os.path.join(output_dir, "op.hex")

    # determine whether pass bin data path as bin_data argument
    if isinstance(bin_data, str) and len(bin_data) < 256 and os.path.exists(bin_data):
        bin_path = bin_data
        with open(bin_path, "rb") as f:
            op_hex_lines = bin2hex(f.read(), hex_path)
    else:
        op_hex_lines = bin2hex(bin_data, hex_path)

    # determine whether pass lst data path as lst_data argument
    if len(lst_data) < 256 and os.path.exists(lst_data):
        with open(lst_data) as f:
            lst_data = f.read()
    lst_path = os.path.join(output_dir, "op.lst")
    with open(lst_path, "w") as f:
        f.write(lst_data)

    # load and save ddr
    load_ddr_cmds = []
    save_ddr_cmds = []
    tensor_ddr_addrs = []
    tensor_sizes = []
    cur_ddr_addr = 0
    for i, t in enumerate(tensors):
        input_tensor_path = "input_tensor_%d.dat" % i
        output_tensor_path = "output_tensor_%d.dat" % i
        with open(os.path.join(output_dir, input_tensor_path), "w") as tfile:
            data = t.numpy().tobytes()
            nbytes = len(data)
            linenum = (nbytes + 31) // 32
            for lineno in range(linenum):
                offset = lineno * 32
                hex_data = data[offset : offset + 32][::-1].hex()
                tfile.write(hex_data.rjust(64, "0"))
                tfile.write("\n")
        # we consequently allocate ddr addresses with at least 128 bytes alignment
        tensor_ddr_addrs.append(cur_ddr_addr)
        tensor_sizes.append(nbytes)
        start_ddr_addr = hex(cur_ddr_addr)
        end_ddr_addr = hex(cur_ddr_addr + nbytes - 1)
        cur_ddr_addr += (nbytes + 127) // 128 * 128
        load_ddr_cmds.append("load ddr %s %s" % (input_tensor_path, start_ddr_addr))
        save_ddr_cmds.append(
            "save ddr %s %s:%s" % (output_tensor_path, start_ddr_addr, end_ddr_addr)
        )
    if len(tensors) % 2 != 0:
        tensor_ddr_addrs.append(0)

    # copy dcl main program
    subprocess.call(["cp", "%s/nnp_main_main.hex" % nnp_main_dir, output_dir])
    subprocess.call(["cp", "%s/nnp_main_cpp.lst" % nnp_main_dir, output_dir])

    # launching configuration with pre-gen cfg
    nnp_queue_cfg_magic = [
        "ff010000000000f00000000100000000",
        "00000000000000010000003000000000",
        "00000000000000010000000000000000",
        "0000000000000000f100000000000000",
        "00000000000000010000000100000001",
        "00000000000000000000000000000000",
        "00000000000000000000000000000000",
        "00000000000000000000000000000000",
        "00000000000000000000000000000000",
        "00000000000000000000000000000000",
        "00000000000000000000000000000000",
        "00000000000000000000000000000000",
        "00000000000000000000000000000000",
        "00000000000000000000000000000000",
        "00000000000000000000000000000000",
    ]
    nnp_code_tran_lens = op_hex_lines * 2
    nnp_queue_cfg_magic[3] = (
        struct.pack("i", nnp_code_tran_lens)[::-1].hex() + nnp_queue_cfg_magic[3][8:]
    )
    with open(os.path.join(output_dir, "nnp_queue_cfg"), "w") as outf:
        outf.write("\n".join(nnp_queue_cfg_magic))

    # two 64-bit ddr per line,  low address is at the right
    tensor_addr_cfg_magic = []
    for i in range((len(tensor_ddr_addrs) + 1) // 2):
        addr0 = struct.pack("Q", tensor_ddr_addrs[i * 2])[::-1].hex()
        addr1 = struct.pack("Q", tensor_ddr_addrs[i * 2 + 1])[::-1].hex()
        tensor_addr_cfg_magic.append(addr1 + addr0)
    with open(os.path.join(output_dir, "tensor_addr_cfg"), "w") as outf:
        outf.write("\n".join(tensor_addr_cfg_magic))

    # reference to dcl server init
    start_pc = tvm.get_global_func("tvm.edgex.get_iss_start_pc")()
    start_lines = start_pc / 4
    nu_breakpoint_lines = "0xf4"
    nnp_main_cmds = [
        "set dis_lvl 0x0",
        "load ddr nnp_main_main.hex 0x00f0000000",
        "load ppm nnp_main_cpp.lst",
        "w reg mbx_ram:25 %d" % start_lines,
        "w reg ccm:4 1",
        "w reg pdma:9 0x00000000",
        "w reg pdma:0 0xf0000000",
        "w reg pdma:1 0",
        "w reg pdma:2 %d" % start_lines,
        "w reg pdma:3 0",
        "w reg pdma:5 1",
        "w reg pdma:7 1",
        "w reg ccm:0 1",
        "w reg vpdma0:9 0x00000000",
        "w reg vpdma0:0 0xf0000000",
        "w reg vpdma0:1 0",
        "w reg vpdma0:2 %d" % start_lines,
        "w reg vpdma0:3 0",
        "w reg vpdma0:5 1",
        "w reg vpdma0:7 1",
        "w reg ccm:1 1",
        "w reg vpdma1:9 0x00000000",
        "w reg vpdma1:0 0xf0000000",
        "w reg vpdma1:1 0",
        "w reg vpdma1:2 %d" % start_lines,
        "w reg vpdma1:3 0",
        "w reg vpdma1:5 1",
        "w reg vpdma1:7 1",
        "w reg ccm:2 1",
        "w reg vpdma2:9 0x00000000",
        "w reg vpdma2:0 0xf0000000",
        "w reg vpdma2:1 0",
        "w reg vpdma2:2 %d" % start_lines,
        "w reg vpdma2:3 0",
        "w reg vpdma2:5 1",
        "w reg vpdma2:7 1",
        "w reg ccm:3 1",
        "w reg vpdma3:9 0x00000000",
        "w reg vpdma3:0 0xf0000000",
        "w reg vpdma3:1 0",
        "w reg vpdma3:2 %d" % start_lines,
        "w reg vpdma3:3 0",
        "w reg vpdma3:5 1",
        "w reg vpdma3:7 1",
        "b %s" % nu_breakpoint_lines,
        "c",
    ]
    prepare_task_cmds = [
        "load ddr op.hex 0x00f1000000",
        "load ddr128 nnp_queue_cfg 0x00ff000000",
        "load ddr128 tensor_addr_cfg 0x00ff010000",
        "w reg mbx_ram:0 0xff000000",
        "w reg mbx_ram:1 0x00000000",
        "w reg mbx_ram:24 0",
        "w reg mbx_ram:26 1",
        "w reg mbx_reg:0 1",
        "w reg mbx_reg:1 1",
        "load ppm op.lst",
    ]

    # record iss commands
    with open(os.path.join(output_dir, "init_commands.txt"), "w") as out_file:
        for cmd in nnp_main_cmds:
            out_file.write(cmd + "\n")
        for cmd in load_ddr_cmds:
            out_file.write(cmd + "\n")
        for cmd in prepare_task_cmds:
            out_file.write(cmd + "\n")
    with open(os.path.join(output_dir, "save_commands.txt"), "w") as out_file:
        for cmd in save_ddr_cmds:
            out_file.write(cmd + "\n")

    # launch nnp iss bin
    iss_bin = os.path.join(iss_dir, "nnp_iss")
    p = subprocess.Popen(
        iss_bin,
        cwd=output_dir,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    def fd_printer(fd):
        for line in fd:
            if not isinstance(line, str):
                line = line.decode()
            sys.stdout.write(line)

    stdout_printer_thread = threading.Thread(target=fd_printer, args=(p.stdout,))
    stdout_printer_thread.setDaemon(True)
    stdout_printer_thread.start()
    stderr_printer_thread = threading.Thread(target=fd_printer, args=(p.stderr,))
    stderr_printer_thread.setDaemon(True)
    stderr_printer_thread.start()

    def execute_iss_cmd(cmd):
        if isinstance(cmd, str):
            cmd = str.encode(cmd.strip() + "\n")
        p.stdin.write(cmd)
        p.stdin.flush()

    try:
        for cmd in nnp_main_cmds:
            execute_iss_cmd(cmd)
        for cmd in load_ddr_cmds:
            execute_iss_cmd(cmd)
        for cmd in prepare_task_cmds:
            execute_iss_cmd(cmd)
        if interactive:
            for line in sys.stdin:
                if line.strip() == "save":
                    for cmd in save_ddr_cmds:
                        execute_iss_cmd(cmd)
                    continue
                execute_iss_cmd(line)
                if line.strip() == "exit":
                    break
        else:
            execute_iss_cmd("c")
            for cmd in save_ddr_cmds:
                execute_iss_cmd(cmd)
            execute_iss_cmd("exit")
    except BrokenPipeError as _:
        pass
    p.wait()
    if p.returncode != 0:
        raise RuntimeError("nnp_iss exit with code=%d" % p.returncode)

    # write back tensors
    for i, t in enumerate(tensors):
        output_tensor_path = "%s/output_tensor_%d.dat" % (output_dir, i)
        if not os.path.exists(output_tensor_path):
            logging.warning("Output %d not found: %s", i, output_tensor_path)
            continue
        with open(output_tensor_path) as tfile:
            data = bytearray(tensor_sizes[i])
            offset = 0
            for line in tfile.readlines():
                hex_data = bytearray.fromhex(line)
                line_size = len(hex_data)
                if offset + line_size < len(data):
                    data[offset : offset + line_size] = hex_data[::-1]
                else:
                    data[offset:] = hex_data[::-1][: len(data) - offset]
                offset += line_size
            np_arr = np.frombuffer(data, dtype=str(t.dtype))
            t.copyfrom(np_arr)

    stdout_printer_thread.join()
    stderr_printer_thread.join()
    if not keep_tmp_dir and tmp_file_dir is not None:
        shutil.rmtree(tmp_file_dir)
