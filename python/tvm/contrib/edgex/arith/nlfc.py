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
"""TVM EdgeX nlfc utils"""
# pylint: disable-msg=C0103,W0401,W0614
import os
import struct
import numpy as np
import tvm


DEFAULT_NLFC_TABLE_DIR = os.path.join(tvm.__path__[0], "contrib/edgex/arith/nlfc_tables")

NLFC_BUCKET_NUM = 128
NLFC_TABLE_PART_BYTES = NLFC_BUCKET_NUM * 4


def nlfc_table_hex_to_bin(hex_data):
    """Convert nlfc table hex to bin
    Nlfc hex is of 64 hex chars per line, 32 lines for non-iter
    and 64 lines for iter algorithm.
    For non-iter version, the layout is [B, K]
    For iter version, the layout is [B0, K, B1, B2, ... Bk]
    """
    lines = hex_data.strip().split("\n")
    assert len(lines) % 32 == 0
    bin_data = bytearray(len(lines) * 32)
    for i, line in enumerate(lines):
        assert len(line) == 64
        for j in range(32):
            byte = int(line[64 - 2 * j - 2 : 64 - 2 * j], 16)
            bin_data[i * 32 + j] = byte
    return bin_data


def nlfc_table_bin_to_hex(bin_data):
    """Convert nlfc table bin to hex format"""
    hex_data = ""
    assert len(bin_data) % 32 == 0
    num_lines = len(bin_data) // 32
    for i in range(num_lines):
        byte_list = struct.unpack("32B", bin_data[i * 32 : i * 32 + 32])
        for byte in reversed(byte_list):
            hex_data += hex(byte)[2:].rjust(2, "0")
        if i != num_lines - 1:
            hex_data += "\n"
    return hex_data


def nlfc_table_numpy_to_bin(k, first_bias, *next_biases):
    """Convert nlfc table numpy arrays to bin"""
    assert k.dtype == np.float32 or k.dtype == np.float16
    assert len(k.shape) == 1 and k.shape[0] == NLFC_BUCKET_NUM
    assert k.dtype == first_bias.dtype
    assert len(first_bias.shape) == 1 and first_bias.shape[0] == NLFC_BUCKET_NUM
    for b in next_biases:
        assert k.dtype == b.dtype
        assert len(b.shape) == 1 and b.shape[0] == NLFC_BUCKET_NUM
    if k.dtype == np.float16:
        k = np.repeat(k, 2)
    if first_bias.dtype == np.float16:
        first_bias = np.repeat(first_bias, 2)
    res = first_bias.tobytes() + k.tobytes()
    for b in next_biases:
        if b.dtype == np.float16:
            b = np.repeat(b, 2)
        res += b.tobytes()
    return res


def nlfc_table_bin_to_numpy(bin_data, dtype):
    """Convert nlfc table bin to numpy arrays k, b0, b1, b2"""
    res = []
    assert dtype in ["float16", "float32"]
    n_bytes = len(bin_data)
    assert n_bytes % NLFC_TABLE_PART_BYTES == 0 and n_bytes // NLFC_TABLE_PART_BYTES >= 2
    offset = 0
    while offset < n_bytes:
        arr = np.frombuffer(bin_data[offset : offset + NLFC_TABLE_PART_BYTES], "int8")
        if dtype == "float16":
            arr = np.reshape(arr, [128, 2, 2])[:, 0, :].flatten().view("float16")
        else:
            arr = arr.view("float32")
        res.append(arr)
        offset += NLFC_TABLE_PART_BYTES
    first_bias = res[0]
    res[0] = res[1]
    res[1] = first_bias
    return res


def extract_nlfc_params(primfunc: tvm.tir.PrimFunc):
    """Extract nlfc param and data from primfunc, also return primfunc without data annotation.

    Parameters
    ----------
    primfunc: tir.PrimFunc
        Function after ConvertFpToNlfc.

    Returns
    -------
    ret: (list[Var], list[NDArray], tir.PrimFunc)
    """
    if primfunc.attrs is None:
        return None, None, primfunc
    attrs = dict(primfunc.attrs)
    if "NlfcTableParams" in attrs and "NlfcTableData" in attrs:
        nlfc_params = attrs["NlfcTableParams"]
        nlfc_tables = attrs["NlfcTableData"]
        assert len(nlfc_params) == len(nlfc_tables)
        # drop attrs which is non-printable
        attrs.pop("NlfcTableData")
        attrs.pop("NlfcTableParams")
        attrs = None if len(attrs) == 0 else tvm.ir.make_node("DictAttrs", **attrs)
        primfunc = tvm.tir.PrimFunc(
            primfunc.params,
            primfunc.body,
            ret_type=primfunc.ret_type,
            buffer_map=primfunc.buffer_map,
            attrs=attrs,
        )
        return nlfc_params, nlfc_tables, primfunc
    return None, None, primfunc


@tvm._ffi.register_func("tvm.edgex.get_nlfc_table")
def get_nlfc_table(nlfc_op_name, key, dtype):
    """Get ndarray of nlfc table from local file storage
    ${nlfc_dir}/${nlfc_op_name}/{key}.hex

    Parameters
    ----------
    nlfc_op_name: str
        Op identifier.

    key : str
        Table key identifier under op.

    dtype : str
        Datatype, should be either float16 or float32

    Returns
    -------
    ret: tvm.nd.NDArray
    """
    if nlfc_op_name.startswith("tir."):
        nlfc_op_name = nlfc_op_name[4:]
    table_dir = os.environ.get("EDGEX_NLFC_TABLE_DIR", DEFAULT_NLFC_TABLE_DIR)
    hex_path = os.path.join(table_dir, nlfc_op_name, key + "_" + dtype + ".txt")
    if not os.path.exists(hex_path):
        raise IOError(f"Can not find nlfc table hex {hex_path}")
    with open(hex_path) as infile:
        hex_data = infile.read()
    bin_data = nlfc_table_hex_to_bin(hex_data)
    np_arr = np.frombuffer(bin_data, "int8")
    return tvm.nd.array(np_arr)
