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
from tvm import tir
from tvm.contrib.edgex.arith import nlfc
import numpy as np


def test_convert_nlfc_tables_f16():
    k0 = np.random.uniform(-64, 64, [128]).astype("float16")
    b0 = np.random.uniform(-64, 64, [128]).astype("float16")
    b1 = np.random.uniform(-64, 64, [128]).astype("float16")
    b2 = np.random.uniform(-64, 64, [128]).astype("float16")
    bin_data = nlfc.nlfc_table_numpy_to_bin(k0, b0, b1, b2)
    hex_data = nlfc.nlfc_table_bin_to_hex(bin_data)
    bin_data_rev = nlfc.nlfc_table_hex_to_bin(hex_data)
    k0_rev, b0_rev, b1_rev, b2_rev = nlfc.nlfc_table_bin_to_numpy(bin_data_rev, "float16")
    np.testing.assert_allclose(k0, k0_rev)
    np.testing.assert_allclose(b0, b0_rev)
    np.testing.assert_allclose(b1, b1_rev)
    np.testing.assert_allclose(b2, b2_rev)


def test_convert_nlfc_tables_f32():
    k0 = np.random.uniform(-64, 64, [128]).astype("float32")
    b0 = np.random.uniform(-64, 64, [128]).astype("float32")
    b1 = np.random.uniform(-64, 64, [128]).astype("float32")
    b2 = np.random.uniform(-64, 64, [128]).astype("float32")
    bin_data = nlfc.nlfc_table_numpy_to_bin(k0, b0, b1, b2)
    hex_data = nlfc.nlfc_table_bin_to_hex(bin_data)
    bin_data_rev = nlfc.nlfc_table_hex_to_bin(hex_data)
    k0_rev, b0_rev, b1_rev, b2_rev = nlfc.nlfc_table_bin_to_numpy(bin_data_rev, "float32")
    np.testing.assert_allclose(k0, k0_rev)
    np.testing.assert_allclose(b0, b0_rev)
    np.testing.assert_allclose(b1, b1_rev)
    np.testing.assert_allclose(b2, b2_rev)


if __name__ == "__main__":
    test_convert_nlfc_tables_f16()
    test_convert_nlfc_tables_f32()
