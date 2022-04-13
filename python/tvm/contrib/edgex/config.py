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
# pylint: disable=invalid-name, redefined-outer-name
"""Edgex hardware configuration"""
import os
import sys
import json
import tvm
from tvm.contrib.edgex.base.edgexlog import EdgexLog as el


def get_python_edgex_absdir():
    """Get the edgex python AP"""
    return os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))


def get_cfg_file(target="default"):
    """Get the config json file"""
    curr_path = get_python_edgex_absdir()
    cfg_path = os.path.join(curr_path, "config/edgex_config_%s.json" % target)
    if not os.path.exists(cfg_path):
        el.e("%s not exists." % cfg_path)
    return cfg_path


class EdgexConfig(object):
    """Edgex configuration object.

    This object contains all the information
    needed for compiling to a specific edgex backend.

    Parameters
    ----------
    cfg : dict of str to value, or config file name.
    """

    _current = None

    def __init__(self, cfg):
        if isinstance(cfg, str):
            with open(get_cfg_file(cfg)) as f:
                cfg = json.load(f)
        # config attrs
        self._cfg = cfg
        self.__dict__.update(cfg)
        # cls attrs
        self._last_cfg = None
        self.cube_enable = int(self.PE_NUM) - 1

    def __enter__(self):
        self._last_cfg = EdgexConfig._current
        EdgexConfig._current = self
        return self

    def __exit__(self, ptype, value, trace):
        if self._last_cfg is None:
            el.e("Exit failed, last config is None.")
        EdgexConfig._current = self._last_cfg

    @classmethod
    def make_default_config(cls):
        with open(get_cfg_file()) as f:
            cfg = json.load(f)
        edgex_cfg = cls(cfg)
        return edgex_cfg

    @classmethod
    def get_current(cls):
        if cls._current is None:
            el.e("Get EdgexConfig failed. {}, line:{}.".format(__file__, sys._getframe().f_lineno))
        return cls._current

    @classmethod
    def set_current(cls, curr):
        if curr is None:
            el.e("Set EdgexConfig failed. {}, line:{}.".format(__file__, sys._getframe().f_lineno))
        cls._current = curr


@tvm.register_func("tvm.edgex.get_current_hw_config")
def get_current_hw_config():
    return EdgexConfig.get_current()._cfg


cfg = EdgexConfig.make_default_config()
EdgexConfig.set_current(cfg)
