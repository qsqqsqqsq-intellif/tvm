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
# pylint: disable=invalid-name
"""Edgex logger"""
import sys
import os
import logging


__all__ = ["EdgexLog"]

"""
1. Set the "EDGEX_LOG_LEVEL" environment variable
  to one of _LOGGING_LEVELS keys to change log level,
  the default EDGEX_LOG_LEVEL is "DEBUG"

2. Set the "EDGEX_LOG_COLOR" environment variable
  to 0/1 to disable/enable print colored log in console,
  the default EDGEX_LOG_COLOR value is "1"
"""

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

# The value of log level
_DEBUG = 10
_INFO = 20
_WARNING = 30
_ERROR = 40
_CRITICAL = 50

# Define the logging._levelToName and logging._nameToLevel
logging._levelToName = {
    _DEBUG: "D",
    _INFO: "I",
    _WARNING: "W",
    _ERROR: "E",
    _CRITICAL: "C",
}

logging._nameToLevel = {
    "D": _DEBUG,
    "I": _INFO,
    "W": _WARNING,
    "E": _ERROR,
    "C": _CRITICAL,
}

# Set te compiler logging to warning level, it's too long and useless generally
logging.getLogger("te_compiler").setLevel(logging.WARNING)

# Escape codes to get colored ouput
# The format: \033[display mode;foreground color;background color m
# The background is set with 40 plus the number of the color, and the foreground with 30
_RESET_SEQ = "\033[0m"
_COLOR_SEQ = "\033[1;%dm"
_BOLD_SEQ = "\033[1m"

_COLORS = {
    "D": BLUE,
    "I": GREEN,
    "W": YELLOW,
    "E": RED,
    "C": RED,
}

# Edgex log level key-value map
# key: log level environment variable, value: level str name
_LOGGING_LEVELS = {
    "DEBUG": "D",
    "INFO": "I",
    "WARNING": "W",
    "ERROR": "E",
    "CRITICAL": "C",
}

# Get edgex log level environment variable
_level = os.getenv("EDGEX_LOG_LEVEL")
if _level in _LOGGING_LEVELS.keys():
    _LOG_LEVEL = logging.getLevelName(_LOGGING_LEVELS.get(_level))
else:
    # Set the default log level to DEBUG
    _LOG_LEVEL = _DEBUG

# Get edgex log color environment variable
_LOG_COLOR = int(os.getenv("EDGEX_LOG_COLOR", "1"))


def formatter_message(message, use_color=True):
    if use_color:
        message = message.replace("$RESET", _RESET_SEQ).replace("$BOLD", _BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message


class ColoredFormatter(logging.Formatter):
    """A formatter allows colored the levelname.
    Intended to help to create more readable logging.
    """

    def __init__(self, fmt: str, use_color=True, datefmt=None, style="%"):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in _COLORS:
            levelname_color = _COLOR_SEQ % (30 + _COLORS[levelname]) + levelname + _RESET_SEQ
            record.levelname = levelname_color
            # if need colored the message enable the following code.
            # message = record.msg
            # message_color = _COLOR_SEQ % (30 + _COLORS[levelname]) + message + _RESET_SEQ
            # record.msg = message_color
        return logging.Formatter.format(self, record)


class EdgexLog(object):
    """Wrap a edgex logging manager.

     Examples
    --------
    .. code-block:: python

    import sys
    from tvm.contrib.edgex.base.edgexlog import EdgexLog as el
    ...
    el.i("Start set config.")
    el.w("The config is None.")
    el.e("Invalid EdgexConfig object. file:{}, line:{}.".format(
        __file__, sys._getframe().f_lineno))
    ...
    """

    FORMAT = "[$BOLD%(name)s$RESET][ %(levelname)s ]  %(message)s$RESET"
    COLOR_FORMAT = formatter_message(FORMAT, _LOG_COLOR)
    logger_initialized = False
    warning_count = 0

    if not logger_initialized:
        logger = logging.getLogger(name="EdgexLogger")
        logger.setLevel(_DEBUG)
        logger.propagate = False
        log_handler = logging.StreamHandler(sys.stderr)
        log_handler.setLevel(_LOG_LEVEL)
        color_formatter = ColoredFormatter(COLOR_FORMAT, _LOG_COLOR)
        log_handler.setFormatter(color_formatter)
        logger.addHandler(log_handler)
        logger_initialized = True

    @staticmethod
    def p(msg):
        # print message
        sys.stdout.write("\rI %s" % msg)
        sys.stdout.flush()

    @staticmethod
    def d(msg):
        # print debug message
        EdgexLog.logger.debug(msg)

    @staticmethod
    def i(msg):
        # print info message
        EdgexLog.logger.info(msg)

    @staticmethod
    def w(msg):
        # print warning message
        EdgexLog.logger.warning(msg)
        EdgexLog.warning_count += 1

    @staticmethod
    def e(msg, abort=True):
        # print error message
        EdgexLog.logger.error(msg)
        if abort:
            EdgexLog.print_log_count()
            # Raise error for callstack backstrace
            raise ValueError(msg)

    @staticmethod
    def reset_log_count():
        EdgexLog.warning_count = 0

    @staticmethod
    def print_log_count():
        if EdgexLog.warning_count > 0:
            EdgexLog.w("----------------Warning({})----------------".format(EdgexLog.warning_count))
        else:
            EdgexLog.i("----------------Warning({})----------------".format(EdgexLog.warning_count))
        EdgexLog.reset_log_count()
