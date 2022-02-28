#!/bin/sh
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

# How to run?
# - compiler your tvm first
# - set your `WORK_DIR` and `DEMODEL_DIR`
# - set your `TARGET`, either `tflite` or `onnx`
# - set list of models in `modelList`, eg: `ls /data/share/demodels-lfs/tflite/Visual/ > modelList`
# - comment `relay_debug` stages in case you don't need them.
# - run test_end2end_edgex_networks.sh > model.log, better to record the process.

# Notes
# - model.log contains time spend of each model.
# - the results are in tfliteV_x/xx_f, where xx means in-between stage.
# - Find directories that do not contain a directory: `for x in *;do [ -e $x/public_html/bar ] || echo $x;done`. change echo $x to mv if necessary.

alias relay_debug="python3 -c 'import onnx; import tvm; from tvm.contrib.edgex.relay.graph_debugger import main; main()'"

DEMODEL_DIR=/data/share/demodels-lfs/
# change this based your workspace
TARGET="tflite"

cat modelList | while read line
do
  if [ -z "${line}" ]; then
    break
  fi
  if [ "${TARGET}" = "tflite" ]; then
    MODEL=`ls ${DEMODEL_DIR}/tflite/Visual/${line}/*.tflite | head -n 1`
  elif [ "${TARGET}" = "onnx" ]; then
    MODEL=`ls ${DEMODEL_DIR}/onnx/${line}/*.onnx | head -n 1`
  fi
  # change this based your workspace
  WORK_DIR=${HOME}/dev/models/test_models/tfliteV/${line}
  mkdir -p ${WORK_DIR}
  ln -sf ${MODEL} ${WORK_DIR}/${line}.${TARGET}
  start=`date +%s`
  {
    relay_debug convert ${WORK_DIR}/${line}.tflite -o ${WORK_DIR}/origin
    echo "************frontend done***********"
    relay_debug quantize --json ${WORK_DIR}/origin/${line}.json --params ${WORK_DIR}/origin/${line}.params -o ${WORK_DIR}/quantized
    echo "************quantize done***********"
    relay_debug fuse --json ${WORK_DIR}/quantized/${line}.json --params ${WORK_DIR}/quantized/${line}.params -o ${WORK_DIR}/fused
    echo "************FusionStitch done***********"
    relay_debug init --json ${WORK_DIR}/fused/${line}.json --params ${WORK_DIR}/fused/${line}.params -d ${WORK_DIR}/debug
    relay_debug status
    relay_debug run -v
  } >${WORK_DIR}/log.txt 2>&1
  end=`date +%s`
  runtime=$((end-start))
  echo "Process model: ${line}, time spent: ${runtime}s, workdir=${WORK_DIR}"

done
wait
