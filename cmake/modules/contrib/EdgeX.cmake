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

if(USE_EDGEX)
  set(EDGEX_ROOT_DIR ${USE_EDGEX})
  include_directories(SYSTEM ${EDGEX_ROOT_DIR}/include)
  include_directories(SYSTEM ${EDGEX_ROOT_DIR}/ass)
  message(STATUS "Build with EdgeX support")
  if(DEFINED EDGEX_GRPC_LIB)
    message(STATUS "Use EdgeX grpc dependency libraries at ${EDGEX_GRPC_LIB}")
    file(GLOB EDGEX_LIB
      ${EDGEX_ROOT_DIR}/lib/libdcl*.so
      ${EDGEX_ROOT_DIR}/lib/libprotobuf.so
      ${EDGEX_ROOT_DIR}/lib/liberror_string.so
      ${EDGEX_ROOT_DIR}/lib/librpc.so
      ${EDGEX_GRPC_LIB}/lib*.so)
  else()
    file(GLOB EDGEX_LIB ${EDGEX_ROOT_DIR}/lib/*.so)
  endif()
  file(GLOB RUNTIME_EDGEX_SRCS src/runtime/edgex/*.cc)
  list(APPEND TVM_LINKER_LIBS ${EDGEX_LIB})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${EDGEX_LIB})
  list(APPEND RUNTIME_SRCS ${RUNTIME_EDGEX_SRCS})
  message(STATUS "Build with contrib.edgex")

  # llvm codegen should be compiled with no-rtti
  file(GLOB EDGEX_LLVM_SRC src/target/edgex/*.cc)
  if(NOT MSVC)
    set_source_files_properties(${EDGEX_LLVM_SRC}
      PROPERTIES COMPILE_DEFINITIONS "DMLC_ENABLE_RTTI=0")
    set_source_files_properties(${EDGEX_LLVM_SRC}
      PROPERTIES COMPILE_FLAGS "-fno-rtti")
  endif()
  list(APPEND COMPILER_SRCS ${EDGEX_LLVM_SRC})

  file(GLOB_RECURSE EDGEX_CONTRIB_SRC src/contrib/edgex/*.cc)
  list(APPEND COMPILER_SRCS ${EDGEX_CONTRIB_SRC})
else()
  if(DEFINED USE_EDGEX_QUANTIZATION)
      message(STATUS "Build with EdgeX quantization support only")
      file(GLOB QUANTIZATION_SRCS src/contrib/edgex/quantization/*.cc)
      list(APPEND COMPILER_SRCS ${QUANTIZATION_SRCS})
  endif()
endif(USE_EDGEX)
