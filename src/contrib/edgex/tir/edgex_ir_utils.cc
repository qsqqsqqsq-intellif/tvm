/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file edgex_ir_utils.cc
 * \brief Helper functions to edgex tirs.
 */
#include "edgex_ir_utils.h"

#include "./attrs.h"
#include "./op/builtin.h"

namespace tvm {
namespace tir {
namespace edgex {

int GetValueByKey(const CallNode* call, const std::string& key) {
  CHECK(call) << "Invalid call node.";
  for (const PrimExpr& arg : call->args) {
    if (const StringImmNode* sn = arg.as<StringImmNode>()) {
      std::string arg_str = sn->value;
      if (arg_str.find(key) != std::string::npos) {
        std::size_t pos = arg_str.find("=");
        std::string str_val = arg_str.substr(pos + 1);
        if (str_val.find("0x") != std::string::npos) {
          int val{0};
          std::istringstream(str_val) >> std::hex >> val;
          return val;
        }
        return atoi(str_val.c_str());
      }
    }
  }
  return -1;
}

std::ostream& operator<<(std::ostream& os, NNPUnitKind kind) {
  if (kind == NNPUnitKind::ALL) {
    os << "all";
  } else if (kind == NNPUnitKind::CU) {
    os << "cu";
  } else {
    os << "vcu";
  }
  return os;
}

PrimExpr CreateNNPInlineAsmVcu(const std::string& constraint, const std::string& inline_asm,
                               size_t vf, const DataType& result_type,
                               const std::vector<DataType>& state_types,
                               const Array<PrimExpr>& args,
                               const Array<PrimExpr>& placeholder_args) {
  ICHECK_EQ(vf & (vf - 1), 0);
  Array<PrimExpr> intrin_args;
  intrin_args.push_back(StringImm(constraint));
  intrin_args.push_back(StringImm(inline_asm));
  intrin_args.push_back(make_const(DataType::Int(32), vf));
  intrin_args.push_back(make_const(DataType::Int(32), state_types.size()));
  for (size_t i = 0; i < state_types.size(); ++i) {
    intrin_args.push_back(TypeAnnotation(state_types[i]));
  }
  intrin_args.push_back(make_const(DataType::Int(32), args.size()));
  for (size_t i = 0; i < args.size(); ++i) {
    intrin_args.push_back(args[i]);
  }
  intrin_args.push_back(make_const(DataType::Int(32), placeholder_args.size()));
  for (size_t i = 0; i < placeholder_args.size(); ++i) {
    intrin_args.push_back(placeholder_args[i]);
  }
  return std::move(Call(result_type, tvm::tir::edgex::builtin::nnp_inline_asm_vcu(), intrin_args));
}

NlfcOpInfo::NlfcOpInfo(const Array<String>& table_keys, String inst_name, int32_t nlf_th_value,
                       bool nlf_th_mode, bool nlf_th_sel) {
  ObjectPtr<NlfcOpInfoNode> node = make_object<NlfcOpInfoNode>();
  node->table_keys = table_keys;
  node->inst_name = inst_name;
  node->nlf_th_value = nlf_th_value;
  node->nlf_th_mode = nlf_th_mode;
  node->nlf_th_sel = nlf_th_sel;
  data_ = std::move(node);
}

}  // namespace edgex

DataType GetBufferElementType(const Var& buffer_var) {
  const auto* ptr_type = buffer_var->type_annotation.as<PointerTypeNode>();
  ICHECK(ptr_type) << "The provided variable is not of buffer pointer type: " << buffer_var << " "
                   << buffer_var->type_annotation;
  const auto* prim_type = ptr_type->element_type.as<PrimTypeNode>();
  ICHECK(prim_type) << "The provided variable is not of buffer of prim datatypes: " << buffer_var
                    << " " << buffer_var->type_annotation;
  return prim_type->dtype;
}

}  // namespace tir
}  // namespace tvm
