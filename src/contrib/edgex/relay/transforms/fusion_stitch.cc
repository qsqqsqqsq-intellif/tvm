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
 *
 * \file src/relay/transforms/fusion_stitch.cc
 *
 * \brief This is a backend-aware optimization pass.
 *   Fuse necessary ops into a single one using fusion stitch.
 */
#include "edgex_graph.h"

// #define FUSIONS_DEBUG_0
// #define FUSIONS_DEBUG_1
// #define FUSIONS_DEBUG_2
namespace tvm {
namespace relay {

enum AtomicOpType {
  ATOMIC_CONV,
  ATOMIC_FC,
  ATOMIC_POOL,
  ATOMIC_ADAPTIVEPOOL,
  ATOMIC_ELTWISE,
  ATOMIC_REDUCTION,
  ATOMIC_BIASADD,
  ATOMIC_RIGHTSHIFT,
  ATOMIC_RELU,
  ATOMIC_PRELU,
  ATOMIC_SOFTRELU,
  ATOMIC_SOFTMAX,
  ATOMIC_L2NORMALIZE,
  ATOMIC_BATCHFLATTEN,
  ATOMIC_SQUEEZE,
  ATOMIC_TRANSPOSE,
  ATOMIC_SLICE,
  ATOMIC_PAD,
  ATOMIC_CAST,
  ATOMIC_TUPLE,
  ATOMIC_CONCAT,
  ATOMIC_DENSE,
  ATOMIC_VISION,
  ATOMIC_STN,
  ATOMIC_RESHAPE,
  ATOMIC_DEVICECOPY,
  ATOMIC_ANONYMOUS,
  ATOMIC_VU_ELTWISE,
  ATOMIC_VU_POOL,
};

std::vector<std::string> atomic_op_type_name{
    "ATOMIC_CONV",        "ATOMIC_FC",           "ATOMIC_POOL",      "ATOMIC_ADAPTIVEPOOL",
    "ATOMIC_ELTWISE",     "ATOMIC_REDUCTION",    "ATOMIC_BIASADD",   "ATOMIC_RIGHTSHIFT",
    "ATOMIC_RELU",        "ATOMIC_PRELU",        "ATOMIC_SOFTRELU",  "ATOMIC_SOFTMAX",
    "ATOMIC_L2NORMALIZE", "ATOMIC_BATCHFLATTEN", "ATOMIC_SQUEEZE",   "ATOMIC_TRANSPOSE",
    "ATOMIC_SLICE",       "ATOMIC_PAD",          "ATOMIC_CAST",      "ATOMIC_TUPLE",
    "ATOMIC_CONCAT",      "ATOMIC_DENSE",        "ATOMIC_VISION",    "ATOMIC_STN",
    "ATOMIC_RESHAPE",     "ATOMIC_DEVICECOPY",   "ATOMIC_ANONYMOUS", "ATOMIC_VU_ELTWISE",
    "ATOMIC_VU_POOL"};

typedef int OpDeviceType;
OpDeviceType OP_DEVICE_EDGEX = 16;
OpDeviceType OP_DEVICE_DEDSP = 1;
OpDeviceType OP_DEVICE_NNP300 = 2;
OpDeviceType OP_DEVICE_UNKNOWN = 3;

class AtomicGraph {
 public:
  class AtomicOp {
   public:
    AtomicOp(size_t s, size_t e, size_t i, AtomicOpType t, OpDeviceType d)
        : start(s), end(e), index(i), type(t), device(d) {}
    AtomicOp(size_t s, size_t e, AtomicOpType t, OpDeviceType d)
        : start(s), end(e), type(t), device(d) {}

    size_t start{SIZE_MAX};
    size_t end{SIZE_MAX};
    size_t index{SIZE_MAX};
    AtomicOpType type;
    OpDeviceType device;
    std::vector<AtomicOp*> inputs;
    std::vector<AtomicOp*> outputs;

    std::string detail() {
      std::ostringstream os;
      os << "Index[" << index << "] Type[" << atomic_op_type_name[type] << "] Device[" << device
         << "] Scope[" << start << ", " << end << "] inputs_size = " << inputs.size()
         << " outputs_size = " << outputs.size();
      return os.str();
    }
  };

  class AtomicOpSpec {
   public:
    explicit AtomicOpSpec(AtomicOpType type) : type_(type) {}

    size_t size() { return spec_.size(); }

    void push_back(std::string relay_op_name) { spec_.push_back(relay_op_name); }

    void pop_back() { spec_.pop_back(); }

    std::string operator[](size_t index) { return spec_[index]; }

    AtomicOpType type() { return type_; }

   private:
    AtomicOpType type_;
    std::vector<std::string> spec_;
  };

 public:
  AtomicGraph(support::Arena* arena, const EdgexDependencyGraph* p_graph, int device_type)
      : arena_(arena), p_graph_(p_graph), device_type_(device_type) {
    atomic_op_specs_[OP_DEVICE_EDGEX] = generate_nnp200_atomic_op_specs();

    generate_dgraph_op_device_types(device_type);
    genearete_atomic_ops();
  }

  AtomicOp* operator[](size_t index) const { return post_dfs_atomic_ops_[index]; }

  size_t size() const { return post_dfs_atomic_ops_.size(); }

  int device_type() const { return device_type_; }

  const EdgexDependencyGraph* get_dependency_graph() const { return p_graph_; }

 private:
  /*! \brief allocator of all the internal node object */
  support::Arena* arena_;
  const EdgexDependencyGraph* p_graph_;
  int device_type_;
  std::vector<AtomicOp*> post_dfs_atomic_ops_;
  std::unordered_map<OpDeviceType, std::vector<AtomicOpSpec>> atomic_op_specs_;
  std::vector<OpDeviceType> dgraph_op_device_types_;

  void update_device_info_for_outputs(const EdgexDependencyGraph::Node* graph_node,
                                      OpDeviceType device_type) {
    for (auto* link = graph_node->parents.head; link != nullptr; link = link->next) {
      if (link->value->tvm_node != nullptr) {
        dgraph_op_device_types_[link->value->index] = device_type;
      }
    }
  }

  OpDeviceType get_device_info_from_outputs(const EdgexDependencyGraph::Node* graph_node) {
    int device_type = OP_DEVICE_UNKNOWN;
    for (auto* link = graph_node->parents.head; link != nullptr; link = link->next) {
      if (link->value->tvm_node != nullptr) {
        auto next_graph_node = link->value;
        int dt = dgraph_op_device_types_[next_graph_node->index];
        if (auto* callnode = GetRef<ObjectRef>(next_graph_node->tvm_node).as<CallNode>()) {
          if (callnode->op.as<OpNode>()) {
            if (callnode->op.same_as(Op::Get("device_copy"))) {
              auto attr = callnode->attrs.as<DeviceCopyAttrs>();
              if (attr->src_virtual_device->device_type() ==
                      tvm::Target("nnp200")->kind->device_type &&
                  attr->dst_virtual_device->device_type() ==
                      tvm::Target("dedsp")->kind->device_type) {
                dt = OP_DEVICE_EDGEX;
              }
              if (attr->src_virtual_device->device_type() ==
                      tvm::Target("dedsp")->kind->device_type &&
                  attr->dst_virtual_device->device_type() ==
                      tvm::Target("nnp200")->kind->device_type) {
                dt = OP_DEVICE_DEDSP;
              }
            }
          }
        }
        // NOTE(cww): the outputs of function inputs could on
        // different device.
        // if (GetRef<ObjectRef>(graph_node->ref).as<VarNode>()) {
        //   LOG(WARNING) << "The outputs of function inputs could on different device. var = "
        //                << GetRef<ObjectRef>(graph_node->ref);
        // } else
        {
          CHECK((device_type == OP_DEVICE_UNKNOWN) || (device_type == dt))
              << "Error in get_device_info_from_outputs";
        }
        device_type = dt;
      }
    }
    return (OpDeviceType)device_type;
  }

  void generate_dgraph_op_device_types(int device_type) {
    auto graph = *p_graph_;
    dgraph_op_device_types_ = std::vector<OpDeviceType>(graph.size(), OP_DEVICE_UNKNOWN);
    // phase 0: forward
    bool found_device_info_in_phase0 = false;
    for (size_t nid = 0; nid < graph.size(); ++nid) {
      // the group of current node has been specified already.
      auto* graph_node = graph[nid];

      // set next_device_type same with current device type in default
      int next_device_type = dgraph_op_device_types_[nid];
      if (auto* callnode = GetRef<ObjectRef>(graph_node->tvm_node).as<CallNode>()) {
        if (callnode->op.as<OpNode>()) {
          if (callnode->op.same_as(Op::Get("device_copy"))) {
            auto attr = callnode->attrs.as<DeviceCopyAttrs>();
            if (attr->src_virtual_device->device_type() ==
                    tvm::Target("nnp200")->kind->device_type &&
                attr->dst_virtual_device->device_type() ==
                    tvm::Target("dedsp")->kind->device_type) {
              next_device_type = OP_DEVICE_DEDSP;
            } else if (attr->src_virtual_device->device_type() ==
                           tvm::Target("dedsp")->kind->device_type &&
                       attr->dst_virtual_device->device_type() ==
                           tvm::Target("nnp200")->kind->device_type) {
              next_device_type = OP_DEVICE_EDGEX;
            } else {
              LOG(FATAL) << "Unsupported.";
            }
            dgraph_op_device_types_[nid] = OP_DEVICE_DEDSP;
          }
        }
      }

      // TODO(cww): remove if
      if (!GetRef<ObjectRef>(graph_node->tvm_node).as<VarNode>() &&
          !GetRef<ObjectRef>(graph_node->tvm_node).as<ConstantNode>()) {
        update_device_info_for_outputs(graph_node, (OpDeviceType)next_device_type);
      } else {
        CHECK(0);
      }

      // once is enough
      found_device_info_in_phase0 =
          (next_device_type != OP_DEVICE_UNKNOWN) || found_device_info_in_phase0;
    }

    // this->PrintDebugInfo(graph, post_dom_tree);
    if (found_device_info_in_phase0) {
      // phase 1: backward, ignore the last node
      CHECK_NE(dgraph_op_device_types_[graph.size() - 1], OP_DEVICE_UNKNOWN)
          << "Please check device_type of the last node";
      for (size_t nid = graph.size() - 2; nid >= 0 && nid < graph.size() - 1; --nid) {
        auto* graph_node = graph[nid];
        if (dgraph_op_device_types_[nid] == OP_DEVICE_UNKNOWN) {
          dgraph_op_device_types_[nid] = get_device_info_from_outputs(graph_node);
        }
      }
    } else {
      // phase 1: set all device_type = OP_DEVICE_EDGEX
      for (size_t nid = 0; nid < graph.size(); ++nid) {
        if (device_type == kDLEdgeX) {
          dgraph_op_device_types_[nid] = OP_DEVICE_EDGEX;
        } else {
          LOG(FATAL) << "Unsupported.";
        }
      }
    }
  }

  void add_specs(AtomicOpType type, const std::vector<std::vector<std::string>>& op_names_list,
                 int from_index, AtomicOpSpec* p_cur_spec, std::vector<AtomicOpSpec>* p_specs) {
    if (from_index >= static_cast<int>(op_names_list.size())) {
      if (p_cur_spec->size() > 0) {
#ifdef FUSIONS_DEBUG_2
        std::ostringstream os;
        os << "[Add spec]: ";
        for (size_t i = 0; i < p_cur_spec->size(); i++) {
          auto pa = (*p_cur_spec)[i];
          os << " " << pa;
        }
        LOG(INFO) << os.str();
#endif
        p_specs->push_back(*p_cur_spec);
      }
      return;
    }

    auto op_names = op_names_list[from_index];
    for (auto op_name : op_names) {
      if (op_name != "") {
        p_cur_spec->push_back(op_name);
      }
      add_specs(type, op_names_list, from_index + 1, p_cur_spec, p_specs);
      if (op_name != "") {
        p_cur_spec->pop_back();
      }
    }
  }

  void add_atomic_op_specs(std::vector<AtomicOpSpec>* p_specs, AtomicOpType type,
                           const std::vector<std::vector<std::string>>& op_names_list) {
    AtomicOpSpec cur_spec(type);
    add_specs(type, op_names_list, 0, &cur_spec, p_specs);
  }

  std::vector<AtomicOp*> genearete_serial_atomic_ops(size_t start_index, size_t end_index,
                                                     OpDeviceType device_type,
                                                     bool enable_anonymous = false) {
    std::vector<AtomicOp*> ops;

    auto is_matched = [](const ObjectRef& node_ref, const std::string& spec_item,
                         AtomicOpType spec_type, OpDeviceType device_type,
                         bool is_first_item) -> bool {
      if (auto* node = node_ref.as<CallNode>()) {
        auto* ttype_in = node->type_args[0].as<TensorTypeNode>();
        auto* ttype_out = node->checked_type().as<TensorTypeNode>();
        bool float_involved = ((ttype_in && ttype_in->dtype.is_float()) ||
                               (ttype_out && ttype_out->dtype.is_float()));
        auto arg0_call = node->args[0].as<CallNode>();
        if (OP_DEVICE_NNP300 == device_type && arg0_call && !is_first_item &&
            arg0_call->op.as<OpNode>()->name == "nn.pad") {
          if (arg0_call->attrs.as<PadAttrs>()->pad_mode == "constant" &&
              (*(tir::as_const_int(arg0_call->attrs.as<PadAttrs>()->pad_width[0][0])) > 0 ||
               *(tir::as_const_int(arg0_call->attrs.as<PadAttrs>()->pad_width[0][1])) > 0 ||
               *(tir::as_const_int(arg0_call->attrs.as<PadAttrs>()->pad_width[1][0])) > 0 ||
               *(tir::as_const_int(arg0_call->attrs.as<PadAttrs>()->pad_width[1][1])) > 0 ||
               *(tir::as_const_int(arg0_call->attrs.as<PadAttrs>()->pad_width[2][0])) > 15 ||
               *(tir::as_const_int(arg0_call->attrs.as<PadAttrs>()->pad_width[2][1])) > 15 ||
               *(tir::as_const_int(arg0_call->attrs.as<PadAttrs>()->pad_width[3][0])) > 15 ||
               *(tir::as_const_int(arg0_call->attrs.as<PadAttrs>()->pad_width[3][1])) > 15)) {
            return false;
          }
        }
        // for NNP300, ATOMIC_ELTWISE is nu only, vu use type ATOMIC_VU_ELTWISE
        bool is_nu_only = (spec_type == ATOMIC_ELTWISE);
        if (OP_DEVICE_NNP300 == device_type && is_nu_only && float_involved) {
          return false;
        } else if (spec_item != "tuple" && spec_item != "tuple_get" &&
                   spec_item != "tuple_getitem" && node->op.same_as(Op::Get(spec_item))) {
          return true;
        }
      } else if (node_ref.as<TupleNode>()) {
        if (spec_item == "tuple") return true;
      } else if (node_ref.as<TupleGetItemNode>()) {
        if (spec_item == "tuple_get" || spec_item == "tuple_getitem") return true;
      }
      return false;
    };

    auto& specs = atomic_op_specs_[device_type];
    size_t i = start_index;
    while (i <= end_index) {
      // search pattern from patterns
      bool found = false;
      AtomicOpType type = ATOMIC_ANONYMOUS;
      for (auto spec : specs) {
        if (i + spec.size() - 1 > end_index) continue;
        size_t ii = i;
        bool matched = true;
        type = spec.type();
        // add ii <= end_index and ii < end_index to avoid array bounds overflow.
        for (size_t j = 0; j < spec.size() && ii <= end_index; j++) {
          ObjectRef node_ref = GetRef<ObjectRef>((*p_graph_)[ii]->tvm_node);
          if (!is_matched(node_ref, spec[j], type, device_type, i == ii)) {
            matched = false;
            break;
          }
          ii += 1;
        }
        if (matched) {
#ifdef FUSIONS_DEBUG_1
          std::ostringstream os;
          os << "Matched for spec:";
          for (size_t t = 0; t < spec.size(); t++) {
            os << " " << spec[t];
          }
          LOG(INFO) << os.str();
#endif
          size_t end = ii - 1;
          AtomicOp* atomic_op = arena_->make<AtomicOp>(i, end, type, device_type);
          ops.push_back(atomic_op);
          i = ii;
          found = true;
          break;
        }
      }
      if (!found) {
        if (enable_anonymous) {
#if 0
          if (anonymous_op.start == -1) anonymous_op.start = i;
#else
#ifdef FUSIONS_DEBUG_1
          LOG(INFO) << "Matched for enable_anonymous pattern";
#endif
          type = ATOMIC_ANONYMOUS;
          AtomicOp* anonymous_op = arena_->make<AtomicOp>(i, i, type, device_type);
          ops.push_back(anonymous_op);
#endif
          i++;
        } else {
          LOG(FATAL) << device_type << ":Cannot match pattern from index=" << i;
        }
      }
    }

#if 0
    for (size_t k = 0; k < ops.size(); k++) {
      auto r = ops[k];
      LOG(INFO) << "ops range " << k << ": [" << r->start << ", " << r->end << "]";
    }
#endif
    return ops;
  }

  std::vector<AtomicOpSpec> generate_nnp200_atomic_op_specs() {
    std::vector<AtomicOpSpec> atomic_op_specs;
    // Pattern rules:
    // - empty string must be in the end of the array to make sure longest matching.
    // - put high priority patterns of a ATOMIC_OP ahead.
    // - make long specs unique(less "") to reduce search space.
    // ATOMIC_CONV  + ATOMIC_ELTWISE
    add_atomic_op_specs(&atomic_op_specs, ATOMIC_CONV,
                        {{"nn.conv2d", "nn.conv2d_transpose"},
                         {"nn.bias_add"},
                         {"cast"},
                         {"multiply"},
                         {"right_shift", "round_right_shift"},
                         {"clip"},
                         {"cast"},
                         {"nn.relu", ""}});
    add_atomic_op_specs(&atomic_op_specs, ATOMIC_CONV,
                        {{"cast", ""},
                         {"nn.conv2d", "nn.conv2d_transpose"},
                         {"nn.bias_add", ""},
                         {"right_shift", "round_right_shift", ""},
                         {"clip", ""},
                         {"tuple", "tuple_get", ""},
                         {"nn.relu", "nn.prelu", "nn.leaky_relu"},
                         {"clip", ""}});
    add_atomic_op_specs(&atomic_op_specs, ATOMIC_CONV, {{"nn.conv2d"}});
    // QAT: [cast-sum_pool2d-multiply]
    add_atomic_op_specs(&atomic_op_specs, ATOMIC_POOL,
                        {{"cast"}, {"sum", ""}, {"nn.sum_pool2d"}, {"multiply"}});
    // ATOMIC_POOL + ATOMIC_ELTWISE
    add_atomic_op_specs(
        &atomic_op_specs, ATOMIC_POOL,
        {{"nn.max_pool2d", "nn.global_max_pool2d", "nn.global_avg_pool2d", "nn.sum_pool2d"},
         {"cast"},
         {"multiply"},
         {"round_right_shift"},
         {"clip"},
         {"cast"}});
    add_atomic_op_specs(&atomic_op_specs, ATOMIC_POOL,
                        {{"nn.max_pool2d", "nn.global_max_pool2d", "nn.global_avg_pool2d",
                          "nn.sum_pool2d", "nn.avg_pool2d"}});
    add_atomic_op_specs(&atomic_op_specs, ATOMIC_ADAPTIVEPOOL, {{"nn.adaptive_avg_pool2d"}});
    add_atomic_op_specs(&atomic_op_specs, ATOMIC_ELTWISE,
                        {{"cast"},
                         {"cast"},
                         {"add", "divide"},
                         {"cast"},
                         {"add", "subtract", "multiply"},
                         {"right_shift", "round_right_shift"},
                         {"clip"},
                         {"cast"},
                         {"nn.relu", "nn.prelu", "nn.leaky_relu", ""}});
    add_atomic_op_specs(&atomic_op_specs, ATOMIC_ELTWISE,
                        {{"cast", ""},
                         {"sum", ""},
                         {"add", "subtract", "multiply"},
                         {"nn.bias_add", ""},
                         {"right_shift", "round_right_shift", ""},
                         {"add", ""},
                         {"clip", ""},
                         {"cast", ""}});
    add_atomic_op_specs(&atomic_op_specs, ATOMIC_BIASADD,
                        {{"nn.bias_add"}, {"right_shift", ""}, {"clip", ""}});
    add_atomic_op_specs(&atomic_op_specs, ATOMIC_RIGHTSHIFT, {{"right_shift", ""}, {"clip", ""}});
    add_atomic_op_specs(&atomic_op_specs, ATOMIC_RELU, {{"nn.relu"}, {"clip", ""}});
    add_atomic_op_specs(
        &atomic_op_specs, ATOMIC_PRELU,
        {{"nn.prelu", "nn.leaky_relu"}, {"nn.relu", "right_shift", ""}, {"clip", ""}});
    add_atomic_op_specs(&atomic_op_specs, ATOMIC_TRANSPOSE,
                        {{"transpose"}, {"nn.batch_flatten", ""}, {"right_shift", ""}});
    add_atomic_op_specs(&atomic_op_specs, ATOMIC_BATCHFLATTEN, {{"nn.batch_flatten"}});
    add_atomic_op_specs(&atomic_op_specs, ATOMIC_SQUEEZE, {{"squeeze"}});
    add_atomic_op_specs(&atomic_op_specs, ATOMIC_SLICE, {{"strided_slice"}, {"clip", ""}});
    add_atomic_op_specs(&atomic_op_specs, ATOMIC_SOFTMAX, {{"nn.softmax"}});
    add_atomic_op_specs(&atomic_op_specs, ATOMIC_CAST, {{"cast"}});
    add_atomic_op_specs(&atomic_op_specs, ATOMIC_CONCAT, {{"tuple", "tuple_get"}, {"concatenate"}});
    add_atomic_op_specs(&atomic_op_specs, ATOMIC_CONCAT, {{"concatenate"}});
    add_atomic_op_specs(&atomic_op_specs, ATOMIC_TUPLE, {{"tuple", "tuple_get"}});
    add_atomic_op_specs(&atomic_op_specs, ATOMIC_RESHAPE, {{"reshape"}});
    add_atomic_op_specs(&atomic_op_specs, ATOMIC_RESHAPE, {{"expand_dims"}});
    add_atomic_op_specs(&atomic_op_specs, ATOMIC_DEVICECOPY, {{"device_copy"}});
    add_atomic_op_specs(&atomic_op_specs, ATOMIC_DENSE, {{"nn.dense"}, {"nn.bias_add", ""}});
    add_atomic_op_specs(&atomic_op_specs, ATOMIC_PAD, {{"nn.pad"}});
    return std::move(atomic_op_specs);
  }

  bool is_op_of_name(EdgexDependencyGraph::Node* node, std::string op_name) {
    bool ret = false;
    if (auto callnode = GetRef<ObjectRef>(node->tvm_node).as<CallNode>()) {
      if (callnode->op.same_as(Op::Get(op_name))) {
        ret = true;
      }
    }
    return ret;
  }

  void genearete_atomic_ops() {
    std::vector<size_t> serial_relay_ops;
    std::unordered_map<size_t, std::set<size_t>> relay_op_id_to_input_atomic_op_ids;
    for (size_t i = 0; i < p_graph_->size(); i++) {
#ifdef FUSIONS_DEBUG_2
      LOG(INFO) << "--Collect i = " << i << " device_type = " << dgraph_op_device_types_[i]
                << " detail: " << (*p_graph_)[i]->detail();
#endif
      serial_relay_ops.push_back(i);
      bool is_end = false;
      if ((*p_graph_)[i]->parents_size() == 1) {
        auto out_graph_node = (*p_graph_)[i]->parents.head->value;
        auto next_graph_node = (*p_graph_)[i + 1];
        // since we need to fuse Op::add and its operands in a row
        if (out_graph_node != next_graph_node && !is_op_of_name(out_graph_node, "add")) {
          is_end = true;
#ifdef FUSIONS_DEBUG_1
          LOG(INFO) << "out_graph_node = " << out_graph_node->index
                    << " next_graph_node = " << next_graph_node->index;
#endif
        } else if (dgraph_op_device_types_[i] != dgraph_op_device_types_[next_graph_node->index]) {
          is_end = true;
        } else if (next_graph_node->children_size() > 1 && !is_op_of_name(out_graph_node, "add")) {
          is_end = true;
        }

        if (is_op_of_name(out_graph_node, "vision.non_max_suppression")) {
          is_end = false;
        }
      } else {
        is_end = true;
        if (is_op_of_name((*p_graph_)[i], "vision.multibox_transform_loc")) {
          is_end = false;
        }
      }

      if (is_end) {
        size_t start_index = serial_relay_ops.front();
        size_t end_index = serial_relay_ops.back();
        std::vector<AtomicOp*> serial_atomic_ops;
        auto dev_type = dgraph_op_device_types_[start_index];
        bool anonymous_en = (dev_type == OP_DEVICE_DEDSP || dev_type == OP_DEVICE_NNP300);
        serial_atomic_ops =
            genearete_serial_atomic_ops(start_index, end_index, dev_type, anonymous_en);

#ifdef FUSIONS_DEBUG_1
        LOG(INFO) << "-- genearete atomic_op [" << post_dfs_atomic_ops_.size() << ", "
                  << post_dfs_atomic_ops_.size() + serial_atomic_ops.size() - 1 << "]"
                  << " relay_id[" << start_index << ", " << end_index << "]";
#endif

        // set index for serial_atomic_ops
        for (size_t j = 0; j < serial_atomic_ops.size(); j++) {
          serial_atomic_ops[j]->index = post_dfs_atomic_ops_.size() + j;
        }

        // construct ProdcuerConsumer relation for ops of serial_atomic_ops
        for (auto atomic_op : serial_atomic_ops) {
          for (size_t relay_id = atomic_op->start; relay_id <= atomic_op->end; relay_id++) {
#ifdef FUSIONS_DEBUG_2
            LOG(INFO) << "check input atomic op of atomic_op[" << atomic_op->index << "] relay_id ["
                      << relay_id << "]";
#endif
            if (relay_op_id_to_input_atomic_op_ids.count(relay_id)) {
              for (size_t input_atomic_id : relay_op_id_to_input_atomic_op_ids[relay_id]) {
#ifdef FUSIONS_DEBUG_2
                LOG(INFO) << "use relay_op_id_to_input_atomic_op_ids: relay_id[" << relay_id
                          << "] to atomic_id[" << input_atomic_id << "]";
#endif
                auto input_atomic_op = post_dfs_atomic_ops_[input_atomic_id];
                if (std::find(atomic_op->inputs.begin(), atomic_op->inputs.end(),
                              input_atomic_op) == atomic_op->inputs.end()) {
                  atomic_op->inputs.push_back(input_atomic_op);
                }
                if (std::find(input_atomic_op->outputs.begin(), input_atomic_op->outputs.end(),
                              atomic_op) == input_atomic_op->outputs.end()) {
                  input_atomic_op->outputs.push_back(atomic_op);
                }
              }
            }
          }
        }

        // update relay_op_id_to_input_atomic_op_ids for end op of serial_atomic_ops
        auto cur_end_graph_op = (*p_graph_)[serial_atomic_ops.back()->end];
        size_t cur_end_atomic_op_id = post_dfs_atomic_ops_.size() + serial_atomic_ops.size() - 1;
        for (auto output = cur_end_graph_op->parents.head; output != nullptr;
             output = output->next) {
          size_t relay_index = output->value->index;
          if (relay_op_id_to_input_atomic_op_ids.count(relay_index) == 0) {
            relay_op_id_to_input_atomic_op_ids[relay_index] = std::set<size_t>();
          }
#ifdef FUSIONS_DEBUG_2
          LOG(INFO) << "insert relay_op_id_to_input_atomic_op_ids: relay_id[" << relay_index
                    << "] to atomic_id[" << cur_end_atomic_op_id << "]";
#endif
          relay_op_id_to_input_atomic_op_ids[relay_index].insert(cur_end_atomic_op_id);
        }

        // construct ProdcuerConsumer relation for internal ops of serial_atomic_ops
        for (size_t j = 0; j < serial_atomic_ops.size() - 1; j++) {
          auto tmp_atomic_op = serial_atomic_ops[j];
          auto next_atomic_op = serial_atomic_ops[j + 1];
          tmp_atomic_op->outputs.push_back(next_atomic_op);
          next_atomic_op->inputs.push_back(tmp_atomic_op);
        }

        post_dfs_atomic_ops_.insert(post_dfs_atomic_ops_.end(), serial_atomic_ops.begin(),
                                    serial_atomic_ops.end());
        serial_relay_ops.clear();
      }
    }
  }
};

class FusionPlanGenerator {
 public:
  std::set<std::set<size_t>> Run(const AtomicGraph& atomic_graph) {
    std::set<std::set<size_t>> atomic_op_patterns;
    GenerateOldPatterns(atomic_graph, &atomic_op_patterns);
    // GenerateSubstitutionFusionPatterns(atomic_graph, &atomic_op_patterns);
    // GenerateExploratoryFusionPatterns(atomic_graph, &atomic_op_patterns);
    return atomic_op_patterns;
  }

 private:
  bool is_stride_gt_1(Array<IndexExpr> strides) {
    for (auto stride : strides) {
      if (analyzer_.CanProve(stride < 2)) {
        return false;
      }
    }
    return true;
  }

  bool is_oprange_stride_gt_1(const EdgexDependencyGraph& graph, AtomicGraph::AtomicOp op) {
    for (size_t i = op.start; i <= op.end; i++) {
      auto graph_node = graph[i];
      // auto group_node = groups_[i];
      if (auto* callnode = GetRef<ObjectRef>(graph_node->tvm_node).as<CallNode>()) {
        if (callnode->op.as<OpNode>()) {
          // collect stride info
          Array<IndexExpr> strides;
          if (callnode->op.same_as(Op::Get("nn.conv2d"))) {
            strides = callnode->attrs.as<Conv2DAttrs>()->strides;
          } else if (callnode->op.same_as(Op::Get("nn.max_pool2d"))) {
            strides = callnode->attrs.as<MaxPool2DAttrs>()->strides;
          }
          if (strides.size() > 0 && is_stride_gt_1(strides)) {
            // LOG(INFO) << "is_oprange_stride_gt_1=true";
            return true;
          }
        }
      }
    }
    // LOG(INFO) << "is_oprange_stride_gt_1=false";
    return false;
  }

  bool needs_eodma_stride(const EdgexDependencyGraph& graph, AtomicGraph::AtomicOp op) {
    for (size_t i = op.start; i <= op.end; i++) {
      auto graph_node = graph[i];
      if (auto* callnode = GetRef<ObjectRef>(graph_node->tvm_node).as<CallNode>()) {
        if (callnode->op.as<OpNode>()) {
          // collect stride info
          Array<IndexExpr> strides;
          if (callnode->op.same_as(Op::Get("nn.max_pool2d"))) {
            strides = callnode->attrs.as<MaxPool2DAttrs>()->strides;
          }
          if (strides.size() > 0 && analyzer_.CanProve(strides[1] > 2)) {
            return true;
          }
        }
      }
    }
    return false;
  }

  bool is_oprange_data_oversize(const EdgexDependencyGraph& graph, AtomicGraph::AtomicOp op) {
    int pad_index = -1;
    Array<IndexExpr> paddings({0, 0, 0, 0});  // top/left/bottom/right
    // auto func_data_oversize = runtime::Registry::Get("edgex.is_data_oversize");
    auto func_cm_size = runtime::Registry::Get("");
    auto func_cm_oversize = runtime::Registry::Get("edgex.is_cm_oversize");
    // auto func_get_co_gp_size = runtime::Registry::Get("edgex.get_co_gp_size");
    int total_cm_size = 0;
    bool has_bias = false;
    bool has_right_shift = false;
    Array<Array<IndexExpr>> master_input_shapes;
    Array<IndexExpr> master_output_shape;
    Array<IndexExpr> master_ibuf_insertions;
    Attrs master_attrs;
    for (size_t i = op.start; i <= op.end; i++) {
      auto graph_node = graph[i];
      if (auto* callnode = GetRef<ObjectRef>(graph_node->tvm_node).as<CallNode>()) {
        if (callnode->op.as<OpNode>()) {
          if (callnode->op.same_as(Op::Get("nn.pad"))) {
            pad_index = i;
            auto attrs = callnode->attrs.as<PadAttrs>();
            auto first_paddings = attrs->pad_width;
            paddings.Set(0, first_paddings[2][0]);  // top
            paddings.Set(1, first_paddings[3][0]);  // left
            paddings.Set(2, first_paddings[2][1]);  // bottom
            paddings.Set(3, first_paddings[3][1]);  // right
          } else if (callnode->op.same_as(Op::Get("nn.conv2d")) ||
                     callnode->op.same_as(Op::Get("nn.conv2d_transpose")) ||
                     callnode->op.same_as(Op::Get("nn.max_pool2d")) ||
                     callnode->op.same_as(Op::Get("nn.sum_pool2d")) ||
                     callnode->op.same_as(Op::Get("nn.global_sum_pool2d")) ||
                     callnode->op.same_as(Op::Get("add"))) {
            // handle input_shape and output_shape
            Array<Array<IndexExpr>> input_shapes;
            if (pad_index == -1) {
              auto input_shape = callnode->args[0]->checked_type_.as<TensorTypeNode>()->shape;
              input_shapes.push_back(input_shape);
            } else {
              auto pad_graph_node = graph[pad_index];
              auto pad_callnode = GetRef<ObjectRef>(pad_graph_node->tvm_node).as<CallNode>();
              auto input_shape = pad_callnode->args[0]->checked_type_.as<TensorTypeNode>()->shape;
              input_shapes.push_back(input_shape);
            }
            if (callnode->op.same_as(Op::Get("add"))) {
              Array<IndexExpr> input_shape1 =
                  callnode->args[1]->checked_type_.as<TensorTypeNode>()->shape;
              input_shapes.push_back(input_shape1);
            }
            auto output_shape = callnode->checked_type_.as<TensorTypeNode>()->shape;
            // handle ibuf_insertions
            Array<IndexExpr> ibuf_insertions =
                Array<IndexExpr>({0, 0, 0, 0, 0, 0});  // top/left/bottom/right/h/w
            if (analyzer_.CanProve(output_shape[2] > input_shapes[0][2]) &&
                analyzer_.CanProve(output_shape[3] > input_shapes[0][3])) {
              Array<IndexExpr> current_padding({0, 0});  // height/width
              if (callnode->op.same_as(Op::Get("nn.conv2d"))) {
                auto attrs = callnode->attrs.as<Conv2DAttrs>();
                current_padding = attrs->padding;
              } else if (callnode->op.same_as(Op::Get("nn.max_pool2d"))) {
                auto attrs = callnode->attrs.as<MaxPool2DAttrs>();
                current_padding = attrs->padding;
              } else if (callnode->op.same_as(Op::Get("nn.conv2d_transpose"))) {
                auto attrs = callnode->attrs.as<Conv2DTransposeAttrs>();
                auto d_paddings = attrs->padding;
                auto d_kernels = attrs->kernel_size;
                auto d_strides = attrs->strides;
                current_padding.Set(0, d_kernels[0] - d_paddings[0] - 1);
                current_padding.Set(1, d_kernels[1] - d_paddings[1] - 1);
                ibuf_insertions.Set(4, d_strides[0] - 1);
                ibuf_insertions.Set(5, d_strides[1] - 1);
              }
              ibuf_insertions.Set(0, paddings[0] + current_padding[0]);
              ibuf_insertions.Set(2, paddings[2] + current_padding[0]);
              ibuf_insertions.Set(1, paddings[1] + current_padding[1]);
              ibuf_insertions.Set(3, paddings[3] + current_padding[1]);
            }

            master_input_shapes = input_shapes;
            master_output_shape = output_shape;
            master_attrs = callnode->attrs;
            master_ibuf_insertions = ibuf_insertions;
          } else if (callnode->op.same_as(Op::Get("nn.bias_add"))) {
            Array<IndexExpr> weight_shape =
                callnode->args[1]->checked_type_.as<TensorTypeNode>()->shape;
            const auto output_shape = callnode->checked_type_.as<TensorTypeNode>()->shape;
            int cm_size = (*func_cm_size)(weight_shape, master_input_shapes, master_output_shape,
                                          master_attrs, master_ibuf_insertions, "nn.bias_add");
            total_cm_size += cm_size;
            has_bias = true;
          } else if (callnode->op.same_as(Op::Get("right_shift"))) {
            if (!has_right_shift) {
              if (master_input_shapes.empty() && master_output_shape.empty()) {
                Array<IndexExpr> ibuf_insertions =
                    Array<IndexExpr>({0, 0, 0, 0, 0, 0});  // top/left/bottom/right/h/w
                Array<Array<IndexExpr>> input_shapes;
                auto input_shape = callnode->args[0]->checked_type_.as<TensorTypeNode>()->shape;
                auto output_shape = callnode->checked_type_.as<TensorTypeNode>()->shape;
                input_shapes.push_back(input_shape);
                master_input_shapes = input_shapes;
                master_output_shape = output_shape;
                master_attrs = callnode->attrs;
                master_ibuf_insertions = ibuf_insertions;
              }
              Array<IndexExpr> weight_shape =
                  callnode->args[1]->checked_type_.as<TensorTypeNode>()->shape;
              const auto output_shape = callnode->checked_type_.as<TensorTypeNode>()->shape;
              int cm_size = (*func_cm_size)(weight_shape, master_input_shapes, master_output_shape,
                                            master_attrs, master_ibuf_insertions, "right_shift");
              total_cm_size += cm_size;
              has_right_shift = true;
            }
          } else if (callnode->op.same_as(Op::Get("nn.prelu"))) {
            if (!has_bias) {
              Array<IndexExpr> weight_shape =
                  callnode->args[1]->checked_type_.as<TensorTypeNode>()->shape;
              const auto output_shape = callnode->checked_type_.as<TensorTypeNode>()->shape;
              int cm_size = (*func_cm_size)(weight_shape, master_input_shapes, master_output_shape,
                                            master_attrs, master_ibuf_insertions, "nn.prelu");
              total_cm_size += cm_size;
            }
          }
        }
      }
    }
    if ((*func_cm_oversize)(total_cm_size)) return true;
    return false;
  }

  bool need_cut_behind(AtomicOpType op_type) {
    if (op_type == ATOMIC_VU_ELTWISE || op_type == ATOMIC_VU_POOL || op_type == ATOMIC_SLICE ||
        op_type == ATOMIC_VISION || op_type == ATOMIC_TRANSPOSE || op_type == ATOMIC_STN ||
        op_type == ATOMIC_SOFTRELU || op_type == ATOMIC_SOFTMAX || op_type == ATOMIC_CAST ||
        op_type == ATOMIC_ANONYMOUS || op_type == ATOMIC_ADAPTIVEPOOL ||
        op_type == ATOMIC_L2NORMALIZE || op_type == ATOMIC_PAD) {
      return true;
    } else {
      return false;
    }
  }

  void GenerateOldPatterns(const AtomicGraph& atomic_graph,
                           std::set<std::set<size_t>>* atomic_op_patterns) {
    std::vector<size_t> serial_set;
    int split_cnt = 0;
    bool multi_outs_graph_en = false;
    if (atomic_graph.device_type() == kDLEdgeX) {
#ifdef ENABLE_SPLIT_GRAPH
      multi_outs_graph_en = true;
#endif
    }
    for (size_t i = 0; i < atomic_graph.size(); i++) {
      serial_set.push_back(i);
      auto atomic_op = atomic_graph[i];
      bool handle_serial_ops = false;
      if (atomic_op->outputs.size() == 1) {
        auto output_atomic_op = atomic_op->outputs[0];
        auto next_atomic_op = atomic_graph[i + 1];
        if (atomic_op->device != output_atomic_op->device) {
          handle_serial_ops = true;
        } else if (output_atomic_op != next_atomic_op && !multi_outs_graph_en) {
          handle_serial_ops = true;
        } else if (atomic_op->device == OP_DEVICE_DEDSP) {
          handle_serial_ops = true;
        } else if (atomic_op->device == OP_DEVICE_NNP300 && need_cut_behind(atomic_op->type)) {
          handle_serial_ops = true;
        } else if (next_atomic_op->type == ATOMIC_TUPLE) {
          handle_serial_ops = true;
        } else {
          auto p_graph = atomic_graph.get_dependency_graph();
          auto cur_device = atomic_op->device;
          if (needs_eodma_stride(*p_graph, *atomic_op) ||
              (is_oprange_stride_gt_1(*p_graph, *atomic_op) && serial_set.size() > 2 &&
               (cur_device == OP_DEVICE_EDGEX))) {
            handle_serial_ops = true;
          }
        }

        auto cur_type = atomic_op->type;
        auto cur_device = atomic_op->device;
        if (cur_device == OP_DEVICE_EDGEX || cur_device == OP_DEVICE_NNP300) {
          if (cur_type == ATOMIC_BATCHFLATTEN || cur_type == ATOMIC_RESHAPE ||
              cur_type == ATOMIC_CONCAT) {
            handle_serial_ops = true;
          }
        }
        if (cur_type == ATOMIC_TUPLE || cur_type == ATOMIC_DEVICECOPY) {
          handle_serial_ops = true;
        }

        auto next_type = atomic_graph[i + 1]->type;
        auto next_device = atomic_graph[i + 1]->device;
        if (next_device == OP_DEVICE_EDGEX || cur_device == OP_DEVICE_NNP300) {
          if (next_type == ATOMIC_BATCHFLATTEN) {
            handle_serial_ops = true;
          }
        }
        if (next_type == ATOMIC_SQUEEZE || next_type == ATOMIC_RESHAPE ||
            next_type == ATOMIC_CONCAT || next_type == ATOMIC_ADAPTIVEPOOL) {
          handle_serial_ops = true;
        }

      } else if (atomic_op->outputs.size() == 2 && multi_outs_graph_en) {
        split_cnt += 1;
        if (split_cnt >= 3) {
          handle_serial_ops = true;
          split_cnt = 0;
        } else {
          continue;
        }
      } else {  // atomic_op->outputs.size() != 1
        handle_serial_ops = true;
      }

      if (handle_serial_ops) {
        split_cnt = 0;
        size_t start = serial_set.front();
        size_t len = serial_set.size();
        for (size_t l = 1; l <= len; l++) {
          for (size_t s = start; s <= start + l - 1; s++) {
            std::set<size_t> tmp_set;
            size_t e = start + l - 1;
            for (size_t j = s; j <= e; j++) {
              tmp_set.insert(j);
            }
            atomic_op_patterns->insert(tmp_set);
          }
        }
        serial_set.clear();
      }
    }
  }

  void GenerateSubstitutionFusionPatterns(const AtomicGraph& atomic_graph,
                                          std::set<std::set<size_t>>* atomic_op_patterns) {
    auto algorithm = [&](std::set<size_t> partition_atomic_ops) {
      std::set<size_t> tmp_set;
      for (size_t i = 0; i < atomic_graph.size(); i++) {
        if (partition_atomic_ops.count(i)) {
          if (tmp_set.size() > 0) {
            atomic_op_patterns->insert(tmp_set);
            tmp_set.clear();
          }
        } else {
          tmp_set.insert(i);
        }
      }
      if (tmp_set.size() > 0) {
        atomic_op_patterns->insert(tmp_set);
        tmp_set.clear();
      }
    };

    std::set<size_t> partition_atomic_ops;
    for (size_t i = 0; i < atomic_graph.size(); i++) {
      if (atomic_graph[i]->type == ATOMIC_CONV) {
        partition_atomic_ops.insert(i);
      }
    }
    algorithm(partition_atomic_ops);
  }

  void algorithm_explore(const AtomicGraph& atomic_graph, std::set<size_t>* seed_pattern,
                         std::set<std::set<size_t>>* atomic_op_patterns) {
    auto can_fuse = [](size_t c) -> bool { return true; };  // TODO(cww)

    std::set<size_t> candidates;
    // ProducerExpansion and ConsumerExpansion
    for (size_t i : *seed_pattern) {
      if (atomic_graph[i]->inputs.size() == 1) {
        for (auto input : atomic_graph[i]->inputs) {
          if (seed_pattern->count(input->index)) continue;
          if (input->type == ATOMIC_CONV || input->type == ATOMIC_ELTWISE ||
              input->type == ATOMIC_REDUCTION) {
            if (input->outputs.size() == 1) {
              candidates.insert(input->index);
            }
          }
        }
      }

      if (atomic_graph[i]->outputs.size() == 1) {
        for (auto output : atomic_graph[i]->outputs) {
          if (seed_pattern->count(output->index)) continue;
          if (output->type == ATOMIC_CONV || output->type == ATOMIC_ELTWISE ||
              output->type == ATOMIC_REDUCTION) {
            if (output->inputs.size() == 1) {
              candidates.insert(output->index);
            }
          }
        }
      }
    }

    // run algorithm_explore iteratively
    for (size_t c : candidates) {
      if (can_fuse(c)) {
        seed_pattern->insert(c);
        if (atomic_op_patterns->count(*seed_pattern) == 0) {
          atomic_op_patterns->insert(*seed_pattern);
          algorithm_explore(atomic_graph, seed_pattern, atomic_op_patterns);
        }
        seed_pattern->erase(c);
      }
    }
  }

  void GenerateExploratoryFusionPatterns(const AtomicGraph& atomic_graph,
                                         std::set<std::set<size_t>>* atomic_op_patterns) {
    std::set<std::set<size_t>> seed_patterns;
    for (size_t i = 0; i < atomic_graph.size(); i++) {
      if (atomic_graph[i]->type == ATOMIC_CONV) {
        std::set<size_t> seed_pattern{i};
        seed_patterns.insert(seed_pattern);
      }
    }

    for (auto seed_pattern : seed_patterns) {
      atomic_op_patterns->insert(seed_pattern);  // TODO(cww)
      algorithm_explore(atomic_graph, &seed_pattern, atomic_op_patterns);
    }
  }

  arith::Analyzer analyzer_;
};

size_t IdSetHash(const std::set<size_t>& id_set) {
  std::ostringstream os;
  for (size_t id : id_set) {
    os << "_" << id;
  }
  return std::hash<std::string>()(os.str());
}

std::string format_patterns(const std::vector<std::set<size_t>>& patterns) {
  std::ostringstream os;
  os << "{";
  for (size_t i = 0; i < patterns.size(); i++) {
    auto pattern = patterns[i];
    os << "{";
    auto vpattern = std::vector<size_t>(pattern.begin(), pattern.end());
    for (size_t j = 0; j < vpattern.size() - 1; j++) {
      os << vpattern[j] << ", ";
    }
    os << vpattern.back() << "}";
    if (i != patterns.size() - 1) {
      os << ", ";
    }
  }
  os << "};";
  return os.str();
}

std::string format_gains(const std::vector<int64_t>& gains) {
  std::ostringstream os;
  os << "{";
  for (size_t i = 0; i < gains.size() - 1; i++) {
    os << gains[i] << ", ";
  }
  os << gains.back() << "};";
  return os.str();
}

class ILPFusionPlanning {
 public:
  ILPFusionPlanning() {}

  std::set<std::set<size_t>> Run(const Expr& func, const AtomicGraph& atomic_graph,
                                 const std::set<std::set<size_t>>& atomic_op_patterns) {
    // compute cost for every single atomic_op
    std::vector<int64_t> single_op_cycles;
    for (size_t i = 0; i < atomic_graph.size(); i++) {
      auto single_op_cycle = evaluate_atomic_op_pattern(func, atomic_graph, {i});
      single_op_cycles.push_back(single_op_cycle);
    }

    // compute cost for every atomic_op_pattern
    std::set<std::set<size_t>> invalid_patterns;
    std::vector<std::set<size_t>> valid_patterns;
    std::vector<int64_t> valid_pattern_cycles;

    for (auto pattern : atomic_op_patterns) {
      int64_t pattern_cycle;
      bool must_be_failed = false;
      for (auto invalid_pattern : invalid_patterns) {
        std::set<size_t> intersection;
        std::set_intersection(pattern.begin(), pattern.end(), invalid_pattern.begin(),
                              invalid_pattern.end(),
                              std::inserter(intersection, intersection.begin()));
        if (intersection == invalid_pattern) {
          must_be_failed = true;
          break;
        }
      }
      if (must_be_failed) {
        pattern_cycle = INT64_MAX;
#ifdef FUSIONS_DEBUG_1
        LOG(INFO) << "must_be_failed = " << TextOfStdSet(pattern);
#endif
      } else {
        pattern_cycle = evaluate_atomic_op_pattern(func, atomic_graph, pattern);
      }
      if (pattern_cycle != INT64_MAX) {
#ifdef FUSIONS_DEBUG_1
        LOG(INFO) << "valid_pattern_cycle = " << pattern_cycle;
#endif
        valid_pattern_cycles.push_back(pattern_cycle);
        valid_patterns.push_back(pattern);
      } else {
#ifdef FUSIONS_DEBUG_1
        LOG(INFO) << "invalid_pattern_cycle = " << pattern_cycle;
#endif
        invalid_patterns.insert(pattern);
      }
    }

    // compute gain for every pattern, and ignore the pattern with negative value
    std::vector<int64_t> pattern_gains =
        compute_gains(single_op_cycles, valid_pattern_cycles, valid_patterns);

#ifdef FUSIONS_DEBUG_0
    LOG(INFO) << "std::vector<std::set<size_t>> valid_patterns = "
              << format_patterns(valid_patterns) << std::endl;
    LOG(INFO) << "std::vector<int64_t> pattern_gains = " << format_gains(pattern_gains)
              << std::endl;
#endif

    // ILP algorithm
    Array<tvm::PrimExpr> pulp_gains;
    for (int64_t gain : pattern_gains) {
      pulp_gains.push_back(tvm::IntImm(DataType::Int(64), gain));
    }

    Array<Array<tvm::PrimExpr>> pulp_conflicts;
    for (size_t i = 0; i < valid_patterns.size(); i++) {
      Array<tvm::PrimExpr> tmp_conflict;
      for (size_t j = 0; j < valid_patterns.size(); j++) {
        int conflict_value = 0;
        for (size_t id : valid_patterns[i]) {
          if (valid_patterns[j].count(id)) {
            conflict_value = 1;
            break;
          }
        }
        tmp_conflict.push_back(tvm::IntImm(DataType::Int(64), conflict_value));
      }
      pulp_conflicts.push_back(tmp_conflict);
    }
    std::string func_name = "edgex.util.pulp_compute";
    const auto* pulp = runtime::Registry::Get(func_name);
    CHECK(pulp) << "Cannot find " << func_name;
    Array<tvm::PrimExpr> pulp_result = (*pulp)(pulp_gains, pulp_conflicts);

#ifdef FUSIONS_DEBUG_1
    LOG(INFO) << "valid_patterns size = " << valid_patterns.size();
    LOG(INFO) << "pulp_result size = " << pulp_result.size();
#endif
    std::set<std::set<size_t>> result;
    for (size_t i = 0; i < valid_patterns.size(); i++) {
      if (pulp_result[i].as<IntImmNode>()->value == 1) {
        // LOG(INFO) << "pattern " << i << ": " << TextOfStdSet(atomic_op_pattern_array2[i]);
        result.insert(valid_patterns[i]);
      }
      // LOG(INFO) << "result " << i << ": " << tmp_result[i];
    }

    return result;
  }

 private:
  int64_t compile_and_evaluate(const Expr& f) {
    try {
      // TODO(yiheng): use single atomic_op temporally
      // unsigned int seed = time(NULL);
      int64_t cycle = INT_MIN;  // rand_r(&seed)%1673492;
      return cycle;
    } catch (...) {
      std::string e = TVMGetLastError();
      if (e.find("Unknown error!") != std::string::npos) {
        throw;
      } else if (e.find("OSError") != std::string::npos) {
        throw;
      } else if (e.empty()) {
        throw;
      } else {
        std::string func_name = "edgex.logging.enable_for_debug";
        auto enable_for_debug = runtime::Registry::Get(func_name);
        if (enable_for_debug) {
          bool is_enabled = (*enable_for_debug)();
          if (is_enabled) {
            LOG(INFO) << "\n[Trivial exception]: " << TVMGetLastError();
          }
        }
      }
      return INT64_MAX;
    }
  }

  std::unordered_map<const std::set<size_t>, int64_t, decltype(&IdSetHash)> cost_cache_{100,
                                                                                        IdSetHash};

  int64_t evaluate_subgraph(const Expr& func, const std::set<size_t>& subgraph_relay_ids,
                            const OpDeviceType device_type) {
    if (cost_cache_.find(subgraph_relay_ids) != cost_cache_.end()) {
#ifdef FUSIONS_DEBUG_1
      LOG(INFO) << "found in cache for subgraph_relay_ids = " << TextOfStdSet(subgraph_relay_ids);
#endif
      return cost_cache_[subgraph_relay_ids];
    }
    auto subgraph_expr = EdgexGraphPartitionByIdSet(func, subgraph_relay_ids);
    std::string func_name = "tvm.edgex.replace_constants";
    const auto* replace_constants = runtime::Registry::Get(func_name);
    CHECK(replace_constants) << "Cannot find " << func_name;
    Function func0 = (*replace_constants)(subgraph_expr);
    func0 = WithAttr(std::move(func0), "Grouped", tvm::Integer(1));
    // TODO(yiheng): fix with S_DEVICE_EDGEX
    func0 = WithAttr(std::move(func0), "DeviceType", tvm::Integer(S_DEVICE_EDGEX));

    int64_t cycle = compile_and_evaluate(func0);
    cost_cache_[subgraph_relay_ids] = cycle;
    return cycle;
  }

  int64_t evaluate_atomic_op_pattern(const Expr& func, const AtomicGraph& atomic_graph,
                                     const std::set<size_t>& pattern) {
    if (pattern.size() == 1) {
#ifdef FUSIONS_DEBUG_2
      LOG(INFO) << "-- evaluate single atomic_op[" << TextOfStdSet(pattern)
                << " ] detail: " << atomic_graph[(*pattern.begin())]->detail();
#endif
      auto type = atomic_graph[*pattern.begin()]->type;
      if (type == ATOMIC_TUPLE) {
#ifdef FUSIONS_DEBUG_2
        LOG(INFO) << "cycle = " << 0;
#endif
        return 0;
      }
    } else {
#ifdef FUSIONS_DEBUG_2
      LOG(INFO) << "-- evaluate atomic_op_pattern: " << TextOfStdSet(pattern);
      for (size_t id : pattern) {
        std::cout << "atomic_op " << id << ": " << atomic_graph[id]->detail() << std::endl;
      }
      LOG(INFO);
#endif
    }

    std::set<size_t> subgraph_ids;
    for (size_t atomic_id : pattern) {
      for (size_t j = atomic_graph[atomic_id]->start; j <= atomic_graph[atomic_id]->end; j++) {
        subgraph_ids.insert(j);
      }
    }

    int64_t cycle = evaluate_subgraph(func, subgraph_ids, atomic_graph[(*pattern.begin())]->device);
#ifdef FUSIONS_DEBUG_2
    LOG(INFO) << "cycle = " << cycle;
#endif
    return cycle;
  }

  std::vector<int64_t> compute_gains(const std::vector<int64_t>& single_op_cycles,
                                     const std::vector<int64_t>& atomic_op_pattern_cycles,
                                     const std::vector<std::set<size_t>>& atomic_op_patterns) {
    std::vector<int64_t> pattern_gains;
    for (size_t i = 0; i < atomic_op_patterns.size(); i++) {
      int64_t gain;
      auto pattern = atomic_op_patterns[i];
      int64_t pattern_cycle = atomic_op_pattern_cycles[i];
      int64_t single_op_total_cycle = 0;
      if (pattern.size() > 1) {
        if (pattern_cycle < 0) {
          gain = INT_MIN;  // current pattern schedule successed but it's invalid
        } else {
          for (size_t atomic_id : pattern) {
            single_op_total_cycle += single_op_cycles[atomic_id];
          }
          gain = single_op_total_cycle - pattern_cycle;
        }
      } else {
        gain = 1;  // TODO(cww)
      }

      pattern_gains.push_back(gain);
#ifdef FUSIONS_DEBUG_1
      LOG(INFO) << "-- compute gain for atomic_op_pattern[" << TextOfStdSet(pattern)
                << " ] gain: " << gain << ", single_op_total_cycle: " << single_op_total_cycle
                << "; pattern_cycle: " << pattern_cycle;
#endif
    }
    return pattern_gains;
  }
};

class SubFunctionFuser : private ExprMutator {
 public:
  // Run the transform
  Expr Transform(const Expr& body, const EdgexDependencyGraph& graph,
                 const std::vector<std::set<size_t>>& subgraph_sets,
                 const std::vector<OpDeviceType>& device_types) {
    Function origin_func = Downcast<Function>(body);

    std::unordered_map<size_t, size_t> node_2_subgraph;
    std::vector<Expr> subgraph_exprs;
    std::vector<Expr> final_subgraph_exprs;
    std::vector<std::vector<size_t>> all_subgraph_starts;
    std::vector<std::vector<size_t>> all_subgraph_ends;

    for (size_t i = 0; i < subgraph_sets.size(); i++) {
      auto subgraph_set = subgraph_sets[i];

#ifdef FUSIONS_DEBUG_2
      LOG(INFO) << "==Collect info: subgraph i = " << i
                << " -- subgraph_set = " << TextOfStdSet(subgraph_set);
#endif

      bool use_empty_expr = false;
      // handle tuple_get and tuple
      if (subgraph_set.size() == 1) {
        if (GetRef<ObjectRef>(graph[*subgraph_set.begin()]->tvm_node).as<TupleGetItemNode>()) {
          use_empty_expr = true;
        }
        if (GetRef<ObjectRef>(graph[*subgraph_set.begin()]->tvm_node).as<TupleNode>()) {
          use_empty_expr = true;
        }
      }

      if (use_empty_expr) {
        subgraph_exprs.push_back(Expr());
      } else {
        auto subgraph_expr = EdgexGraphPartitionByIdSet(body, subgraph_set);
        subgraph_exprs.push_back(subgraph_expr);
#ifdef FUSIONS_DEBUG_2
        LOG(INFO) << AsText(subgraph_expr, false);
#endif
      }

      SubgraphIdSet subgraph_idset = CreateSubgraphIdSet(graph, subgraph_set);
      all_subgraph_starts.push_back(
          std::vector<size_t>(subgraph_idset.starts.begin(), subgraph_idset.starts.end()));
      all_subgraph_ends.push_back(
          std::vector<size_t>(subgraph_idset.ends.begin(), subgraph_idset.ends.end()));

      for (size_t id : subgraph_set) {
        // map from node id to subgraph id
        node_2_subgraph[id] = i;
      }
    }

    final_subgraph_exprs.resize(subgraph_exprs.size());
    for (size_t i = 0; i < subgraph_sets.size(); i++) {
      auto subgraph_set = subgraph_sets[i];
      auto subgraph_func = subgraph_exprs[i];
      auto subgraph_starts = all_subgraph_starts[i];
      auto subgraph_device_type = device_types[i];

#ifdef FUSIONS_DEBUG_2
      LOG(INFO) << "==Transform: subgraph i = " << i;
      LOG(INFO) << "-- subgraph_set = " << TextOfStdSet(subgraph_set);
      if (subgraph_func.defined()) {
        LOG(INFO) << AsText(InferType(subgraph_func), false);
      }
#endif
      // collect input ids of subgraph
      SubgraphIdSet subgraph_idset = CreateSubgraphIdSet(graph, subgraph_set);
      Array<Expr> args = get_argments_of_subgraph(graph, subgraph_idset, node_2_subgraph,
                                                  all_subgraph_ends, final_subgraph_exprs);
      // handle tuple_get and tuple
      if (subgraph_set.size() == 1) {
        if (auto tuple_get =
                GetRef<ObjectRef>(graph[*subgraph_set.begin()]->tvm_node).as<TupleGetItemNode>()) {
          CHECK_EQ(args.size(), 1);
          auto new_tuple_get = TupleGetItem(args[0], tuple_get->index);
          final_subgraph_exprs[i] = new_tuple_get;
          continue;
        }

        if (GetRef<ObjectRef>(graph[*subgraph_set.begin()]->tvm_node).as<TupleNode>()) {
          auto new_tuple = Tuple(args);
          final_subgraph_exprs[i] = new_tuple;
          continue;
        }
      }

      // auto mod = IRModule::FromExpr(subgraph_func);
      // mod = transform::FoldConstant()(mod);
      std::string func_name = "tvm.edgex.replace_constants";
      const auto* replace_constants = runtime::Registry::Get(func_name);
      CHECK(replace_constants) << "Cannot find " << func_name;
      Expr new_expr = (*replace_constants)(subgraph_func);
#ifdef FUSIONS_DEBUG_2
      LOG(INFO) << "-- replace_constants expr" << AsText(InferType(new_expr), false);
#endif
      Function new_func = Downcast<Function>(new_expr);
      new_func = WithAttr(std::move(new_func), "Primitive", tvm::Integer(1));
      int func_dev_type = S_DEVICE_NULL;
      if (subgraph_device_type == OP_DEVICE_EDGEX) {
        func_dev_type = S_DEVICE_EDGEX;
        new_func = WithAttr(std::move(new_func), "Grouped", tvm::Integer(1));
      }
      new_func = WithAttr(std::move(new_func), "DeviceType", tvm::Integer(func_dev_type));
      auto new_call = Call(new_func, args, Attrs(), {});  // TODO(cww)
      final_subgraph_exprs[i] = new_call;
    }

    auto new_func = Function(origin_func->params, final_subgraph_exprs[subgraph_sets.size() - 1],
                             origin_func->ret_type, {});
    LOG(INFO) << "[FusionS] Engine Done";
    return new_func;
  }

 private:
  void collect_input_ids_by_post_dfs(const EdgexDependencyGraph& graph,
                                     const SubgraphIdSet& subgraph_ids,
                                     std::set<size_t>* p_visited_ids,
                                     std::vector<size_t>* p_input_ids, size_t cur_id) {
    if (p_visited_ids->count(cur_id) > 0) return;
    if (std::find(subgraph_ids.all.begin(), subgraph_ids.all.end(), cur_id) ==
        subgraph_ids.all.end()) {
      if (std::find(p_input_ids->begin(), p_input_ids->end(), cur_id) == p_input_ids->end()) {
        p_input_ids->push_back(cur_id);
      }
      return;
    }

    auto graph_node = graph[cur_id];
    for (auto p = graph_node->children.head; p != nullptr; p = p->next) {
      if (p->value->index != SIZE_MAX) {
        collect_input_ids_by_post_dfs(graph, subgraph_ids, p_visited_ids, p_input_ids,
                                      p->value->index);
      }
    }
    p_visited_ids->insert(cur_id);
  }

  std::vector<size_t> get_input_ids_of_subgraph(const EdgexDependencyGraph& graph,
                                                const SubgraphIdSet& subgraph_ids) {
    std::vector<size_t> input_ids;
    std::set<size_t> visited_ids;
    for (size_t index = 0; index < subgraph_ids.ends.size(); index++) {
      size_t end_id = subgraph_ids.ends[index];
      collect_input_ids_by_post_dfs(graph, subgraph_ids, &visited_ids, &input_ids, end_id);
    }
    return input_ids;
  }

  void collect_argments_by_post_dfs(const EdgexDependencyGraph& graph,
                                    const SubgraphIdSet& subgraph_ids,
                                    const std::unordered_map<size_t, size_t>& node_2_subgraph,
                                    const std::vector<std::vector<size_t>>& all_subgraph_ends,
                                    const std::vector<Expr>& final_subgraph_exprs,
                                    std::set<size_t>* p_visited_ids, std::vector<Expr>* p_argments,
                                    size_t cur_id) {
    if (p_visited_ids->count(cur_id) > 0) return;

    // save argment which is not var and constant
    if (std::find(subgraph_ids.all.begin(), subgraph_ids.all.end(), cur_id) ==
        subgraph_ids.all.end()) {
      CHECK(node_2_subgraph.find(cur_id) != node_2_subgraph.end()) << "OP not recognized in graph.";
      size_t subgraph_id = node_2_subgraph.at(cur_id);
      auto subgraph_expr = final_subgraph_exprs[subgraph_id];
      auto sub_graph_ends = all_subgraph_ends[subgraph_id];
      if (sub_graph_ends.size() > 1) {
        auto id_iter = std::find(sub_graph_ends.begin(), sub_graph_ends.end(), cur_id);
        if (id_iter != sub_graph_ends.end()) {
          int id_index = std::distance(sub_graph_ends.begin(), id_iter);
          auto arg_ = tvm::relay::TupleGetItem(subgraph_expr, id_index);
          if (std::find(p_argments->begin(), p_argments->end(), arg_) == p_argments->end()) {
            p_argments->push_back(arg_);
          }
        }
      } else {
        if (std::find(p_argments->begin(), p_argments->end(), subgraph_expr) == p_argments->end()) {
          p_argments->push_back(subgraph_expr);
        }
      }
      p_visited_ids->insert(cur_id);
      return;
    }

    auto graph_node = graph[cur_id];
    auto the_expr = GetRef<ObjectRef>(graph_node->tvm_node);
    Array<Expr> args_of_start;
    if (auto call_node = the_expr.as<CallNode>()) {
      args_of_start = call_node->args;
    } else if (auto tuple_node = the_expr.as<TupleNode>()) {
      args_of_start = tuple_node->fields;
    } else if (auto tuple_get_node = the_expr.as<TupleGetItemNode>()) {
      args_of_start.push_back(tuple_get_node->tuple);
    } else {
      CHECK(0) << "Unsupported relay node type: " << the_expr->_type_key;
    }

    for (auto arg : args_of_start) {
      if ((arg.as<ConstantNode>() && arg.as<ConstantNode>()->is_scalar())) continue;
      if (arg.as<VarNode>() || arg.as<ConstantNode>()) {
        if (std::find(p_argments->begin(), p_argments->end(), arg) == p_argments->end()) {
          p_argments->push_back(arg);
        }
      } else {
        auto p = graph_node->children.head;
        for (; p != nullptr; p = p->next) {
          if (p->value->tvm_node == arg.get()) {
            break;
          }
        }
        collect_argments_by_post_dfs(graph, subgraph_ids, node_2_subgraph, all_subgraph_ends,
                                     final_subgraph_exprs, p_visited_ids, p_argments,
                                     p->value->index);
      }
    }
    p_visited_ids->insert(cur_id);
  }

  Array<Expr> get_argments_of_subgraph(const EdgexDependencyGraph& graph,
                                       const SubgraphIdSet& subgraph_ids,
                                       const std::unordered_map<size_t, size_t>& node_2_subgraph,
                                       const std::vector<std::vector<size_t>>& all_subgraph_ends,
                                       const std::vector<Expr>& final_subgraph_exprs) {
    std::vector<Expr> argments;
    std::set<size_t> visited_ids;
    for (size_t index = 0; index < subgraph_ids.ends.size(); index++) {
      size_t end_id = subgraph_ids.ends[index];
      collect_argments_by_post_dfs(graph, subgraph_ids, node_2_subgraph, all_subgraph_ends,
                                   final_subgraph_exprs, &visited_ids, &argments, end_id);
    }
    return argments;
  }
};

Expr FusionStitch(const Expr& expr, int device_type, int fuse_opt_level, const IRModule& m) {
  auto func = InferType(expr);

  LOG(INFO) << "[FusionS] Start Engine";
#ifdef FUSIONS_DEBUG_0
  LOG(INFO) << AsText(func, false);
#endif

  support::Arena arena;
  auto graph = EdgexDependencyGraph::Create(&arena, func);

  AtomicGraph atomic_graph(&arena, &graph, device_type);

#ifdef FUSIONS_DEBUG_0
  LOG(INFO) << "[FusionS] Create AtomicGraph graph size = " << atomic_graph.size();
  for (size_t i = 0; i < atomic_graph.size(); i++) {
    std::cout << "[FusionS: Atomic Op] " << atomic_graph[i]->detail() << std::endl;
  }
  LOG(INFO);
#endif

  std::set<std::set<size_t>> atomic_op_patterns = FusionPlanGenerator().Run(atomic_graph);

#ifdef FUSIONS_DEBUG_2
  LOG(INFO) << "-- After FusionPlanGenerator, atomic_op_patterns";
  for (auto pattern : atomic_op_patterns) {
    std::cout << TextOfStdSet(pattern) << std::endl;
  }
  LOG(INFO);
#endif

  LOG(INFO) << "[FusionS] Performance searching... May take a long time";

  std::set<std::set<size_t>> atomic_groups =
      ILPFusionPlanning().Run(func, atomic_graph, atomic_op_patterns);

  std::vector<std::set<size_t>> relay_groups;
  std::vector<OpDeviceType> relay_group_device_types;

#ifdef FUSIONS_DEBUG_0
  LOG(INFO) << "[FusionS Result, Atomic Group]";
#endif
  for (auto atomic_group : atomic_groups) {
    std::set<size_t> relay_group;
    for (size_t atomic_id : atomic_group) {
      for (size_t i = atomic_graph[atomic_id]->start; i <= atomic_graph[atomic_id]->end; i++) {
        relay_group.insert(i);
      }
    }
#ifdef FUSIONS_DEBUG_0
    std::cout << TextOfStdSet(atomic_group);
    std::cout << " with its relay group:" << TextOfStdSet(relay_group) << std::endl;
#endif
    relay_groups.push_back(relay_group);
    relay_group_device_types.push_back(atomic_graph[(*atomic_group.begin())]->device);
  }

  return SubFunctionFuser().Transform(func, graph, relay_groups, relay_group_device_types);
}

namespace transform {

Pass FusionStitch(int fuse_opt_level, int device_type) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        int opt_level = fuse_opt_level == -1 ? pc->opt_level : fuse_opt_level;
        return Downcast<Function>(FusionStitch(f, device_type, opt_level, m));
      };
  return CreateFunctionPass(pass_func, 1, "FusionStitch", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay.edgex.transform.FusionStitch").set_body_typed(FusionStitch);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
