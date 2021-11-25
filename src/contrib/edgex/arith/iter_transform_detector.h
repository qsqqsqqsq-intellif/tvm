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
 * \file iter_transform_detector.h
 */
#ifndef TVM_CONTRIB_EDGEX_ARITH_ITER_TRANSFORM_DETECTOR_H_
#define TVM_CONTRIB_EDGEX_ARITH_ITER_TRANSFORM_DETECTOR_H_
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace arith {

using namespace tir;
using tvm::runtime::Array;

enum IterOperationType { FUSE, SPLIT };
enum ShapeOperationType { RESHAPE, TRANSPOSE };

/*!
 * \brief Describe split/fuse operation on iters.
 */
class IterOperation {
 public:
  IterOperationType type;
  std::pair<size_t, size_t> fuse_src() const {
    CHECK(type == FUSE);
    return {i0, i1};
  }
  size_t split_src() const {
    CHECK(type == SPLIT);
    return i0;
  }
  size_t fuse_dst() const {
    CHECK(type == FUSE);
    return i2;
  }
  std::pair<size_t, size_t> split_dst() const {
    CHECK(type == SPLIT);
    return {i1, i2};
  }
  static IterOperation Fuse(size_t src0, size_t src1, size_t dst) {
    return IterOperation(FUSE, src0, src1, dst);
  }
  static IterOperation Split(size_t src, size_t dst0, size_t dst1) {
    return IterOperation(SPLIT, src, dst0, dst1);
  }
  bool Depends(const IterOperation& other) const;

 private:
  IterOperation(IterOperationType t, size_t i0, size_t i1, size_t i2)
      : type(t), i0(i0), i1(i1), i2(i2) {}
  size_t i0;
  size_t i1;
  size_t i2;
};

/*!
 * \brief Describe transpose/reshape operation.
 */
class ShapeOperation {
 public:
  ShapeOperationType type;
  std::vector<int64_t> values;
  std::vector<std::vector<size_t>> iters;
  ShapeOperation(ShapeOperationType t, const std::vector<int64_t>& vec,
                 const std::vector<std::vector<size_t>>& iters)
      : type(t), values(vec), iters(iters) {}
  size_t ndim() const { return values.size(); }
};

class IterTransformDetector {
 public:
  explicit IterTransformDetector(bool respect_input_dom, bool verbose)
      : respect_input_dom(respect_input_dom), verbose(verbose) {}

  /**
   *! \brief Detect reshape/transpose operation sequence from output exprs to input variables.
   * that is, for a load/store stmt X[v0, v1, ..., vm] = Y[f0(v*), f1(v*), ..., fn(v*)], try to
   * recover how Y can be transformed to X with reshape/transpose operations.
   *  \param X  Input iteration variables.
   *  \param Y  Output expressions.
   *  \param dom_map Iteration ranges dict.
   *  \return detect success or not.
   */
  bool DetectReshapeTranspose(const Array<Var>& X, const Array<PrimExpr>& Y,
                              const Map<Var, Range>& dom_map);

  /**
   *! \brief Detect fuse/split operation sequence from input variables to produce output exprs.
   *  \param X  Input iteration variables.
   *  \param Y  Output expressions.
   *  \param dom_map Iteration ranges dict.
   *  \return detect success or not.
   */
  bool DetectFuseSplit(const Array<Var>& X, const Array<PrimExpr>& Y,
                       const Map<Var, Range>& dom_map);

  /*! \brief get binding expr for iteration `iter_idx`. */
  PrimExpr GetBinding(size_t iter_idx) const { return iter_bindings[iter_idx]; }

  /*! \brief get iteration extent for iteration `iter_idx`. */
  int64_t GetExtent(size_t iter_idx) const { return iter_extents[iter_idx]; }

  /*! \brief get iteration id of binding expr. */
  int64_t FindIterId(const PrimExpr& e) const;

  /* some common utilities */
  void ShowOps() const;
  void ShowShapeOps() const;
  std::string FormatOp(const IterOperation& op) const;
  std::string FormatOp(const ShapeOperation& op) const;

  size_t AddNewIter(const PrimExpr& binding, int64_t extent);
  void AddIterOperation(const IterOperation& op);
  void UpdateIterOperation(size_t op_idx, const IterOperation& op);
  void UpdateIterInfo(size_t idx, const PrimExpr& binding, int64_t extent);

  /*! \brief Inference iteration storage stride by external strides binding. */
  int64_t InferIterStride(size_t iter_idx,
                          const std::unordered_map<size_t, int64_t>& strides_map) const;
  bool InferIterDivision(const std::vector<size_t>& fuse_iters,
                         const std::unordered_map<size_t, int64_t>& strides_map0,
                         const std::unordered_map<size_t, int64_t>& strides_map1,
                         std::vector<int64_t>* p_shape, std::vector<int64_t>* p_strides) const;

  bool respect_input_dom;
  bool verbose{false};

  /*! \brief fuse/split operations applied on input variables, in reverse order. */
  std::vector<IterOperation> op_seq;

  /*! \brief iteration binding expr indexed by iter id/ */
  std::vector<PrimExpr> iter_bindings;

  /*! \brief iteration extent indexed by iter id/ */
  std::vector<int64_t> iter_extents;

  /*! \brief iteration stride indexed by iter id/ */
  std::vector<int64_t> iter_strides;

  /*! \brief producer op idx indexed by iter id, -1 means a root iter,
   * the producer uniqueness is ensured. */
  std::vector<int64_t> iter_producer_mapping;

  /*! \brief consumer op idx indexed by iter id, -1 means a output iter,
   * the consumer uniqueness is ensured. */
  std::vector<int64_t> iter_consumer_mapping;

  /*! \brief transpose/reshape op sequence. */
  std::vector<ShapeOperation> shape_ops;

 private:
  /*! \brief Utility to recover extents for all root&leaf&intermediate iterations. */
  bool InferExtents(const std::vector<size_t>& root_iters);

  /*! \brief optimize fuse split operation orders. */
  void ReorderFuseSplit();

  /*! \brief optimize reshape transpose operations. */
  void OptimizeReshapeTranspose();

  /*! \brief mapping from input itervar to iter extent. */
  std::unordered_map<const VarNode*, int64_t> input_extents;
};

/**
 *! \brief Utility to extract sum components for expr.
 */
void ExtractSumFactors(const PrimExpr& e, bool sign, std::vector<std::pair<PrimExpr, int>>* factors,
                       int* constant);

}  // namespace arith
}  // namespace tvm
#endif  // TVM_CONTRIB_EDGEX_ARITH_ITER_TRANSFORM_DETECTOR_H_
