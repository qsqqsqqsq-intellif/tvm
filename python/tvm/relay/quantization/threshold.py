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
# pylint: disable=unused-argument,inconsistent-return-statements,len-as-condition
"""Automatic quantization toolkit."""

import logging
import multiprocessing
import numpy
from scipy import stats
from tvm._ffi import runtime_ctypes
from .method_dtype import Method, _get_dtype_info

LOGGER = logging.getLogger("quantize")


class MinMax:
    """Calibration function: Minmax"""

    args = []

    def __init__(self, node, axis, config):
        LOGGER.debug("use Threshold.MinMax...")
        self.axis = axis
        self.min = float("inf")
        self.max = float("-inf")
        self._finished_post_process = False

    def statistics_min_max(self, x):
        pass

    def update_axis(self, new_axis):
        self.axis = new_axis

    def run(self, x):
        """Execute calibrate"""
        if self.axis == -1:
            y_min = numpy.amin(x)
            y_max = numpy.amax(x)
        elif self.axis > 10:
            groups = self.axis - 10
            LOGGER.info("Threshold.MinMax meet conv2d_transpose, groups = %d", groups)
            i_c, o_c, k_h, k_w = x.shape
            val = numpy.reshape(x, (groups, i_c // groups, o_c, k_h * k_w))
            val = numpy.transpose(val, (0, 2, 1, 3))
            val = numpy.reshape(val, (o_c * groups, i_c // groups, k_h, k_w))
            y_min = numpy.amin(val, (1, 2, 3))
            y_max = numpy.amax(val, (1, 2, 3))
        else:
            shape1 = list(range(len(x.shape)))
            shape2 = shape1[self.axis]
            shape1.remove(shape2)
            shape3 = [shape2] + shape1
            temp_x = numpy.transpose(x, shape3)
            temp_x = temp_x.reshape(temp_x.shape[0], -1)
            y_min = numpy.amin(temp_x, axis=1)
            y_max = numpy.amax(temp_x, axis=1)

        self.min = numpy.minimum(self.min, y_min)
        self.max = numpy.maximum(self.max, y_max)

    def _post_process(self):
        self._finished_post_process = True

    @property
    def min_max(self):
        if not self._finished_post_process:
            self._post_process()
        return {"min": self.min, "max": self.max}


class MovingAverageMinMax:
    """Calibration function: MovingAverageMinMax"""

    args = [{"name": "averaging", "default": numpy.array(0.01, numpy.float32), "min": 0, "max": 1}]

    def __init__(self, node, axis, config):
        LOGGER.debug("use Threshold.MovingAverageMinMax...")
        self.axis = axis
        self.averaging = config["threshold_arg"]["averaging"]
        assert 0 < self.averaging < 1
        self._first_run = True
        self._finished_post_process = False

    def statistics_min_max(self, x):
        pass

    def update_axis(self, new_axis):
        self.axis = new_axis

    def run(self, x):
        """Execute calibrate"""
        if self.axis == -1:
            if self._first_run:
                self.min = numpy.amin(x)
                self.max = numpy.amax(x)
            else:
                y_min = numpy.amin(x)
                y_max = numpy.amax(x)
                self.min = self.min + self.averaging * (y_min - self.min)
                self.max = self.max + self.averaging * (y_max - self.max)
        else:
            shape1 = list(range(len(x.shape)))
            shape2 = shape1[self.axis]
            shape1.remove(shape2)
            shape3 = [shape2] + shape1
            temp_x = numpy.transpose(x, shape3)
            temp_x = temp_x.reshape(temp_x.shape[0], -1)
            if self._first_run:
                self.min = numpy.amin(temp_x, axis=1)
                self.max = numpy.amax(temp_x, axis=1)
            else:
                y_min = numpy.amin(temp_x, axis=1)
                y_max = numpy.amax(temp_x, axis=1)
                self.min = self.min + self.averaging * (y_min - self.min)
                self.max = self.max + self.averaging * (y_max - self.max)

        self._first_run = False

    def _post_process(self):
        self._finished_post_process = True

    @property
    def min_max(self):
        if not self._finished_post_process:
            self._post_process()
        return {"min": self.min, "max": self.max}


class Percentile:
    """Calibration function: Percentile"""

    args = [
        {"name": "percentile", "default": numpy.array(0.9999, numpy.float32), "min": 0, "max": 1}
    ]

    def __init__(self, node, axis, config):
        LOGGER.debug("use Threshold.Percentile...")
        self.axis = axis
        self.percentile = config["threshold_arg"]["percentile"]
        assert 0 < self.percentile < 1
        shape = node.checked_type.concrete_shape
        self.shape = shape
        self.node = node
        nums = numpy.array(shape).prod()
        self.nums = round((1 - self.percentile) * nums * config["threshold_arg"]["calibrate_num"])
        self.nums = 1 if self.nums == 0 else self.nums
        if self.axis == -1:
            self.collected_min = numpy.empty(0, node.checked_type.dtype)
            self.collected_max = numpy.empty(0, node.checked_type.dtype)
        else:
            self.collected_min = numpy.empty([shape[self.axis], 0], node.checked_type.dtype)
            self.collected_max = numpy.empty([shape[self.axis], 0], node.checked_type.dtype)
        self._finished_post_process = False

    def update_axis(self, new_axis):
        self.axis = new_axis
        if self.axis == -1:
            self.collected_min = numpy.empty(0, self.node.checked_type.dtype)
            self.collected_max = numpy.empty(0, self.node.checked_type.dtype)
        else:
            self.collected_min = numpy.empty(
                [self.shape[self.axis], 0], self.node.checked_type.dtype
            )
            self.collected_max = numpy.empty(
                [self.shape[self.axis], 0], self.node.checked_type.dtype
            )

    def statistics_min_max(self, x):
        pass

    def run(self, x):
        """Execute perncentile"""

        if self.axis == -1:
            temp_x = x.reshape(-1)
            self.collected_min = numpy.concatenate([self.collected_min, temp_x], 0)
            self.collected_max = numpy.concatenate([self.collected_max, temp_x], 0)

            if self.collected_min.size > self.nums:
                sort = numpy.sort(self.collected_min, -1)
                self.collected_min = sort[: self.nums]
            if self.collected_max.size > self.nums:
                sort = numpy.sort(self.collected_max, -1)
                self.collected_max = sort[-self.nums :]
        else:
            shape1 = list(range(len(x.shape)))
            shape2 = shape1[self.axis]
            shape1.remove(shape2)
            shape3 = [shape2] + shape1
            temp_x = numpy.transpose(x, shape3)
            temp_x = temp_x.reshape(temp_x.shape[0], -1)
            self.collected_min = numpy.concatenate([self.collected_min, temp_x], 1)
            self.collected_max = numpy.concatenate([self.collected_max, temp_x], 1)

            nums = round(self.nums / self.collected_min.shape[0])
            nums = 1 if nums == 0 else nums
            if self.collected_min.size >= self.nums:
                sort = numpy.sort(self.collected_min, 1)
                self.collected_min = sort[:, :nums]
            if self.collected_max.size > self.nums:
                sort = numpy.sort(self.collected_max, 1)
                self.collected_max = sort[:, -nums:]

    def _post_process(self):
        if self.axis == -1:
            self.min = self.collected_min[-1]
            self.max = self.collected_max[0]
        else:
            self.min = self.collected_min[:, -1]
            self.max = self.collected_max[:, 0]

        self._finished_post_process = True

    @property
    def min_max(self):
        if not self._finished_post_process:
            self._post_process()
        return {"min": self.min, "max": self.max}


class L2Norm:
    """
    Search the distribution in the histogram for optimal min/max values.
    The search for the min/max values ensures the minimization of the
    quantization error with respect to the floating point model.
    """

    args = [{"name": "bins", "default": numpy.array(2048, numpy.int32), "min": 0, "max": numpy.inf}]

    def __init__(self, node, axis, config):
        LOGGER.debug("use Threshold.L2Norm...")
        self.axis = axis
        self.bins = config["threshold_arg"]["bins"]
        assert self.bins > 0

        if "DataType" in runtime_ctypes.__dict__:
            self.dst_nbins = 2 ** runtime_ctypes.DataType(config["dtype"]).bits
        elif "TVMType" in runtime_ctypes.__dict__:
            self.dst_nbins = 2 ** runtime_ctypes.TVMType(config["dtype"]).bits

        self.min = float("inf")
        self.max = float("-inf")
        shape = node.checked_type.concrete_shape
        self.shape = shape
        if axis == -1:
            self.histogram = numpy.zeros(self.bins, numpy.int64)
        else:
            self.histogram = numpy.zeros([shape[axis], self.bins], numpy.int64)
        self._finished_post_process = False

    def update_axis(self, new_axis):
        self.axis = new_axis
        if self.axis == -1:
            self.histogram = numpy.zeros(self.bins, numpy.int64)
        else:
            self.histogram = numpy.zeros([self.shape[new_axis], self.bins], numpy.int64)

    def statistics_min_max(self, x):
        """statistics_min_max"""
        if self.axis == -1:
            y_min = numpy.amin(x)
            y_max = numpy.amax(x)
        else:
            shape1 = list(range(len(x.shape)))
            shape2 = shape1[self.axis]
            shape1.remove(shape2)
            shape3 = [shape2] + shape1
            temp_x = numpy.transpose(x, shape3)
            temp_x = temp_x.reshape(temp_x.shape[0], -1)
            y_min = numpy.amin(temp_x, axis=1)
            y_max = numpy.amax(temp_x, axis=1)

        self.min = numpy.minimum(self.min, y_min)
        self.max = numpy.maximum(self.max, y_max)

    def run(self, x):
        """run"""
        if self.axis == -1:
            histogram = numpy.histogram(x, self.bins, (self.min, self.max))[0]
            self.histogram = self.histogram + histogram
        else:
            shape1 = list(range(len(x.shape)))
            shape2 = shape1[self.axis]
            shape1.remove(shape2)
            shape3 = [shape2] + shape1
            temp_x = numpy.transpose(x, shape3)
            temp_x = temp_x.reshape(temp_x.shape[0], -1)
            for i, _ in enumerate(temp_x):
                histogram = numpy.histogram(temp_x[i], self.bins, (self.min[i], self.max[i]))[0]
                self.histogram[i] = self.histogram[i] + histogram

    def _get_norm(self, delta_begin, delta_end, density):
        """
        Compute the norm of the values uniformaly distributed between delta_begin and delta_end.
        Currently only L2 norm is supported.

        norm = density * (integral_{begin, end} x^2)
             = density * (end^3 - begin^3) / 3
        """
        norm = (delta_end * delta_end * delta_end - delta_begin * delta_begin * delta_begin) / 3
        return density * norm

    def _compute_quantization_error(self, min_, max_, histogram, next_start_bin, next_end_bin):
        """
        Compute the quantization error if we use start_bin
        to end_bin as the min and max to do the quantization.
        """
        bin_width = (max_ - min_) / self.bins

        dst_bin_width = bin_width * (next_end_bin - next_start_bin + 1) / self.dst_nbins
        if dst_bin_width == 0.0:
            return 0.0

        src_bin = numpy.arange(self.bins)
        # distances from the beginning of first dst_bin to the beginning and
        # end of src_bin
        src_bin_begin = (src_bin - next_start_bin) * bin_width
        src_bin_end = src_bin_begin + bin_width

        # which dst_bins the beginning and end of src_bin belong to?
        dst_bin_of_begin = numpy.clip(src_bin_begin // dst_bin_width, 0, self.dst_nbins - 1)
        dst_bin_of_begin_center = (dst_bin_of_begin + 0.5) * dst_bin_width

        dst_bin_of_end = numpy.clip(src_bin_end // dst_bin_width, 0, self.dst_nbins - 1)
        dst_bin_of_end_center = (dst_bin_of_end + 0.5) * dst_bin_width

        density = histogram / bin_width

        norm = numpy.zeros(self.bins)

        delta_begin = src_bin_begin - dst_bin_of_begin_center
        delta_end = dst_bin_width / 2
        norm += self._get_norm(delta_begin, numpy.ones(self.bins) * delta_end, density)

        norm += (dst_bin_of_end - dst_bin_of_begin - 1) * self._get_norm(
            numpy.array(-dst_bin_width / 2), numpy.array(dst_bin_width / 2), density
        )

        dst_bin_of_end_center = dst_bin_of_end * dst_bin_width + dst_bin_width / 2

        delta_begin = -dst_bin_width / 2
        delta_end = src_bin_end - dst_bin_of_end_center
        norm += self._get_norm(numpy.array(delta_begin), delta_end, density)

        return norm.sum()

    def _non_linear_param_search(self):
        """
        Non-linear parameter search.

        An approximation for L2 error minimization for selecting min/max.
        By selecting new min/max, we filter out outliers in input distribution.
        This follows the implementation of NormMinimization::NonlinearQuantizationParamsSearch in
        caffe2/quantization/server/norm_minimization.cc
        """

        def tmp(min_, max_, histogram):
            """cal core"""
            bin_width = (max_ - min_) / self.bins

            # cumulative sum
            total = numpy.sum(histogram)
            csum = numpy.cumsum(histogram, 0)

            stepsize = 1e-5  # granularity
            alpha = 0.0  # lower bound
            beta = 1.0  # upper bound
            start_bin = 0
            end_bin = self.bins - 1
            norm_min = float("inf")

            while alpha < beta:
                # Find the next step
                next_alpha = alpha + stepsize
                next_beta = beta - stepsize

                # find the left and right bins between the quantile bounds
                l_bin = start_bin
                r_bin = end_bin
                while l_bin < end_bin and csum[l_bin] < next_alpha * total:
                    l_bin = l_bin + 1
                while r_bin > start_bin and csum[r_bin] > next_beta * total:
                    r_bin = r_bin - 1

                # decide the next move
                next_start_bin = start_bin
                next_end_bin = end_bin
                if (l_bin - start_bin) > (end_bin - r_bin):
                    # move the start bin
                    next_start_bin = l_bin
                    alpha = next_alpha
                else:
                    # move the end bin
                    next_end_bin = r_bin
                    beta = next_beta

                if next_start_bin == start_bin and next_end_bin == end_bin:
                    continue

                # calculate the quantization error using next_start_bin and next_end_bin
                norm = self._compute_quantization_error(
                    min_, max_, histogram, next_start_bin, next_end_bin
                )

                if norm > norm_min:
                    break
                norm_min = norm
                start_bin = next_start_bin
                end_bin = next_end_bin

            new_min = min_ + bin_width * start_bin
            new_max = min_ + bin_width * (end_bin + 1)
            return new_min, new_max

        if len(self.max.shape) == 0:
            new_min, new_max = tmp(self.min, self.max, self.histogram)
        else:
            new_min = []
            new_max = []
            for i, _ in enumerate(self.min):
                t1_, t2_ = tmp(self.min[i], self.max[i], self.histogram[i])
                new_min.append(t1_)
                new_max.append(t2_)
            new_min = numpy.array(new_min)
            new_max = numpy.array(new_max)

        self.min = new_min
        self.max = new_max

    def _post_process(self):
        self._non_linear_param_search()
        self._finished_post_process = True

    @property
    def min_max(self):
        if not self._finished_post_process:
            self._post_process()
        return {"min": self.min, "max": self.max}


class RelativeEntropy:
    """relative entropy"""

    args = [{"name": "bins", "default": numpy.array(2048, numpy.int32), "min": 0, "max": numpy.inf}]

    def __init__(self, node, axis, config):
        LOGGER.debug("use Threshold.RelativeEntropy...")
        self.axis = axis
        self.bins = config["threshold_arg"]["bins"]
        assert self.bins > 0
        self.min = float("inf")
        self.max = float("-inf")
        shape = node.checked_type.concrete_shape
        if axis == -1:
            self.histogram = numpy.zeros(self.bins, numpy.int64)
        else:
            self.histogram = numpy.zeros([shape[axis], self.bins], numpy.int64)
        if "DataType" in runtime_ctypes.__dict__:
            self.q_bins = 2 ** runtime_ctypes.DataType(config["dtype"]).bits
        elif "TVMType" in runtime_ctypes.__dict__:
            self.q_bins = 2 ** runtime_ctypes.TVMType(config["dtype"]).bits
        self._first_run = True
        self._finished_post_process = False

        self.node = node
        self.shape = shape

    def update_axis(self, new_axis):
        self.axis = new_axis
        if new_axis == -1:
            self.histogram = numpy.zeros(self.bins, numpy.int64)
        else:
            self.histogram = numpy.zeros([self.shape[new_axis], self.bins], numpy.int64)

    def statistics_min_max(self, x):
        """statistics_min_max"""
        if self.axis == -1:
            y_min = numpy.amin(x)
            y_max = numpy.amax(x)
        else:
            shape1 = list(range(len(x.shape)))
            shape2 = shape1[self.axis]
            shape1.remove(shape2)
            shape3 = [shape2] + shape1
            temp_x = numpy.transpose(x, shape3)
            temp_x = temp_x.reshape(temp_x.shape[0], -1)
            y_min = numpy.amin(temp_x, axis=1)
            y_max = numpy.amax(temp_x, axis=1)

            # from tvm import relay
            # if isinstance(self.node, relay.Call) and not isinstance(self.node.op, relay.Function):
            #     LOGGER.debug("self.node is " + self.node.op.name)
            # if isinstance(self.node, relay.Call) and isinstance(self.node.op, relay.Function):
            #     LOGGER.debug("convd biasadd")
            # LOGGER.debug("shape1 list is ")
            # LOGGER.debug(shape1)
            # LOGGER.debug("x.shape is ")
            # LOGGER.debug(x.shape)
            # LOGGER.debug("self.shape is ")
            # LOGGER.debug(self.shape)

        self.min = numpy.minimum(self.min, y_min)
        self.max = numpy.maximum(self.max, y_max)

        self._first_run = False

    def run(self, x):
        """run"""
        if self.axis == -1:
            histogram = numpy.histogram(x, self.bins, (self.min, self.max))[0]
            self.histogram = self.histogram + histogram
        else:
            shape1 = list(range(len(x.shape)))
            shape2 = shape1[self.axis]
            shape1.remove(shape2)
            shape3 = [shape2] + shape1
            temp_x = numpy.transpose(x, shape3)
            temp_x = temp_x.reshape(temp_x.shape[0], -1)
            for i, _ in enumerate(temp_x):
                histogram = numpy.histogram(temp_x[i], self.bins, (self.min[i], self.max[i]))[0]
                self.histogram[i] = self.histogram[i] + histogram

    def _smooth_distribution(self, p, eps=0.0001):
        """Given a discrete distribution (may have not been normalized to 1),
        smooth it by replacing zeros with eps multiplied by a scaling factor and taking the
        corresponding amount off the non-zero values.
        Ref: http://hanj.cs.illinois.edu/cs412/bk3/KL-divergence.pdf
        """
        is_zeros = (p == 0).astype(numpy.float32)
        is_nonzeros = (p != 0).astype(numpy.float32)
        n_zeros = is_zeros.sum()
        n_nonzeros = p.size - n_zeros
        if not n_nonzeros:
            raise ValueError(
                "The discrete probability distribution is malformed. All entries are 0."
            )
        eps1 = eps * float(n_zeros) / float(n_nonzeros)
        assert eps1 < 1.0, "n_zeros=%d, n_nonzeros=%d, eps1=%f" % (n_zeros, n_nonzeros, eps1)
        hist = p.astype(numpy.float32)
        hist += eps * is_zeros + (-eps1) * is_nonzeros
        assert (hist <= 0).sum() == 0
        return hist

    def _post_process(self):
        """Given a tensor, find the optimal threshold for quantizing it.
        The reference distribution is `q`, and the candidate distribution is `p`.
        `q` is a truncated version of the original distribution.
        Ref:
        http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
        """

        def tmp(min_val, max_val, histogram):
            """cal core"""
            th_ = max(abs(min_val), abs(max_val))
            _, hist_edges = numpy.histogram(0, self.bins, range=(-th_, th_))
            zero_bin_idx = self.bins // 2
            num_half_quantized_bins = self.q_bins // 2

            thresholds = numpy.zeros(self.bins // 2 + 1 - self.q_bins // 2)
            divergence = numpy.zeros_like(thresholds)
            quantized_bins = numpy.zeros(self.q_bins, dtype=numpy.int32)
            # i means the number of bins on half axis excluding the zero bin.
            for i in range(self.q_bins // 2, self.bins // 2 + 1):
                p_bin_idx_start = zero_bin_idx - i
                p_bin_idx_stop = zero_bin_idx + i
                thresholds[i - num_half_quantized_bins] = hist_edges[p_bin_idx_stop]
                sliced_nd_hist = histogram[p_bin_idx_start:p_bin_idx_stop]

                # generate reference distribution p
                p = sliced_nd_hist.copy()
                # assert p.size % 2 == 1
                assert p.size >= self.q_bins
                # put left outlier count in p[0]
                left_outlier_count = numpy.sum(histogram[0:p_bin_idx_start])
                p[0] += left_outlier_count
                # put right outlier count in p[-1]
                right_outlier_count = numpy.sum(histogram[p_bin_idx_stop:])
                p[-1] += right_outlier_count
                # is_nonzeros[k] indicates whether histogram[k] is nonzero
                is_nonzeros = (p != 0).astype(numpy.int32)

                # calculate how many bins should be merged to generate quantized distribution q
                num_merged_bins = sliced_nd_hist.size // self.q_bins
                # merge histogram into q_bins bins
                for j in range(self.q_bins):
                    start = j * num_merged_bins
                    stop = start + num_merged_bins
                    quantized_bins[j] = sliced_nd_hist[start:stop].sum()
                quantized_bins[-1] += sliced_nd_hist[self.q_bins * num_merged_bins :].sum()
                # expand quantized_bins into p.size bins
                q1_ = numpy.zeros(sliced_nd_hist.size, dtype=numpy.float32)
                for j in range(self.q_bins):
                    start = j * num_merged_bins
                    if j == self.q_bins - 1:
                        stop = len(is_nonzeros)
                    else:
                        stop = start + num_merged_bins
                    norm = is_nonzeros[start:stop].sum()
                    if norm != 0:
                        q1_[start:stop] = float(quantized_bins[j]) / float(norm)
                q1_[p == 0] = 0
                p = self._smooth_distribution(p)
                # There is a chance that q is an invalid probability distribution.
                try:
                    q1_ = self._smooth_distribution(q1_)
                    divergence[i - num_half_quantized_bins] = stats.entropy(p, q1_)
                except ValueError:
                    divergence[i - num_half_quantized_bins] = float("inf")

            min_divergence_idx = numpy.argmin(divergence)
            threshold = thresholds[min_divergence_idx]
            return -threshold, threshold

        if self.axis == -1:
            self.neg_thresh, self.pos_thresh = tmp(self.min, self.max, self.histogram)
        else:
            shape = self.histogram.shape[0]
            self.neg_thresh = numpy.zeros([shape], numpy.float32)
            self.pos_thresh = numpy.zeros([shape], numpy.float32)
            for i in range(shape):
                self.neg_thresh[i], self.pos_thresh[i] = tmp(
                    self.min[i], self.max[i], self.histogram[i]
                )
                self.neg_thresh[i], self.pos_thresh[i] = self.min[i], self.max[i]
        self._finished_post_process = True

    @property
    def min_max(self):
        if not self._finished_post_process:
            self._post_process()
        return {"min": self.neg_thresh, "max": self.pos_thresh}


class PercentileAbs:
    """Calibration function: PercentileAbs"""

    args = [
        {"name": "percentile", "default": numpy.array(0.9999, numpy.float32), "min": 0, "max": 1}
    ]

    def __init__(self, node, axis, config):
        LOGGER.debug("use Threshold.PercentileAbs...")
        self.axis = axis
        self.percentile = config["threshold_arg"]["percentile"]
        assert 0 < self.percentile < 1
        shape = node.checked_type.concrete_shape
        self.shape = shape
        self.node = node
        self.calibrate_num = config["threshold_arg"]["calibrate_num"]
        nums = numpy.array(shape).prod()
        self.nums = int(nums * self.calibrate_num * (1 - self.percentile))
        self.nums = 1 if self.nums == 0 else self.nums
        if self.axis == -1:
            self.collected_min = numpy.zeros(1, node.checked_type.dtype)
            self.collected_max = numpy.empty(0, node.checked_type.dtype)
        else:
            self.collected_min = numpy.zeros([shape[self.axis], 1], node.checked_type.dtype)
            self.collected_max = numpy.empty([shape[self.axis], 0], node.checked_type.dtype)
        self._finished_post_process = False

    def update_axis(self, new_axis):
        self.axis = new_axis
        if self.axis == -1:
            self.collected_min = numpy.zeros(1, self.node.checked_type.dtype)
            self.collected_max = numpy.empty(0, self.node.checked_type.dtype)
        else:
            self.collected_min = numpy.zeros(
                [self.shape[self.axis], 1], self.node.checked_type.dtype
            )
            self.collected_max = numpy.empty(
                [self.shape[self.axis], 0], self.node.checked_type.dtype
            )

    def statistics_min_max(self, x):
        pass

    def run(self, x):
        """Execute perncentile abs"""

        if self.axis == -1:
            temp_x = numpy.abs(x.reshape(-1))
            tmp = numpy.concatenate([self.collected_max, temp_x], 0)

            if tmp.size > self.nums:
                sort = numpy.partition(tmp, -self.nums)
                # sort = numpy.sort(tmp, -1)
                self.collected_max = sort[-self.nums :]
        else:
            shape1 = list(range(len(x.shape)))
            shape2 = shape1[self.axis]
            shape1.remove(shape2)
            shape3 = [shape2] + shape1
            temp_x = numpy.abs(numpy.transpose(x, shape3))
            temp_x = temp_x.reshape(temp_x.shape[0], -1)
            tmp = numpy.concatenate([self.collected_max, temp_x], 1)

            nums = int(
                numpy.array(self.shape).prod()
                / self.shape[self.axis]
                * self.calibrate_num
                * (1 - self.percentile)
            )
            nums = 1 if nums == 0 else nums

            if tmp.size > self.nums:
                sort = numpy.partition(tmp, -nums, axis=1)
                # sort = numpy.sort(tmp, 1)
                # sort = torch.sort()
                self.collected_max = sort[:, -nums:]

    def _post_process(self):
        if self.axis == -1:
            self.min = self.collected_min[-1]
            self.max = self.collected_max[0]
        else:
            self.min = self.collected_min[:, -1]
            self.max = self.collected_max[:, 0]

        self._finished_post_process = True

    @property
    def min_max(self):
        if not self._finished_post_process:
            self._post_process()
        return {"min": self.min, "max": self.max}


class KLDAbs:
    """kld abs"""

    args = []

    def __init__(self, node, axis, config):
        LOGGER.debug("use Threshold.KLDAbs...")
        self.axis = axis
        self.bins = 2048
        self.min = 0.0
        self.max = float("-inf")
        shape = node.checked_type.concrete_shape
        if axis == -1:
            self.histogram = numpy.zeros(self.bins, numpy.int64)
        else:
            self.histogram = numpy.zeros([shape[axis], self.bins], numpy.int64)
        self.q_bins = 128
        self._first_run = True
        self._finished_post_process = False

        self.node = node
        self.shape = shape

    def update_axis(self, new_axis):
        self.axis = new_axis
        if new_axis == -1:
            self.histogram = numpy.zeros(self.bins, numpy.int64)
        else:
            self.histogram = numpy.zeros([self.shape[new_axis], self.bins], numpy.int64)

    def statistics_min_max(self, x):
        """statistics_min_max"""
        x = numpy.abs(x)
        if self.axis == -1:
            y_max = numpy.amax(x)
        else:
            shape1 = list(range(len(x.shape)))
            shape2 = shape1[self.axis]
            shape1.remove(shape2)
            shape3 = [shape2] + shape1
            temp_x = numpy.transpose(x, shape3)
            temp_x = temp_x.reshape(temp_x.shape[0], -1)

            y_max = numpy.amax(temp_x, axis=1)

        self.max = numpy.maximum(self.max, y_max)

        self._first_run = False

    def run(self, x):
        """run"""
        x = numpy.abs(x)
        if self.axis == -1:
            # histogram = numpy.histogram(x, self.bins, (self.min, self.max))[0]

            # identity to nnp300
            if numpy.isnan(self.max) or numpy.isinf(self.max):
                assert 0, print(f"threshold meet {self.max}, {x.shape}")
            x = x.flatten()
            width = self.max / (self.bins - 1)
            eps = 0 if width > 0 else 0.0001
            width += eps
            temp = numpy.floor(x / width + 0.5) * width
            histogram = numpy.histogram(temp, self.bins, (self.min, self.max))[0]

            self.histogram = self.histogram + histogram
        else:
            shape1 = list(range(len(x.shape)))
            shape2 = shape1[self.axis]
            shape1.remove(shape2)
            shape3 = [shape2] + shape1
            temp_x = numpy.transpose(x, shape3)
            temp_x = temp_x.reshape(temp_x.shape[0], -1)
            # yhh change 1026 temp todo
            # for i, _ in enumerate(temp_x):
            #     histogram = numpy.histogram(temp_x[i], self.bins, (self.min[i], self.max[i]))[0]
            #     self.histogram[i] = self.histogram[i] + histogram

    def _post_process(self):
        """use 300 strategy"""

        def tmp(max_val, histogram):
            """cal core"""
            histogram[0] = 0.0
            sum_all = 0.0
            m = 0
            kl_num = self.bins // self.q_bins
            kl_array = numpy.zeros(kl_num)
            count = histogram.sum()
            width = max_val / self.bins
            for i in range(self.q_bins, self.bins + 1, self.q_bins):
                p_distrib = numpy.zeros((i))
                for j in range(i):
                    p_distrib[j] = histogram[j]

                for j in range(i, self.bins):
                    p_distrib[i - 1] = p_distrib[i - 1] + histogram[j]

                for j in range(i):
                    p_distrib[j] = p_distrib[j] / count

                expand_size = i // self.q_bins
                idx = 0
                q_distrib = numpy.zeros(i)
                for j in range(i - self.q_bins, i):
                    sum_all = sum_all + histogram[j]

                for j in range(self.q_bins):
                    sum_bin = 0
                    positive_cnt = 0
                    bin_idx = idx
                    k_idx = 0
                    while k_idx < expand_size:
                        sum_bin += histogram[idx]
                        positive_cnt += 1 if histogram[idx] > 0 else 0
                        idx += 1
                        k_idx += 1

                    positive_cnt = 1 if positive_cnt == 0 else positive_cnt
                    q_base = sum_bin / positive_cnt / sum_all

                    while bin_idx < idx:
                        q_distrib[bin_idx] = q_base if histogram[bin_idx] else 0
                        bin_idx += 1

                for idx in range(i):
                    kl_array[m] += p_distrib[idx] * (
                        numpy.log10(p_distrib[idx] + 1e-30) - numpy.log10(q_distrib[idx] + 1e-30)
                    )
                m += 1

            min_divergence_idx = numpy.argmin(kl_array)
            threshold = width * (min_divergence_idx + 1) * self.q_bins
            return threshold

        if self.axis == -1:
            self.neg_thresh = 0
            self.pos_thresh = tmp(self.max, self.histogram)
        else:
            shape = self.histogram.shape[0]
            self.neg_thresh = numpy.zeros([shape], numpy.float32)
            self.pos_thresh = numpy.zeros([shape], numpy.float32)
            for i in range(shape):
                # self.neg_thresh[i], self.pos_thresh[i] = tmp(
                #     self.min[i], self.max[i], self.histogram[i]
                # )
                # yhh change 1026 temp todo
                self.neg_thresh[i], self.pos_thresh[i] = 0.0, self.max[i]
        self._finished_post_process = True

    @property
    def min_max(self):
        if not self._finished_post_process:
            self._post_process()
        return {"min": self.neg_thresh, "max": self.pos_thresh}


class DistanceLinearSearch:
    """Calibration function: DistanceLinearSearch"""

    args = [
        {"name": "averaging", "default": numpy.array(0.01, numpy.float32), "min": 0, "max": 1},
        {
            "name": "distance",
            "default": "norm_2",
            "enumerate": [
                "norm_0",
                "norm_1",
                "norm_2",
                "norm_infinity",
                "cosine",
                "relative_entropy",
            ],
        },
        {"name": "times", "default": 1000, "min": 1},
        {"name": "step", "default": 0.0007, "min": 0, "max": 1},
    ]

    def __init__(self, node, axis, config):
        LOGGER.debug("use Threshold.DistanceLinearSearch...")
        self.axis = axis

        self.averaging = config["threshold_arg"]["averaging"]
        for one in self.args:
            if one["name"] == "averaging":
                assert one["min"] < self.averaging < one["max"]

        distance = config["threshold_arg"]["distance"]
        for one in self.args:
            if one["name"] == "distance":
                assert distance in one["enumerate"]

        self.times = config["threshold_arg"]["times"]
        for one in self.args:
            if one["name"] == "times":
                assert self.times > one["min"]

        self.step = config["threshold_arg"]["step"]
        for one in self.args:
            if one["name"] == "step":
                assert one["min"] < self.step < one["max"]

        if distance == "norm_0":
            self.distance_func = self._norm_0
        elif distance == "norm_1":
            self.distance_func = self._norm_1
        elif distance == "norm_2":
            self.distance_func = self._norm_2
        elif distance == "norm_infinity":
            self.distance_func = self._norm_infinity
        elif distance == "cosine":
            self.distance_func = self._cosine
        elif distance == "relative_entropy":
            self.distance_func = self._relative_entropy

        if config["method"] == Method.Symmetry:
            self.method = self._symmetry
        elif config["method"] == Method.Asymmetry:
            self.method = self._asymmetry
        else:
            raise ValueError

        tmp = _get_dtype_info(config["dtype"])

        self.qmin = tmp["qmin"]
        self.qmax = tmp["qmax"]
        self._cumulate = 0

    def update_axis(self, new_axis):
        self.axis = new_axis

    def statistics_min_max(self, x):
        pass

    def _norm_0(self, x, xqx):
        m = x - xqx
        m = numpy.count_nonzero(m, axis=1)
        distance = m / x[0].size
        return distance

    def _norm_1(self, x, xqx):
        m = x - xqx
        m = numpy.abs(m)
        m = numpy.sum(m, axis=1)
        distance = m / x[0].size
        return distance

    def _norm_2(self, x, xqx):
        m = x - xqx
        m = numpy.power(m, 2)
        m = numpy.sum(m, axis=1)
        m = numpy.power(m, 0.5)
        distance = m / x[0].size
        return distance

    def _norm_infinity(self, x, xqx):
        m = x - xqx
        m = numpy.abs(m)
        distance = numpy.max(m, axis=1)
        return distance

    def _cosine(self, x, xqx):
        m = x * xqx
        dot = numpy.sum(m, axis=1)

        m1_ = numpy.power(x, 2)
        m1_ = numpy.sum(m1_, axis=1)
        m1_ = numpy.power(m1_, 0.5)

        m2_ = numpy.power(xqx, 2)
        m2_ = numpy.sum(m2_, axis=1)
        m2_ = numpy.power(m2_, 0.5)

        length = m1_ * m2_
        distance = numpy.zeros_like(length)
        distance[:] = numpy.inf
        for i in range(distance.shape[0]):
            if length[i] == 0:
                distance[i] = numpy.inf
            else:
                distance[i] = dot[i] / length[i]
        return distance

    def _relative_entropy(self, x, xqx):
        def _smooth_distribution(p, eps=1e-4):
            is_zeros = (p == 0).astype(numpy.float32)
            is_nonzeros = (p != 0).astype(numpy.float32)
            n_zeros = is_zeros.sum()
            n_nonzeros = p.size - n_zeros
            if not n_nonzeros:
                raise ValueError(
                    "The discrete probability distribution is malformed. All entries are 0."
                )
            eps1 = eps * float(n_zeros) / float(n_nonzeros)
            hist = p.astype(numpy.float32)
            hist += eps * is_zeros + (-eps1) * is_nonzeros
            return hist

        distances = numpy.ones(x.shape[0], dtype=x.dtype) * float("inf")

        for i, (x1_, x2_) in enumerate(zip(x, xqx)):
            hist1 = numpy.histogram(x1_, bins=2048)[0]
            hist1 = _smooth_distribution(hist1)
            # hist1 = numpy.maximum(hist1, numpy.zeros_like(hist1)+1e-9)
            hist1 = hist1 / hist1.sum()

            hist2 = numpy.histogram(x2_, bins=2048)[0]
            hist2 = _smooth_distribution(hist2)
            # hist2 = numpy.maximum(hist2, numpy.zeros_like(hist2)+1e-9)
            hist2 = hist2 / hist2.sum()

            distances[i] = (hist1 * numpy.log(hist1 / hist2)).sum()
        return distances

    def _quantize(self, x, scale, zero_point):
        xq_ = x / scale + zero_point
        xq_ = numpy.clip(xq_, self.qmin, self.qmax).round()
        return xq_

    def _dequantize(self, xq_, scale, zero_point):
        xqx = (xq_ - zero_point) * scale
        return xqx

    def _symmetry(self, x_min, x_max):
        max_val = numpy.maximum(numpy.abs(x_min), numpy.abs(x_max))
        scale = numpy.array(max_val / self.qmax)
        scale[numpy.where(scale == 0)] = 1e-2 / self.qmax
        zero_point = numpy.zeros_like(scale)
        return scale, zero_point

    def _asymmetry(self, x_min, x_max):
        min_val = numpy.minimum(x_min, numpy.zeros_like(x_min))
        max_val = numpy.maximum(x_max, numpy.zeros_like(x_max))
        # min_val = x_min
        # max_val = x_max
        scale = numpy.array((max_val - min_val) / float(self.qmax - self.qmin)).astype(
            numpy.float32
        )
        scale[numpy.where(scale == 0)] = 1e-2 / (self.qmax - self.qmin)
        # zero_point = self.qmin - numpy.round(min_val / scale)
        zero_point = self.qmax - numpy.round(max_val / scale)  # 添加对比重建误差
        zero_point = numpy.clip(zero_point, self.qmin, self.qmax)
        return scale, zero_point

    def _group(self, i, shape, x_min, x_max, name, dtype):
        # pylint: disable=import-outside-toplevel
        from multiprocessing import shared_memory

        shared_x = shared_memory.SharedMemory(name=name)
        buffer = numpy.ndarray(shape, dtype=dtype, buffer=shared_x.buf)
        new_min = x_min * (1.0 - (i * self.step))
        new_max = x_max * (1.0 - (i * self.step))
        scale, zero_point = self.method(new_min, new_max)
        scale = scale.reshape(-1, 1)
        zero_point = zero_point.reshape(-1, 1)
        xq_ = self._quantize(buffer, scale, zero_point)
        xqx = self._dequantize(xq_, scale, zero_point)
        distance = self.distance_func(buffer, xqx)
        return distance, new_min, new_max

    def _caculate(self, x, x_min, x_max):
        # pylint: disable=import-outside-toplevel
        from multiprocessing import shared_memory

        if x.size > 200000:
            shape = x.shape
            dtype = x.dtype
            shared_x = shared_memory.SharedMemory(create=True, size=x.nbytes)
            name = shared_x.name
            buffer = numpy.ndarray(shape, dtype=dtype, buffer=shared_x.buf)
            buffer[:] = x[:]

            pool = multiprocessing.Pool(multiprocessing.cpu_count() // 2)
            process_list = []
            for i in range(self.times):
                p = pool.apply_async(func=self._group, args=(i, shape, x_min, x_max, name, dtype))
                process_list.append(p)
            pool.close()
            pool.join()
            shared_x.close()
            shared_x.unlink()

            all_distance = []
            all_min = []
            all_max = []
            for p in process_list:
                tmp1, tmp2, tmp3 = p.get()
                all_distance.append(tmp1)
                all_min.append(tmp2)
                all_max.append(tmp3)
            all_distance = numpy.array(all_distance)
            all_min = numpy.array(all_min)
            all_max = numpy.array(all_max)
            best_distance = numpy.argmin(all_distance, 0)
            best_min = all_min[best_distance, numpy.arange(best_distance.size)]
            best_max = all_max[best_distance, numpy.arange(best_distance.size)]
        else:
            best_distance = numpy.zeros_like(x_min)
            best_distance[:] = float("inf")
            best_min = numpy.zeros_like(x_min)
            best_min[:] = float("inf")
            best_max = numpy.zeros_like(x_max)
            best_max[:] = float("-inf")
            for i in range(self.times):
                new_min = x_min * (1.0 - (i * self.step))
                new_max = x_max * (1.0 - (i * self.step))
                scale, zero_point = self.method(new_min, new_max)
                scale = scale.reshape(-1, 1)
                zero_point = zero_point.reshape(-1, 1)
                xq_ = self._quantize(x, scale, zero_point)
                xqx = self._dequantize(xq_, scale, zero_point)
                distance = self.distance_func(x, xqx)
                cond = numpy.where(distance < best_distance)[0]
                best_distance[cond] = distance[cond]
                best_min[cond] = new_min[cond]
                best_max[cond] = new_max[cond]
        return best_min, best_max

    def run(self, x):
        """run"""
        if self.axis == -1:
            temp_x = x.reshape(1, -1)
        else:
            shape1 = list(range(len(x.shape)))
            shape2 = shape1[self.axis]
            shape1.remove(shape2)
            shape3 = [shape2] + shape1
            temp_x = numpy.transpose(x, shape3)
            temp_x = temp_x.reshape(temp_x.shape[0], -1)

        x_min = numpy.amin(temp_x, axis=1)
        x_max = numpy.amax(temp_x, axis=1)
        best_min, best_max = self._caculate(temp_x, x_min, x_max)

        if self._cumulate == 0:
            self.min = best_min
            self.max = best_max
        else:
            # self.min = self.min + self.averaging * (best_min - self.min)
            # self.max = self.max + self.averaging * (best_max - self.max)
            self.min = (self.min * self._cumulate + best_min) / (self._cumulate + 1)
            self.max = (self.max * self._cumulate + best_max) / (self._cumulate + 1)

        self._cumulate = self._cumulate + 1

    @property
    def min_max(self):
        if self.min.size == 1:
            self.min = self.min[0]
        if self.max.size == 1:
            self.max = self.max[0]
        return {"min": self.min, "max": self.max}


class Threshold:
    """threshold"""

    MinMax = MinMax
    MovingAverageMinMax = MovingAverageMinMax
    Percentile = Percentile
    L2Norm = L2Norm
    RelativeEntropy = RelativeEntropy
    PercentileAbs = PercentileAbs
    KLDAbs = KLDAbs
    DistanceLinearSearch = DistanceLinearSearch
