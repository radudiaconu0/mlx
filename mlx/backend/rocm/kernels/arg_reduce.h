// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/rocm/kernels/iterators/strided_iterator.h"
#include "mlx/backend/rocm/kernels/utils.h"

#include <hip/hip_cooperative_groups.h>
#include <hipcub/block/block_load.hpp>
#include <hipcub/block/block_reduce.hpp>

namespace mlx::core::mxrocm {

namespace cg = cooperative_groups;

template <typename U>
struct IndexValPair {
  uint32_t index;
  U val;
};

template <typename U>
struct ArgMin {
  static constexpr U init = Limits<U>::max;

  __device__ IndexValPair<U> operator()(
      const IndexValPair<U>& best,
      const IndexValPair<U>& current) {
    if (best.val > current.val ||
        (best.val == current.val && best.index > current.index)) {
      return current;
    } else {
      return best;
    }
  }

  template <int N>
  __device__ IndexValPair<U>
  reduce_many(IndexValPair<U> best, U (&vals)[N], uint32_t offset) {
    HIP_UNROLL for (int i = 0; i < N; i++) {
      if (vals[i] < best.val) {
        best.val = vals[i];
        best.index = offset + i;
      }
    }
    return best;
  }
};

template <typename U>
struct ArgMax {
  static constexpr U init = Limits<U>::min;

  __device__ IndexValPair<U> operator()(
      const IndexValPair<U>& best,
      const IndexValPair<U>& current) {
    if (best.val < current.val ||
        (best.val == current.val && best.index > current.index)) {
      return current;
    } else {
      return best;
    }
  }

  template <int N>
  __device__ IndexValPair<U>
  reduce_many(IndexValPair<U> best, U (&vals)[N], uint32_t offset) {
    HIP_UNROLL for (int i = 0; i < N; i++) {
      if (vals[i] > best.val) {
        best.val = vals[i];
        best.index = offset + i;
      }
    }
    return best;
  }
};

template <typename U>
inline __device__ IndexValPair<U> warp_shuffle_down(
    const cg::thread_block_tile<WARP_SIZE>& g,
    const IndexValPair<U>& data,
    int delta) {
  // For ROCm, use explicit shuffle operations for structure members
  uint32_t index = g.shfl_down(data.index, delta);
  U val = g.shfl_down(data.val, delta);
  return {index, val};
}

template <typename T, typename Op, int BLOCK_DIM, int N_READS = 4>
__global__ void arg_reduce_general(
    const T* in,
    uint32_t* out,
    const Shape shape,
    const Strides in_strides,
    const Strides out_strides,
    size_t ndim,
    int64_t axis_stride,
    size_t axis_size) {
  // Shapes and strides *do not* contain the reduction axis. The reduction size
  // and stride are provided in axis_stride and axis_size.
  //
  // Note: in shape == out shape with this convention.
  Op op;

  // Compute the input/output index. There is one beginning and one output for
  // the whole block.
  auto elem = blockIdx.x;
  auto in_idx = elem_to_loc(elem, shape.data(), in_strides.data(), ndim);
  auto out_idx = elem_to_loc(elem, shape.data(), out_strides.data(), ndim);

  IndexValPair<T> best{0, Op::init};

  auto block = cg::this_thread_block();
  for (uint32_t r = 0; r < ceil_div(axis_size, BLOCK_DIM * N_READS); r++) {
    T vals[N_READS];
    auto index = r * BLOCK_DIM + block.thread_rank();
    hipcub::BlockLoad<T, BLOCK_DIM, N_READS, hipcub::BLOCK_LOAD_DIRECT>(
        index,
        make_strided_iterator(in + in_idx, axis_stride),
        vals,
        axis_size,
        Op::init);
    best = op.reduce_many(best, vals, index * N_READS);
  }

  typedef hipcub::BlockReduce<IndexValPair<T>, BLOCK_DIM> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp;

  best = BlockReduceT(temp).Reduce(best, op);

  if (block.thread_rank() == 0) {
    out[out_idx] = best.index;
  }
}

} // namespace mlx::core::mxrocm