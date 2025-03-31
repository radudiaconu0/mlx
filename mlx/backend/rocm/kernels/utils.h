// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/rocm/dtype_utils.h"
#include "mlx/backend/rocm/kernels/hipcomplex_math.h"
#include "mlx/backend/rocm/kernels/fp16_math.h"

#include <array>
#include <limits>

namespace mlx::core {

// The clang-format is not friendly with "#pragma unroll" so use a macro.
#define HIP_UNROLL _Pragma("unroll")

// All AMD hardware has fixed 64 warp (wavefront) size, but we maintain WARP_SIZE
// for compatibility with existing code. Note that this may need adjustment
// for wavefront-specific optimizations.
#define WARP_SIZE 64

// The maximum number of threads per block.
#define MAX_BLOCK_DIM 1024

// (a + b - 1) / b
template <typename T>
inline constexpr T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

inline constexpr dim3 ceil_div(dim3 a, dim3 b) {
  return {ceil_div(a.x, b.x), ceil_div(a.y, b.y), ceil_div(a.z, b.z)};
}

namespace mxrocm {

///////////////////////////////////////////////////////////////////////////////
// HIP kernel utils
///////////////////////////////////////////////////////////////////////////////

// To pass shape/strides to kernels via constant memory, their size must be
// known at compile time.
#define MAX_NDIM 8

using Shape = std::array<int32_t, MAX_NDIM>;
using Strides = std::array<int64_t, MAX_NDIM>;

// Utility to copy data from vector to array in host.
template <typename T>
inline std::array<T, MAX_NDIM> const_param(const std::vector<T>& vec) {
  if (vec.size() > MAX_NDIM) {
    throw std::runtime_error("ndim can not be larger than 8.");
  }
  std::array<T, MAX_NDIM> result;
  std::copy_n(vec.begin(), vec.size(), result.begin());
  return result;
}

// Helper macros for dispatch macros (see below).
#define MLX_INTERNAL_IF_CASE(DIM, THREADS, BLOCK_DIM, ...) \
  if (THREADS <= DIM) {                                    \
    constexpr uint32_t BLOCK_DIM = DIM;                    \
    __VA_ARGS__;                                           \
  }

#define MLX_INTERNAL_IF_CASE_DIMS(...)   \
  MLX_INTERNAL_IF_CASE(32, __VA_ARGS__)  \
  MLX_INTERNAL_IF_CASE(64, __VA_ARGS__)  \
  MLX_INTERNAL_IF_CASE(128, __VA_ARGS__) \
  MLX_INTERNAL_IF_CASE(256, __VA_ARGS__) \
  MLX_INTERNAL_IF_CASE(512, __VA_ARGS__) \
  MLX_INTERNAL_IF_CASE(1024, __VA_ARGS__)

// Some kernels use hipCUB which requires block_dim to be known at compile-time,
// use this to dispatch fixed block_dim from dynamic total_threads.
#define MLX_GET_BLOCK_DIM(THREADS, BLOCK_DIM, ...) \
  MLX_INTERNAL_IF_CASE_DIMS(THREADS, BLOCK_DIM, __VA_ARGS__)

///////////////////////////////////////////////////////////////////////////////
// Type limits utils
///////////////////////////////////////////////////////////////////////////////

template <typename U>
struct Limits {
  static constexpr U max = std::numeric_limits<U>::max();
  static constexpr U min = std::numeric_limits<U>::min();
  static constexpr U finite_max = std::numeric_limits<U>::max();
  static constexpr U finite_min = std::numeric_limits<U>::min();
};

template <>
struct Limits<bool> {
  static constexpr bool max = true;
  static constexpr bool min = false;
};

template <>
struct Limits<hipFloatComplex> {
  static constexpr hipFloatComplex max = {
      std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity()};
  static constexpr hipFloatComplex min = {
      -std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity()};
};

// Like MLX_FORALL_FLOAT_TYPES but use HIP types.
#define MLX_FORALL_HIP_FLOAT_TYPES(_) \
  _(float, float32)                   \
  _(double, float64)                  \
  _(__half, float16)                  \
  _(hip_bfloat16, bfloat16)

// Specialize limits for half types
#define SPECIALIZE_FloatLimits(CPP_TYPE, DTYPE)                          \
  template <>                                                            \
  struct Limits<CPP_TYPE> {                                              \
    static constexpr CPP_TYPE max = infinite_value<CPP_TYPE>();          \
    static constexpr CPP_TYPE min = negative_infinite_value<CPP_TYPE>(); \
    static constexpr CPP_TYPE finite_max = finite_max_value<CPP_TYPE>(); \
    static constexpr CPP_TYPE finite_min = finite_min_value<CPP_TYPE>(); \
  };

MLX_FORALL_HIP_FLOAT_TYPES(SPECIALIZE_FloatLimits)

#undef SPECIALIZE_FloatLimits

///////////////////////////////////////////////////////////////////////////////
// Indexing utils
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// Single Array with generic dims

template <typename IdxT = int64_t>
inline __host__ __device__ IdxT
elem_to_loc(IdxT elem, const int* shape, const int64_t* strides, int ndim) {
  IdxT loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * IdxT(strides[i]);
    elem /= shape[i];
  }
  return loc;
}

// Non templated version to handle arbitrary dims
template <typename IdxT = int64_t>
inline __host__ __device__ IdxT
elem_to_loc(uint3 elem, const int* shape, const int64_t* strides, int ndim) {
  IdxT loc =
      elem.x * IdxT(strides[ndim - 1]) + elem.y * IdxT(strides[ndim - 2]);
  for (int d = ndim - 3; d >= 0; --d) {
    loc += (elem.z % shape[d]) * IdxT(strides[d]);
    elem.z /= shape[d];
  }
  return loc;
}

///////////////////////////////////////////////////////////////////////////////
// Single Array with fixed N dims

template <typename IdxT = int64_t>
inline __host__ __device__ IdxT elem_to_loc_1(uint elem, int64_t stride) {
  return elem * IdxT(stride);
}

template <typename IdxT = int64_t>
inline __host__ __device__ IdxT elem_to_loc_2(uint2 elem, int64_t strides[2]) {
  return elem.x * IdxT(strides[1]) + elem.y * IdxT(strides[0]);
}

template <typename IdxT = int64_t>
inline __host__ __device__ IdxT elem_to_loc_3(uint3 elem, int64_t strides[3]) {
  return elem.x * IdxT(strides[2]) + elem.y * IdxT(strides[1]) +
      elem.z * IdxT(strides[0]);
}

// Helper functions for half/bfloat16 values
template <typename T>
__host__ __device__ constexpr T zero_value() {
  if constexpr (std::is_same<T, __half>::value) {
    return __float2half(0.0f);
  } else if constexpr (std::is_same<T, hip_bfloat16>::value) {
    return __float2bfloat16(0.0f);
  } else if constexpr (std::is_same<T, hipFloatComplex>::value) {
    return make_hipFloatComplex(0.0f, 0.0f);
  } else {
    return 0;
  }
}

template <typename T>
__host__ __device__ constexpr T one_value() {
  if constexpr (std::is_same<T, __half>::value) {
    return __float2half(1.0f);
  } else if constexpr (std::is_same<T, hip_bfloat16>::value) {
    return __float2bfloat16(1.0f);
  } else if constexpr (std::is_same<T, hipFloatComplex>::value) {
    return make_hipFloatComplex(1.0f, 0.0f);
  } else {
    return 1;
  }
}

template <typename T>
__host__ __device__ constexpr T infinite_value() {
  if constexpr (std::is_same<T, __half>::value) {
    return __float2half(std::numeric_limits<float>::infinity());
  } else if constexpr (std::is_same<T, hip_bfloat16>::value) {
    return __float2bfloat16(std::numeric_limits<float>::infinity());
  } else {
    return std::numeric_limits<T>::infinity();
  }
}

template <typename T>
__host__ __device__ constexpr T negative_infinite_value() {
  if constexpr (std::is_same<T, __half>::value) {
    return __float2half(-std::numeric_limits<float>::infinity());
  } else if constexpr (std::is_same<T, hip_bfloat16>::value) {
    return __float2bfloat16(-std::numeric_limits<float>::infinity());
  } else {
    return -std::numeric_limits<T>::infinity();
  }
}

template <typename T>
__host__ __device__ constexpr T finite_max_value() {
  if constexpr (std::is_same<T, __half>::value) {
    return __float2half(65504.0f); // Maximum finite value for half precision
  } else if constexpr (std::is_same<T, hip_bfloat16>::value) {
    return __float2bfloat16(3.38953e+38f); // Maximum finite value for bfloat16
  } else {
    return std::numeric_limits<T>::max();
  }
}

template <typename T>
__host__ __device__ constexpr T finite_min_value() {
  if constexpr (std::is_same<T, __half>::value) {
    return __float2half(-65504.0f); // Minimum finite value for half precision
  } else if constexpr (std::is_same<T, hip_bfloat16>::value) {
    return __float2bfloat16(-3.38953e+38f); // Minimum finite value for bfloat16
  } else {
    return std::numeric_limits<T>::lowest();
  }
}

} // namespace mxrocm

} // namespace mlx::core