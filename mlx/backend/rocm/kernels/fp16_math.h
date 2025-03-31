// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <limits>
#include <type_traits>

namespace mlx::core::mxrocm {

///////////////////////////////////////////////////////////////////////////////
// Unary ops for half types.
///////////////////////////////////////////////////////////////////////////////

// Define unary operations for __half and hip_bfloat16 types
// For ROCm, some of these need to use the float conversion method

// Basic conversion helpers
template <typename T>
__forceinline__ __device__ float to_float(T x) {
  if constexpr (std::is_same<T, __half>::value) {
    return __half2float(x);
  } else if constexpr (std::is_same<T, hip_bfloat16>::value) {
    return __bfloat162float(x);
  } else {
    return x;
  }
}

template <typename T>
__forceinline__ __device__ T from_float(float x) {
  if constexpr (std::is_same<T, __half>::value) {
    return __float2half(x);
  } else if constexpr (std::is_same<T, hip_bfloat16>::value) {
    return __float2bfloat16(x);
  } else {
    return x;
  }
}

// Define operations that have direct ROCm implementations
#define MLX_DEFINE_HALF_OP(NAME, IMPL)                                   \
  template <typename T>                                                  \
  __forceinline__ __device__ auto NAME(T x) {                            \
    if constexpr (std::is_same<T, __half>::value ||                      \
                  std::is_same<T, hip_bfloat16>::value) {                \
      return from_float<T>(IMPL(to_float(x)));                           \
    } else {                                                             \
      return IMPL(x);                                                    \
    }                                                                    \
  }

// Define operations that need a fallback to float
MLX_DEFINE_HALF_OP(abs, ::fabsf)
MLX_DEFINE_HALF_OP(ceil, ::ceilf)
MLX_DEFINE_HALF_OP(cos, ::cosf)
MLX_DEFINE_HALF_OP(exp, ::expf)
MLX_DEFINE_HALF_OP(floor, ::floorf)
MLX_DEFINE_HALF_OP(log, ::logf)
MLX_DEFINE_HALF_OP(log2, ::log2f)
MLX_DEFINE_HALF_OP(log10, ::log10f)
MLX_DEFINE_HALF_OP(rint, ::rintf)
MLX_DEFINE_HALF_OP(rsqrt, ::rsqrtf)
MLX_DEFINE_HALF_OP(sin, ::sinf)
MLX_DEFINE_HALF_OP(sqrt, ::sqrtf)
MLX_DEFINE_HALF_OP(acos, ::acosf)
MLX_DEFINE_HALF_OP(acosh, ::acoshf)
MLX_DEFINE_HALF_OP(asin, ::asinf)
MLX_DEFINE_HALF_OP(asinh, ::asinhf)
MLX_DEFINE_HALF_OP(atan, ::atanf)
MLX_DEFINE_HALF_OP(atanh, ::atanhf)
MLX_DEFINE_HALF_OP(cosh, ::coshf)
MLX_DEFINE_HALF_OP(log1p, ::log1pf)
MLX_DEFINE_HALF_OP(sinh, ::sinhf)
MLX_DEFINE_HALF_OP(tan, ::tanf)
MLX_DEFINE_HALF_OP(tanh, ::tanhf)

#undef MLX_DEFINE_HALF_OP

// Define a specialized isnan for half-precision types
template <typename T>
__forceinline__ __device__ bool isnan(T x) {
  if constexpr (std::is_same<T, __half>::value) {
    return __half2float(x) != __half2float(x);
  } else if constexpr (std::is_same<T, hip_bfloat16>::value) {
    return __bfloat162float(x) != __bfloat162float(x);
  } else if constexpr (std::is_same<T, hipFloatComplex>::value) {
    return ::isnan(x.x) || ::isnan(x.y);
  } else {
    return ::isnan(x);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Binary ops for half types.
///////////////////////////////////////////////////////////////////////////////

// Binary operations with explicit conversion to float
#define MLX_DEFINE_BINARY_OP(NAME, IMPL)                                 \
  template <typename T>                                                  \
  __forceinline__ __device__ auto NAME(T x, T y) {                       \
    if constexpr (std::is_same<T, __half>::value ||                      \
                  std::is_same<T, hip_bfloat16>::value) {                \
      return from_float<T>(IMPL(to_float(x), to_float(y)));              \
    } else {                                                             \
      return IMPL(x, y);                                                 \
    }                                                                    \
  }

MLX_DEFINE_BINARY_OP(max, ::fmaxf)
MLX_DEFINE_BINARY_OP(min, ::fminf)
MLX_DEFINE_BINARY_OP(fmod, ::fmodf)

#undef MLX_DEFINE_BINARY_OP

///////////////////////////////////////////////////////////////////////////////
// Comparison and arithmetic operators for half types
///////////////////////////////////////////////////////////////////////////////

// Comparison operators for half-precision types
#define MLX_DEFINE_HALF_CMP(OP)                                           \
  __device__ inline bool operator OP(__half a, __half b) {                \
    return __half2float(a) OP __half2float(b);                            \
  }                                                                       \
  __device__ inline bool operator OP(hip_bfloat16 a, hip_bfloat16 b) {    \
    return __bfloat162float(a) OP __bfloat162float(b);                    \
  }

// Arithmetic operators for half-precision types
#define MLX_DEFINE_HALF_OP(OP)                                            \
  __device__ inline __half operator OP(__half a, __half b) {              \
    return __float2half(__half2float(a) OP __half2float(b));              \
  }                                                                       \
  __device__ inline hip_bfloat16 operator OP(hip_bfloat16 a, hip_bfloat16 b) { \
    return __float2bfloat16(__bfloat162float(a) OP __bfloat162float(b));  \
  }

// Define comparison operators
MLX_DEFINE_HALF_CMP(==)
MLX_DEFINE_HALF_CMP(!=)
MLX_DEFINE_HALF_CMP(<)
MLX_DEFINE_HALF_CMP(<=)
MLX_DEFINE_HALF_CMP(>)
MLX_DEFINE_HALF_CMP(>=)

// Define arithmetic operators
MLX_DEFINE_HALF_OP(+)
MLX_DEFINE_HALF_OP(-)
MLX_DEFINE_HALF_OP(*)
MLX_DEFINE_HALF_OP(/)

#undef MLX_DEFINE_HALF_CMP
#undef MLX_DEFINE_HALF_OP

} // namespace mlx::core::mxrocm