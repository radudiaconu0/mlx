// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/dtype_utils.h"

#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_complex.h>

namespace mlx::core {

// Maps CPU types to HIP types.
template <typename T>
struct CTypeToHipType {
  using type = T;
};

template <>
struct CTypeToHipType<float16_t> {
  using type = __half;
};

template <>
struct CTypeToHipType<bfloat16_t> {
  using type = hip_bfloat16;
};

template <>
struct CTypeToHipType<complex64_t> {
  using type = hipFloatComplex;
};

template <typename T>
using hip_type_t = typename CTypeToHipType<T>::type;

} // namespace mlx::core