// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <type_traits>

namespace mlx::core {

template <typename T>
inline constexpr bool is_floating_v =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>;

// Throw exception if the HIP API does not succeed.
void check_hip_error(const char* name, hipError_t err);

#define CHECK_HIP_ERROR(cmd) check_hip_error(#cmd, (cmd))

// Check rocBLAS error
void check_rocblas_error(const char* name, rocblas_status err);

#define CHECK_ROCBLAS_ERROR(cmd) check_rocblas_error(#cmd, (cmd))

// Return the 3d block_dim fit for total_threads.
dim3 get_block_dim(dim3 total_threads, int pow2 = 10);

std::string get_primitive_string(Primitive* primitive);

} // namespace mlx::core