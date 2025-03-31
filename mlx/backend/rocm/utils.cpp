// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/utils.h"
#include "mlx/primitives.h"

#include <fmt/format.h>

namespace mlx::core {

void check_hip_error(const char* name, hipError_t err) {
  if (err != hipSuccess) {
    throw std::runtime_error(
        fmt::format("{} failed: {}", name, hipGetErrorString(err)));
  }
}

void check_rocblas_error(const char* name, rocblas_status err) {
  if (err != rocblas_status_success) {
    throw std::runtime_error(
        fmt::format("{} failed: rocBLAS error code {}", name, static_cast<int>(err)));
  }
}

// Implementation identical to the CUDA version
dim3 get_block_dim(dim3 total_threads, int pow2) {
  int pows[3] = {0, 0, 0};
  int sum = 0;
  while (true) {
    int presum = sum;
    // Check all the pows
    if (total_threads.x >= (1 << (pows[0] + 1))) {
      pows[0]++;
      sum++;
    }
    if (sum == 10) {
      break;
    }
    if (total_threads.y >= (1 << (pows[1] + 1))) {
      pows[1]++;
      sum++;
    }
    if (sum == 10) {
      break;
    }
    if (total_threads.z >= (1 << (pows[2] + 1))) {
      pows[2]++;
      sum++;
    }
    if (sum == presum || sum == pow2) {
      break;
    }
  }
  return {1u << pows[0], 1u << pows[1], 1u << pows[2]};
}

std::string get_primitive_string(Primitive* primitive) {
  std::ostringstream op_t;
  primitive->print(op_t);
  return op_t.str();
}

} // namespace mlx::core