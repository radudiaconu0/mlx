// Copyright © 2025 Apple Inc.

#include "mlx/array.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/dtype_utils.h"
#include "mlx/backend/metal/copy.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

#include <rocblas/rocblas.h>
#include <fmt/format.h>

#include <numeric>
#include <sstream>

namespace mlx::core {

namespace {

auto collapse_batches(const array& a, const array& b) {
  // Get and check the shape for the batched dims
  Shape A_bshape{a.shape().begin(), a.shape().end() - 2};
  Shape B_bshape{b.shape().begin(), b.shape().end() - 2};
  if (A_bshape != B_bshape) {
    std::ostringstream msg;
    msg << "[matmul] Got matrices with incorrectly broadcasted shapes: " << "A "
        << a.shape() << ", B " << b.shape() << ".";
    throw std::runtime_error(msg.str());
  }

  Strides A_bstride{a.strides().begin(), a.strides().end() - 2};
  Strides B_bstride{b.strides().begin(), b.strides().end() - 2};

  auto [batch_shape, batch_strides] =
      collapse_contiguous_dims(A_bshape, std::vector{A_bstride, B_bstride});

  auto a_batch_strides = batch_strides[0];
  auto b_batch_strides = batch_strides[1];

  if (batch_shape.empty()) {
    batch_shape.push_back(1);
    a_batch_strides.push_back(0);
    b_batch_strides.push_back(0);
  }

  return std::make_tuple(batch_shape, a_batch_strides, b_batch_strides);
}

std::tuple<bool, int64_t, array>
check_transpose(std::vector<array>& copies, const Stream& s, const array& arr) {
  auto stx = arr.strides()[arr.ndim() - 2];
  auto sty = arr.strides()[arr.ndim() - 1];
  if (sty == 1 && stx == arr.shape(-1)) {
    return std::make_tuple(false, stx, arr);
  } else if (stx == 1 && sty == arr.shape(-2)) {
    return std::make_tuple(true, sty, arr);
  } else {
    array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
    copy_gpu(arr, arr_copy, CopyType::General, s);
    copies.push_back(arr_copy);
    return std::make_tuple(false, arr.shape(-1), arr_copy);
  }
}

// Default workspace size for rocBLAS operations
uint64_t workspace_size = 4 * 1024 * 1024;

// RocmMatMul replaces CudaMatMul, using rocBLAS instead of cublasLt
class RocmMatMul {
public:
  RocmMatMul(
      mxrocm::CommandEncoder& encoder,
      Dtype ab_dtype,
      bool a_transposed,
      uint64_t a_rows,
      uint64_t a_cols,
      int64_t lda,
      bool b_transposed,
      uint64_t b_rows,
      uint64_t b_cols,
      int64_t ldb,
      int32_t batch_count,
      int64_t a_batch_stride,
      int64_t b_batch_stride)
      : handle_(encoder.device().rocblas_handle()),
        a_transposed_(a_transposed),
        b_transposed_(b_transposed),
        a_rows_(a_rows),
        a_cols_(a_cols),
        b_cols_(b_cols),
        lda_(lda),
        ldb_(ldb),
        batch_count_(batch_count),
        a_batch_stride_(a_batch_stride),
        b_batch_stride_(b_batch_stride),
        out_batch_stride_(a_rows * b_cols),
        dtype_(ab_dtype) {
  }

  template <typename T>
  void Run(mxrocm::CommandEncoder& encoder, array& workspace, T* out, T* a, T* b) {
    // Set alpha and beta scalars for GEMM
    T alpha = 1;
    T beta = 0;

    // Launch the appropriate rocBLAS GEMM
    encoder.launch_kernel([&](hipStream_t stream) {
      // Set stream for rocBLAS
      CHECK_ROCBLAS_ERROR(rocblas_set_stream(handle_, stream));

      // Use regular or strided batched GEMM based on batch count
      if (batch_count_ > 1) {
        this->LaunchStridedBatchedGemm(stream, alpha, beta, a, b, out);
      } else {
        this->LaunchGemm(stream, alpha, beta, a, b, out);
      }
    });
  }

private:
  // Launch standard GEMM
  template <typename T>
  void LaunchGemm(hipStream_t stream, T alpha, T beta, T* a, T* b, T* out) {
    // Convert operation types for rocBLAS
    rocblas_operation transA = a_transposed_ ? rocblas_operation_transpose : rocblas_operation_none;
    rocblas_operation transB = b_transposed_ ? rocblas_operation_transpose : rocblas_operation_none;

    // Select appropriate rocBLAS function based on data type
    if constexpr (std::is_same_v<T, float>) {
      CHECK_ROCBLAS_ERROR(rocblas_sgemm(
          handle_, transA, transB,
          a_rows_, b_cols_, a_cols_,
          &alpha, a, lda_, b, ldb_,
          &beta, out, a_rows_));
    }
    else if constexpr (std::is_same_v<T, double>) {
      CHECK_ROCBLAS_ERROR(rocblas_dgemm(
          handle_, transA, transB,
          a_rows_, b_cols_, a_cols_,
          &alpha, a, lda_, b, ldb_,
          &beta, out, a_rows_));
    }
    else if constexpr (std::is_same_v<T, __half>) {
      CHECK_ROCBLAS_ERROR(rocblas_hgemm(
          handle_, transA, transB,
          a_rows_, b_cols_, a_cols_,
          &alpha, a, lda_, b, ldb_,
          &beta, out, a_rows_));
    }
    else {
      throw std::runtime_error(fmt::format(
          "Unsupported data type in RocmMatMul: {}", dtype_to_string(dtype_)));
    }
  }

  // Launch strided batched GEMM
  template <typename T>
  void LaunchStridedBatchedGemm(hipStream_t stream, T alpha, T beta, T* a, T* b, T* out) {
    // Convert operation types for rocBLAS
    rocblas_operation transA = a_transposed_ ? rocblas_operation_transpose : rocblas_operation_none;
    rocblas_operation transB = b_transposed_ ? rocblas_operation_transpose : rocblas_operation_none;

    // Select appropriate rocBLAS function based on data type
    if constexpr (std::is_same_v<T, float>) {
      CHECK_ROCBLAS_ERROR(rocblas_sgemm_strided_batched(
          handle_, transA, transB,
          a_rows_, b_cols_, a_cols_,
          &alpha, a, lda_, a_batch_stride_,
          b, ldb_, b_batch_stride_,
          &beta, out, a_rows_, out_batch_stride_,
          batch_count_));
    }
    else if constexpr (std::is_same_v<T, double>) {
      CHECK_ROCBLAS_ERROR(rocblas_dgemm_strided_batched(
          handle_, transA, transB,
          a_rows_, b_cols_, a_cols_,
          &alpha, a, lda_, a_batch_stride_,
          b, ldb_, b_batch_stride_,
          &beta, out, a_rows_, out_batch_stride_,
          batch_count_));
    }
    else if constexpr (std::is_same_v<T, __half>) {
      CHECK_ROCBLAS_ERROR(rocblas_hgemm_strided_batched(
          handle_, transA, transB,
          a_rows_, b_cols_, a_cols_,
          &alpha, a, lda_, a_batch_stride_,
          b, ldb_, b_batch_stride_,
          &beta, out, a_rows_, out_batch_stride_,
          batch_count_));
    }
    else {
      throw std::runtime_error(fmt::format(
          "Unsupported data type in RocmMatMul: {}", dtype_to_string(dtype_)));
    }
  }

  // Member variables
  rocblas_handle handle_;
  bool a_transposed_;
  bool b_transposed_;
  uint64_t a_rows_;
  uint64_t a_cols_;
  uint64_t b_cols_;
  int64_t lda_;
  int64_t ldb_;
  int32_t batch_count_;
  int64_t a_batch_stride_;
  int64_t b_batch_stride_;
  int64_t out_batch_stride_;
  Dtype dtype_;
};

} // namespace

void Matmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  if (!issubdtype(out.dtype(), floating)) {
    throw std::runtime_error(
        "[matmul] Does not yet support non-floating point types.");
  }
  auto& s = stream();
  auto& encoder = mxrocm::get_command_encoder(s);

  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];
  // Return 0s if either input is empty.
  if (a_pre.size() == 0 || b_pre.size() == 0) {
    array zero = array(0, a_pre.dtype());
    fill_gpu(zero, out, s);
    encoder.add_temporary(std::move(zero));
    return;
  }

  out.set_data(allocator::malloc(out.nbytes()));

  /////////////////////////////////////////////////////////////////////////////
  // Init checks and prep

  int M = a_pre.shape(-2);
  int N = b_pre.shape(-1);
  int K = a_pre.shape(-1);

  // Keep a vector with copies to be cleared in the completed buffer to release
  // the arrays
  std::vector<array> copies;
  auto [a_transposed, lda, a] = check_transpose(copies, s, a_pre);
  auto [b_transposed, ldb, b] = check_transpose(copies, s, b_pre);

  for (auto& temp : copies) {
    encoder.add_temporary(temp);
  }

  /////////////////////////////////////////////////////////////////////////////
  // Check and collapse batch dimensions

  auto [batch_shape, a_batch_strides, b_batch_strides] = collapse_batches(a, b);

  auto batch_count = out.size() / (size_t(M) * size_t(N));

  // Collapse batches into M if needed
  if (batch_count > 1 && !a_transposed && batch_shape.size() == 1 &&
      a.strides()[a.ndim() - 2] == K && a_batch_strides.back() == M * K &&
      b_batch_strides.back() == 0) {
    M *= batch_shape.back();
    batch_count = 1;

    a_batch_strides = {0};
    b_batch_strides = {0};
    batch_shape = {1};
  }

  /////////////////////////////////////////////////////////////////////////////
  // Invoke rocBLAS

  if (batch_shape.size() > 1) {
    // TODO: Implement by looping the matmul
    throw std::runtime_error(
        "Non-contiguous batch gemm is not implemented in ROCm backend.");
  }

  array workspace(
      allocator::malloc(workspace_size),
      {static_cast<int>(workspace_size)},
      uint8);
  encoder.add_temporary(workspace);

  RocmMatMul matmul(
      encoder,
      a.dtype(),
      a_transposed,
      M,
      K,
      lda,
      b_transposed,
      K,
      N,
      ldb,
      batch_count,
      a_batch_strides[0],
      b_batch_strides[0]);

  MLX_SWITCH_FLOAT_TYPES_CHECKED(a.dtype(), "matmul", CTYPE, [&]() {
    using ABType = hip_type_t<CTYPE>;
    matmul.Run(
        encoder,
        workspace,
        out.data<ABType>(),
        a.data<ABType>(),
        b.data<ABType>());
  });
}

} // namespace mlx::core