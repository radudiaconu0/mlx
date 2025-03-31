// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/utils.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernels/iterators/general_iterator.h"
#include "mlx/backend/rocm/kernels/unary_ops.h"
#include "mlx/primitives.h"

#include <rocthrust/device_ptr.h>
#include <rocthrust/transform.h>

namespace mlx::core {

namespace {

template <typename Op, typename In, typename Out>
constexpr bool is_supported_unary_op() {
  if (std::is_same<Op, mxrocm::Abs>::value || std::is_same<Op, mxrocm::Negative>::value ||
      std::is_same<Op, mxrocm::Sign>::value) {
    return std::is_same<In, Out>::value;
  }
  if (std::is_same<Op, mxrocm::ArcCos>::value ||
      std::is_same<Op, mxrocm::ArcCosh>::value ||
      std::is_same<Op, mxrocm::ArcSin>::value ||
      std::is_same<Op, mxrocm::ArcSinh>::value ||
      std::is_same<Op, mxrocm::ArcTan>::value ||
      std::is_same<Op, mxrocm::ArcTanh>::value || std::is_same<Op, mxrocm::Erf>::value ||
      std::is_same<Op, mxrocm::ErfInv>::value || std::is_same<Op, mxrocm::Expm1>::value ||
      std::is_same<Op, mxrocm::Log1p>::value || std::is_same<Op, mxrocm::Log>::value ||
      std::is_same<Op, mxrocm::Log2>::value || std::is_same<Op, mxrocm::Log10>::value ||
      std::is_same<Op, mxrocm::Sigmoid>::value || std::is_same<Op, mxrocm::Sqrt>::value ||
      std::is_same<Op, mxrocm::Rsqrt>::value) {
    return std::is_same<In, Out>::value && is_floating_v<In>;
  }
  if (std::is_same<Op, mxrocm::BitwiseInvert>::value) {
    return std::is_same<In, Out>::value && std::is_integral<In>::value &&
        !std::is_same<In, bool>::value;
  }
  if (std::is_same<Op, mxrocm::Ceil>::value || std::is_same<Op, mxrocm::Floor>::value ||
      std::is_same<Op, mxrocm::Square>::value) {
    return std::is_same<In, Out>::value && !std::is_same<In, hipFloatComplex>::value;
  }
  if (std::is_same<Op, mxrocm::Conjugate>::value) {
    return std::is_same<In, Out>::value && std::is_same<In, hipFloatComplex>::value;
  }
  if (std::is_same<Op, mxrocm::Cos>::value || std::is_same<Op, mxrocm::Cosh>::value ||
      std::is_same<Op, mxrocm::Exp>::value || std::is_same<Op, mxrocm::Round>::value ||
      std::is_same<Op, mxrocm::Sin>::value || std::is_same<Op, mxrocm::Sinh>::value ||
      std::is_same<Op, mxrocm::Tan>::value || std::is_same<Op, mxrocm::Tanh>::value) {
    return std::is_same<In, Out>::value &&
        (is_floating_v<In> || std::is_same<In, hipFloatComplex>::value);
  }
  if (std::is_same<Op, mxrocm::Imag>::value || std::is_same<Op, mxrocm::Real>::value) {
    return std::is_same<In, hipFloatComplex>::value && std::is_same<Out, float>::value;
  }
  if (std::is_same<Op, mxrocm::LogicalNot>::value) {
    return std::is_same<In, Out>::value && std::is_same<In, bool>::value;
  }
  return false;
}

template <typename Op>
void unary_op_gpu_inplace(
    const std::vector<array>& inputs,
    array& out,
    const std::string& op,
    const Stream& s) {
  auto& in = inputs[0];
  if (in.size() == 0) {
    return;
  }

  auto& encoder = mxrocm::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  encoder.launch_thrust([&](auto policy) {
    MLX_SWITCH_ALL_TYPES(in.dtype(), CTYPE_IN, [&]() {
      MLX_SWITCH_ALL_TYPES(out.dtype(), CTYPE_OUT, [&]() {
        if constexpr (is_supported_unary_op<Op, CTYPE_IN, CTYPE_OUT>()) {
          using InType = hip_type_t<CTYPE_IN>;
          using OutType = hip_type_t<CTYPE_OUT>;
          if (in.flags().contiguous) {
            rocthrust::transform(
                policy,
                rocthrust::device_pointer_cast(in.data<InType>()),
                rocthrust::device_pointer_cast(in.data<InType>() + in.data_size()),
                rocthrust::device_pointer_cast(out.data<OutType>()),
                Op());
          } else {
            auto [shape, strides] = collapse_contiguous_dims(in);
            auto [in_begin, in_end] = mxrocm::make_general_iterators<int64_t>(
                rocthrust::device_pointer_cast(in.data<InType>()),
                in.data_size(),
                shape,
                strides);
            auto out_begin = rocthrust::device_pointer_cast(out.data<OutType>());
            rocthrust::transform(policy, in_begin, in_end, out_begin, Op());
          }
        } else {
          throw std::runtime_error(fmt::format(
              "Cannot do unary op {} on input of {} with output of {}.",
              op,
              dtype_to_string(in.dtype()),
              dtype_to_string(out.dtype())));
        }
      });
    });
  });
}

template <typename Op>
void unary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const std::string& op,
    const Stream& s) {
  auto& in = inputs[0];
  if (in.flags().contiguous) {
    if (in.is_donatable() && in.itemsize() == out.itemsize()) {
      out.copy_shared_buffer(in);
    } else {
      out.set_data(
          allocator::malloc(in.data_size() * out.itemsize()),
          in.data_size(),
          in.strides(),
          in.flags());
    }
  } else {
    out.set_data(allocator::malloc(out.nbytes()));
  }
  unary_op_gpu_inplace<Op>(inputs, out, op, s);
}

} // namespace

#define UNARY_GPU(func)                                                     \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) {       \
    auto& s = out.primitive().stream();                                     \
    unary_op_gpu<mxrocm::func>(inputs, out, get_primitive_string(this), s); \
  }

UNARY_GPU(Abs)
UNARY_GPU(ArcCos)
UNARY_GPU(ArcCosh)
UNARY_GPU(ArcSin)
UNARY_GPU(ArcSinh)
UNARY_GPU(ArcTan)
UNARY_GPU(ArcTanh)
UNARY_GPU(BitwiseInvert)
UNARY_GPU(Ceil)
UNARY_GPU(Conjugate)
UNARY_GPU(Cos)
UNARY_GPU(Cosh)
UNARY_GPU(Erf)
UNARY_GPU(ErfInv)
UNARY_GPU(Exp)
UNARY_GPU(Expm1)
UNARY_GPU(Floor)
UNARY_GPU(Imag)
UNARY_GPU(Log1p)
UNARY_GPU(LogicalNot)
UNARY_GPU(Negative)
UNARY_GPU(Real)
UNARY_GPU(Sigmoid)
UNARY_GPU(Sign)
UNARY_GPU(Sin)
UNARY_GPU(Sinh)
UNARY_GPU(Square)
UNARY_GPU(Tan)
UNARY_GPU(Tanh)

void Log::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = out.primitive().stream();
  auto op = get_primitive_string(this);
  switch (base_) {
    case Base::e:
      unary_op_gpu<mxrocm::Log>(inputs, out, op, s);
      break;
    case Base::two:
      unary_op_gpu<mxrocm::Log2>(inputs, out, op, s);
      break;
    case Base::ten:
      unary_op_gpu<mxrocm::Log10>(inputs, out, op, s);
      break;
  }
}

void Round::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  auto& s = out.primitive().stream();
  if (issubdtype(in.dtype(), inexact)) {
    unary_op_gpu<mxrocm::Round>(inputs, out, get_primitive_string(this), s);
  } else {
    // No-op integer types
    out.copy_shared_buffer(in);
  }
}

void Sqrt::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = out.primitive().stream();
  if (recip_) {
    unary_op_gpu<mxrocm::Rsqrt>(inputs, out, "Rsqrt", s);
  } else {
    unary_op_gpu<mxrocm::Sqrt>(inputs, out, "Sqrt", s);
  }
}

} // namespace mlx::core