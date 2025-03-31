// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/binary.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/dtype_utils.h"
#include "mlx/backend/rocm/kernels/binary_ops.h"
#include "mlx/backend/rocm/kernels/iterators/general_iterator.h"
#include "mlx/backend/rocm/kernels/iterators/repeat_iterator.h"
#include "mlx/primitives.h"

#include <rocthrust/copy.h>
#include <rocthrust/device_ptr.h>
#include <rocthrust/transform.h>

namespace mlx::core {

namespace {

template <typename Op, typename In, typename Out>
constexpr bool is_supported_binary_op() {
  if (std::is_same_v<Op, mxrocm::Add> || std::is_same_v<Op, mxrocm::Divide> ||
      std::is_same_v<Op, mxrocm::Maximum> ||
      std::is_same_v<Op, mxrocm::Minimum> ||
      std::is_same_v<Op, mxrocm::Multiply> ||
      std::is_same_v<Op, mxrocm::Subtract> ||
      std::is_same_v<Op, mxrocm::Power> ||
      std::is_same_v<Op, mxrocm::Remainder>) {
    return std::is_same_v<In, Out>;
  }
  if (std::is_same_v<Op, mxrocm::Equal> ||
      std::is_same_v<Op, mxrocm::Greater> ||
      std::is_same_v<Op, mxrocm::GreaterEqual> ||
      std::is_same_v<Op, mxrocm::Less> ||
      std::is_same_v<Op, mxrocm::LessEqual> ||
      std::is_same_v<Op, mxrocm::NotEqual>) {
    return std::is_same_v<Out, bool>;
  }
  if (std::is_same_v<Op, mxrocm::LogicalAnd> ||
      std::is_same_v<Op, mxrocm::LogicalOr>) {
    return std::is_same_v<Out, bool> && std::is_same_v<In, bool>;
  }
  if (std::is_same_v<Op, mxrocm::NaNEqual>) {
    return std::is_same_v<Out, bool> &&
        (is_floating_v<In> || std::is_same_v<In, hipFloatComplex>);
  }
  if (std::is_same_v<Op, mxrocm::LogAddExp> ||
      std::is_same_v<Op, mxrocm::ArcTan2>) {
    return std::is_same_v<In, Out> && is_floating_v<In>;
  }
  if (std::is_same_v<Op, mxrocm::BitwiseAnd> ||
      std::is_same_v<Op, mxrocm::BitwiseOr> ||
      std::is_same_v<Op, mxrocm::BitwiseXor>) {
    return std::is_same_v<In, Out> && std::is_integral_v<In>;
  }
  if (std::is_same_v<Op, mxrocm::LeftShift> ||
      std::is_same_v<Op, mxrocm::RightShift>) {
    return std::is_same_v<In, Out> && std::is_integral_v<In> &&
        !std::is_same_v<In, bool>;
  }
  return false;
}

template <typename Op>
void binary_op_gpu_inplace(
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    std::string_view op,
    const Stream& s) {
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto& out = outputs[0];
  if (out.size() == 0) {
    return;
  }

  auto& encoder = mxrocm::get_command_encoder(s);
  encoder.set_input_array(a, b);
  encoder.set_output_array(out);
  encoder.launch_thrust([&](auto policy) {
    MLX_SWITCH_ALL_TYPES(a.dtype(), CTYPE_IN, [&]() {
      MLX_SWITCH_ALL_TYPES(out.dtype(), CTYPE_OUT, [&]() {
        if constexpr (is_supported_binary_op<Op, CTYPE_IN, CTYPE_OUT>()) {
          using InType = hip_type_t<CTYPE_IN>;
          using OutType = hip_type_t<CTYPE_OUT>;
          auto a_ptr = rocthrust::device_pointer_cast(a.data<InType>());
          auto b_ptr = rocthrust::device_pointer_cast(b.data<InType>());
          auto out_begin = rocthrust::device_pointer_cast(out.data<OutType>());

          auto bopt = get_binary_op_type(a, b);
          if (bopt == BinaryOpType::ScalarScalar) {
            auto a_begin = mxrocm::make_repeat_iterator(a_ptr);
            auto a_end = a_begin + out.data_size();
            auto b_begin = mxrocm::make_repeat_iterator(b_ptr);
            rocthrust::transform(policy, a_begin, a_end, b_begin, out_begin, Op());
          } else if (bopt == BinaryOpType::ScalarVector) {
            auto a_begin = mxrocm::make_repeat_iterator(a_ptr);
            auto a_end = a_begin + out.data_size();
            auto b_begin = b_ptr;
            rocthrust::transform(policy, a_begin, a_end, b_begin, out_begin, Op());
          } else if (bopt == BinaryOpType::VectorScalar) {
            auto a_begin = a_ptr;
            auto a_end = a_begin + out.data_size();
            auto b_begin = mxrocm::make_repeat_iterator(b_ptr);
            rocthrust::transform(policy, a_begin, a_end, b_begin, out_begin, Op());
          } else if (bopt == BinaryOpType::VectorVector) {
            auto a_begin = a_ptr;
            auto a_end = a_begin + out.data_size();
            auto b_begin = b_ptr;
            rocthrust::transform(policy, a_begin, a_end, b_begin, out_begin, Op());
          } else {
            auto [shape, strides] = collapse_contiguous_dims(a, b, out);
            auto [a_begin, a_end] = mxrocm::make_general_iterators<int64_t>(
                a_ptr, out.data_size(), shape, strides[0]);
            auto [b_begin, b_end] = mxrocm::make_general_iterators<int64_t>(
                b_ptr, out.data_size(), shape, strides[1]);
            rocthrust::transform(policy, a_begin, a_end, b_begin, out_begin, Op());
          }
        } else {
          throw std::runtime_error(fmt::format(
              "Can not do binary op {} on inputs of {} with result of {}.",
              op,
              dtype_to_string(a.dtype()),
              dtype_to_string(out.dtype())));
        }
      });
    });
  });
}

template <typename Op>
void binary_op_gpu_inplace(
    const std::vector<array>& inputs,
    array& out,
    std::string_view op,
    const Stream& s) {
  std::vector<array> outputs = {out};
  binary_op_gpu_inplace<Op>(inputs, outputs, op, s);
}

template <typename Op>
void binary_op_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    std::string_view op,
    const Stream& s) {
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, outputs[0], bopt);
  set_binary_op_output_data(a, b, outputs[1], bopt);
  binary_op_gpu_inplace<Op>(inputs, outputs, op, s);
}

template <typename Op>
void binary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    std::string_view op,
    const Stream& s) {
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, out, bopt);
  binary_op_gpu_inplace<Op>(inputs, out, op, s);
}

} // namespace

#define BINARY_GPU(func)                                                     \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) {        \
    auto& s = out.primitive().stream();                                      \
    binary_op_gpu<mxrocm::func>(inputs, out, get_primitive_string(this), s); \
  }

#define BINARY_GPU_MULTI(func)                                         \
  void func::eval_gpu(                                                 \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    auto& s = outputs[0].primitive().stream();                         \
    binary_op_gpu<mxrocm::func>(                                       \
        inputs, outputs, get_primitive_string(this), s);               \
  }

BINARY_GPU(Add)
BINARY_GPU(ArcTan2)
BINARY_GPU(Divide)
BINARY_GPU(Remainder)
BINARY_GPU(Equal)
BINARY_GPU(Greater)
BINARY_GPU(GreaterEqual)
BINARY_GPU(Less)
BINARY_GPU(LessEqual)
BINARY_GPU(LogicalAnd)
BINARY_GPU(LogicalOr)
BINARY_GPU(LogAddExp)
BINARY_GPU(Maximum)
BINARY_GPU(Minimum)
BINARY_GPU(Multiply)
BINARY_GPU(NotEqual)
BINARY_GPU(Power)
BINARY_GPU(Subtract)

void BitwiseBinary::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = out.primitive().stream();
  auto op = get_primitive_string(this);
  switch (op_) {
    case BitwiseBinary::And:
      binary_op_gpu<mxrocm::BitwiseAnd>(inputs, out, op, s);
      break;
    case BitwiseBinary::Or:
      binary_op_gpu<mxrocm::BitwiseOr>(inputs, out, op, s);
      break;
    case BitwiseBinary::Xor:
      binary_op_gpu<mxrocm::BitwiseXor>(inputs, out, op, s);
      break;
    case BitwiseBinary::LeftShift:
      binary_op_gpu<mxrocm::LeftShift>(inputs, out, op, s);
      break;
    case BitwiseBinary::RightShift:
      binary_op_gpu<mxrocm::RightShift>(inputs, out, op, s);
      break;
  }
}

} // namespace mlx::core