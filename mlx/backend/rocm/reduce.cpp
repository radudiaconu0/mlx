// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/reduce.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernels/reduce_ops.h"
#include "mlx/backend/metal/copy.h"
#include "mlx/primitives.h"

#include <assert.h>
#include <rocthrust/copy.h>
#include <rocthrust/device_ptr.h>
#include <rocthrust/iterator/constant_iterator.h>
#include <hipcub/device/device_reduce.hpp>

namespace mlx::core {

namespace {

#define MLX_FORALL_REDUCE_TYPES(_, ...) \
  _(And, __VA_ARGS__)                   \
  _(Or, __VA_ARGS__)                    \
  _(Sum, __VA_ARGS__)                   \
  _(Prod, __VA_ARGS__)                  \
  _(Max, __VA_ARGS__)                   \
  _(Min, __VA_ARGS__)

#define MLX_SWITCH_CASE_REDUCE_TYPE(TYPE, CTYPE, OP, ...) \
  case Reduce::TYPE: {                                    \
    using OP = mxrocm::TYPE<CTYPE>;                       \
    __VA_ARGS__;                                          \
    break;                                                \
  }

#define MLX_SWITCH_REDUCE_TYPES(TYPE, CTYPE, OP, ...)        \
  switch (TYPE) {                                            \
    MLX_FORALL_REDUCE_TYPES(                                 \
        MLX_SWITCH_CASE_REDUCE_TYPE, CTYPE, OP, __VA_ARGS__) \
  }

template <template <typename> class Op, typename T>
constexpr bool is_supported_reduce_op(Op<T>) {
  if (std::is_same_v<Op<T>, mxrocm::And<T>> ||
      std::is_same_v<Op<T>, mxrocm::Or<T>>) {
    return std::is_same_v<T, bool>;
  }
  if (std::is_same_v<Op<T>, mxrocm::Sum<T>> ||
      std::is_same_v<Op<T>, mxrocm::Prod<T>>) {
    return !std::is_same_v<T, bool>;
  }
  if (std::is_same_v<Op<T>, mxrocm::Min<T>> ||
      std::is_same_v<Op<T>, mxrocm::Max<T>>) {
    return true;
  }
  return false;
}

template <typename... Args>
void all_reduce(mxrocm::CommandEncoder& encoder, Args&&... args) {
  // Get required size for temporary storage and allocate it.
  size_t size;
  CHECK_HIP_ERROR(hipcub::DeviceReduce::Reduce(nullptr, size, args...));
  array temp(allocator::malloc(size), {static_cast<int>(size)}, uint8);
  encoder.add_temporary(temp);
  // Actually run reduce.
  CHECK_HIP_ERROR(hipcub::DeviceReduce::Reduce(temp.data<void>(), size, args...));
}

} // namespace

void Reduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  array in = inputs[0];

  // Make sure no identity reductions trickle down here
  assert(!axes_.empty());
  assert(out.size() != in.size());

  out.set_data(allocator::malloc(out.nbytes()));

  auto& s = stream();
  auto& encoder = mxrocm::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);

  // Fill out with init value.
  if (in.size() == 0) {
    encoder.launch_thrust([&](auto policy) {
      MLX_SWITCH_ALL_TYPES(in.dtype(), CTYPE, [&]() {
        using InType = hip_type_t<CTYPE>;
        MLX_SWITCH_REDUCE_TYPES(reduce_type_, InType, OP, {
          if constexpr (is_supported_reduce_op(OP{})) {
            using OutType = std::remove_const_t<decltype(OP::init)>;
            rocthrust::copy_n(
                policy,
                rocthrust::make_constant_iterator(OP::init),
                out.data_size(),
                rocthrust::device_pointer_cast(out.data<OutType>()));
          } else {
            throw std::runtime_error(fmt::format(
                "Can not do reduce init op on dtype {}.",
                dtype_to_string(in.dtype())));
          }
        });
      });
    });
    return;
  }

  // Reduce.
  ReductionPlan plan = get_reduction_plan(in, axes_);

  // If it is a general reduce then copy the input to a contiguous array and
  // recompute the plan.
  if (plan.type == GeneralReduce) {
    array in_copy(in.shape(), in.dtype(), nullptr, {});
    copy_gpu(in, in_copy, CopyType::General, s);
    encoder.add_temporary(in_copy);
    in = in_copy;
    plan = get_reduction_plan(in, axes_);
  }

  encoder.launch_kernel([&](hipStream_t stream) {
    MLX_SWITCH_ALL_TYPES(in.dtype(), CTYPE, [&]() {
      using InType = hip_type_t<CTYPE>;
      MLX_SWITCH_REDUCE_TYPES(reduce_type_, InType, OP, {
        if constexpr (is_supported_reduce_op(OP{})) {
          using OutType = std::remove_const_t<decltype(OP::init)>;
          if (plan.type == ContiguousAllReduce) {
            all_reduce(
                encoder,
                rocthrust::device_pointer_cast(in.data<InType>()),
                rocthrust::device_pointer_cast(out.data<OutType>()),
                in.data_size(),
                OP(),
                OP::init,
                stream);
          } else if (
              plan.type == ContiguousReduce ||
              plan.type == GeneralContiguousReduce) {
            throw std::runtime_error("Reduce not implemented in ROCm backend.");
          } else if (
              plan.type == ContiguousStridedReduce ||
              plan.type == GeneralStridedReduce) {
            throw std::runtime_error("Reduce not implemented in ROCm backend.");
          }
        } else {
          throw std::runtime_error(fmt::format(
              "Can not do reduce op on dtype {}.",
              dtype_to_string(in.dtype())));
        }
      });
    });
  });
}

} // namespace mlx::core