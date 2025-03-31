// Copyright © 2025 Apple Inc.

#include "mlx/backend/metal/metal.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/metal/metal_impl.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core::metal {

bool is_available() {
  return true;
}

void start_capture(std::string) {}
void stop_capture() {}

void eval(array& arr) {
  auto s = arr.primitive().stream();
  auto& d = mxrocm::device(s.device);

  auto outputs = arr.outputs();
  {
    // If the array is a tracer hold a reference
    // to its inputs so they don't get donated
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }

    arr.primitive().eval_gpu(arr.inputs(), outputs);
  }
  std::unordered_set<std::shared_ptr<array::Data>> buffers;
  for (auto& in : arr.inputs()) {
    buffers.insert(in.data_shared_ptr());
  }
  for (auto& s : arr.siblings()) {
    buffers.insert(s.data_shared_ptr());
  }
  // Remove the output if it was donated to by an input
  if (auto it = buffers.find(arr.data_shared_ptr()); it != buffers.end()) {
    buffers.erase(it);
  }

  scheduler::notify_new_task(s);
  d.get_stream(s).add_host_callback([s, buffers = std::move(buffers)] {
    scheduler::notify_task_completion(s);
  });
}

void finalize(Stream) {
  // HIP kernels are launched immediately so there is nothing to do.
}

void synchronize(Stream stream) {
  mxrocm::device(stream.device).get_stream(stream).synchronize();
}

} // namespace mlx::core::metal