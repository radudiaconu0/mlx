// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/utils.h"
#include "mlx/event.h"
#include "mlx/scheduler.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace {

__host__ __device__ void event_wait(
    std::atomic<uint64_t>* ac,
    uint64_t value) {
  uint64_t current;
  while ((current = ac->load()) < value) {
    // In HIP, we use a simple busy-wait approach
    // If needed, this could be optimized using platform-specific atomics
  }
}

__host__ __device__ void event_signal(
    std::atomic<uint64_t>* ac,
    uint64_t value) {
  ac->store(value);
  // No explicit notify mechanism in HIP - use atomic store only
}

__global__ void event_wait_kernel(std::atomic<uint64_t>* ac, uint64_t value) {
  event_wait(ac, value);
}

__global__ void event_signal_kernel(
    std::atomic<uint64_t>* ac,
    uint64_t value) {
  event_signal(ac, value);
}

} // namespace

Event::Event(Stream stream) : stream_(stream) {
  // Allocate atomic on managed memory.
  std::atomic<uint64_t>* ac;
  CHECK_HIP_ERROR(hipMallocManaged(&ac, sizeof(std::atomic<uint64_t>)));
  new (ac) std::atomic<uint64_t>(0);
  // Store it in a shared_ptr.
  auto dtor = [](void* ptr) {
    static_cast<std::atomic<uint64_t>*>(ptr)->~atomic<uint64_t>();
    hipFree(ptr);
  };
  event_ = std::shared_ptr<void>(ac, dtor);
}

void Event::wait() {
  auto* ac = static_cast<std::atomic<uint64_t>*>(event_.get());
  event_wait(ac, value());
}

void Event::signal() {
  auto* ac = static_cast<std::atomic<uint64_t>*>(event_.get());
  event_signal(ac, value());
}

void Event::wait(Stream stream) {
  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [this]() mutable { wait(); });
  } else {
    mxrocm::get_command_encoder(stream).launch_kernel_sequentially(
        [this](hipStream_t s) {
          auto* ac = static_cast<std::atomic<uint64_t>*>(event_.get());
          hipLaunchKernelGGL(
              event_wait_kernel,
              dim3(1), dim3(1), 0, s,
              ac, value());
        });
  }
}

void Event::signal(Stream stream) {
  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [this]() mutable { signal(); });
  } else {
    mxrocm::get_command_encoder(stream).launch_kernel_sequentially(
        [this](hipStream_t s) {
          auto* ac = static_cast<std::atomic<uint64_t>*>(event_.get());
          hipLaunchKernelGGL(
              event_signal_kernel,
              dim3(1), dim3(1), 0, s,
              ac, value());
        });
  }
}

bool Event::is_signaled() const {
  auto* ac = static_cast<std::atomic<uint64_t>*>(event_.get());
  return ac->load() >= value();
}

} // namespace mlx::core