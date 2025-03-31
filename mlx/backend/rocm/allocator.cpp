// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/utils.h"

#include <hip/hip_runtime.h>
#include <fmt/format.h>

namespace mlx::core {

namespace mxrocm {

RocmAllocator::RocmAllocator() {
  size_t free, total;
  CHECK_HIP_ERROR(hipMemGetInfo(&free, &total));
  memory_limit_ = total * 0.8;
}

Buffer RocmAllocator::malloc(size_t size) {
  // TODO: Check memory limit.
  auto* buf = new RocmBuffer{nullptr, size};
  hipError_t err = hipMallocManaged(&buf->data, size);
  if (err != hipSuccess && err != hipErrorOutOfMemory) {
    throw std::runtime_error(
        fmt::format("hipMallocManaged failed: {}", hipGetErrorString(err)));
  }
  active_memory_ += size;
  peak_memory_ = std::max(active_memory_, peak_memory_);
  return Buffer{buf};
}

void RocmAllocator::free(Buffer buffer) {
  auto* buf = static_cast<RocmBuffer*>(buffer.ptr());
  if (!buf) {
    return;
  }
  active_memory_ -= buf->size;
  hipFree(buf->data);
  delete buf;
}

size_t RocmAllocator::size(Buffer buffer) const {
  auto* buf = static_cast<RocmBuffer*>(buffer.ptr());
  if (!buf) {
    return 0;
  }
  return static_cast<RocmBuffer*>(buffer.ptr())->size;
}

RocmAllocator& allocator() {
  // By creating the |allocator_| on heap, the destructor of RocmAllocator
  // will not be called on exit and buffers in the cache will be leaked. This
  // can save some time at program exit.
  static RocmAllocator* allocator_ = new RocmAllocator;
  return *allocator_;
}

} // namespace mxrocm

namespace allocator {

Allocator& allocator() {
  return mxrocm::allocator();
}

void* Buffer::raw_ptr() {
  if (!ptr_) {
    return nullptr;
  }
  return static_cast<mxrocm::RocmBuffer*>(ptr_)->data;
}

} // namespace allocator

size_t get_active_memory() {
  return mxrocm::allocator().get_active_memory();
}
size_t get_peak_memory() {
  return mxrocm::allocator().get_peak_memory();
}
void reset_peak_memory() {
  return mxrocm::allocator().reset_peak_memory();
}
size_t set_memory_limit(size_t limit) {
  return mxrocm::allocator().set_memory_limit(limit);
}
size_t get_memory_limit() {
  return mxrocm::allocator().get_memory_limit();
}

// No-ops for common allocator
size_t get_cache_memory() {
  return 0;
}
size_t set_cache_limit(size_t) {
  return 0;
}
size_t set_wired_limit(size_t) {
  return 0;
}
void clear_cache() {}

} // namespace mlx::core