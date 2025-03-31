// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/utils.h"
#include "mlx/backend/metal/metal.h"

#include <fmt/format.h>

namespace mlx::core {

namespace mxrocm {

DeviceStream::DeviceStream(Device& device, Stream stream) : device_(device) {
  device_.make_current();
  CHECK_HIP_ERROR(hipStreamCreate(&stream_));
}

DeviceStream::~DeviceStream() {
  CHECK_HIP_ERROR(hipStreamDestroy(stream_));
}

void DeviceStream::synchronize() {
  // TODO: Wait for all hip streams in mlx stream.
  hipStreamSynchronize(stream_);
}

hipStream_t DeviceStream::schedule_hip_stream() {
  // TODO: Return a stream that maximizes parallelism.
  return stream_;
}

hipStream_t DeviceStream::last_hip_stream() {
  return stream_;
}

void DeviceStream::add_host_callback(std::function<void()> func) {
  CHECK_HIP_ERROR(hipLaunchHostFunc(
      last_hip_stream(),
      [](void* ptr) {
        auto* func = static_cast<std::function<void()>*>(ptr);
        (*func)();
        delete func;
      },
      new std::function<void()>(std::move(func))));
}

CommandEncoder& DeviceStream::get_encoder() {
  if (!encoder_) {
    encoder_ = std::make_unique<CommandEncoder>(*this);
  }
  return *encoder_;
}

Device::Device(int device) : device_(device) {
  // Verify device supports required features (may need adjustment for ROCm)
  int attr = 0;
  hipDeviceGetAttribute(&attr, hipDeviceAttributeConcurrentManagedAccess, device_);
  if (attr != 1) {
    throw std::runtime_error(fmt::format(
        "Device {} does not support synchronization in managed memory.",
        device_));
  }

  // Initialize rocBLAS handle
  make_current();
  CHECK_ROCBLAS_ERROR(rocblas_create_handle(&rocblas_));
}

Device::~Device() {
  rocblas_destroy_handle(rocblas_);
}

void Device::make_current() {
  // We need to set/get current HIP device very frequently, cache it to reduce
  // actual calls of HIP APIs. This function assumes single-thread in host.
  static int current = 0;
  if (current != device_) {
    CHECK_HIP_ERROR(hipSetDevice(device_));
    current = device_;
  }
}

DeviceStream& Device::get_stream(Stream stream) {
  auto it = streams_.find(stream.index);
  if (it == streams_.end()) {
    it = streams_.try_emplace(stream.index, *this, stream).first;
  }
  return it->second;
}

CommandEncoder::CommandEncoder(DeviceStream& stream)
    : device_(stream.device()), stream_(stream) {}

void CommandEncoder::prefetch_memory(const array& arr) {
  // Note: hipMemPrefetchAsync may need to be used with caution as managed memory
  // support may vary between different ROCm versions
  const void* data = arr.data<void>();
  size_t size = arr.data_size() * arr.itemsize();
  if (data && size > 0) {
    // TODO: Use a stream that maximizes parallelism.
    CHECK_HIP_ERROR(hipMemPrefetchAsync(
        data, size, device_.hip_device(), stream_.last_hip_stream()));
  }
}

Device& device(mlx::core::Device device) {
  static std::unordered_map<int, Device> devices;
  auto it = devices.find(device.index);
  if (it == devices.end()) {
    it = devices.try_emplace(device.index, device.index).first;
  }
  return it->second;
}

DeviceStream& get_stream(Stream stream) {
  return device(stream.device).get_stream(stream);
}

CommandEncoder& get_command_encoder(Stream stream) {
  return get_stream(stream).get_encoder();
}

} // namespace mxrocm

namespace metal {

void new_stream(Stream stream) {
  // Ensure the static stream objects get created.
  mxrocm::get_command_encoder(stream);
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info() {
  throw std::runtime_error(
      "[metal::device_info] Not implemented in ROCm backend.");
};

} // namespace metal

} // namespace mlx::core