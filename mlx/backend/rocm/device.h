// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/rocm/utils.h"
#include "mlx/stream.h"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocthrust/execution_policy.h>

#include <unordered_map>

namespace mlx::core::mxrocm {

class Device;
class CommandEncoder;

// A stream in MLX consists of multiple HIP streams.
class DeviceStream {
 public:
  DeviceStream(Device& device, Stream stream);
  ~DeviceStream();

  DeviceStream(const DeviceStream&) = delete;
  DeviceStream& operator=(const DeviceStream&) = delete;

  // Wait until all current tasks finish.
  void synchronize();

  // Return a HIP stream for launching kernels.
  hipStream_t schedule_hip_stream();

  // Return the last stream used.
  hipStream_t last_hip_stream();

  // Run the function in host after last launched work finishes.
  void add_host_callback(std::function<void()> func);

  CommandEncoder& get_encoder();

  Device& device() {
    return device_;
  }

 private:
  Device& device_;
  hipStream_t stream_;
  std::unique_ptr<CommandEncoder> encoder_;
};

class Device {
 public:
  explicit Device(int device);
  ~Device();

  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  // Make this instance the current HIP device, required by some HIP calls.
  void make_current();

  DeviceStream& get_stream(Stream stream);

  int hip_device() const {
    return device_;
  }

  rocblas_handle rocblas_handle() const {
    return rocblas_;
  }

 private:
  int device_;
  ::rocblas_handle rocblas_;
  std::unordered_map<int, DeviceStream> streams_;
};

class CommandEncoder {
 public:
  explicit CommandEncoder(DeviceStream& stream);

  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;

  template <typename... Arrays, typename = enable_for_arrays_t<Arrays...>>
  void set_input_array(const Arrays&... arrays) {
    (prefetch_memory(arrays), ...);
  }

  template <typename... Arrays, typename = enable_for_arrays_t<Arrays...>>
  void set_output_array(const Arrays&... arrays) {
    (prefetch_memory(arrays), ...);
  }

  template <typename... Arrays, typename = enable_for_arrays_t<Arrays...>>
  void add_temporary(Arrays&&... arrays) {
    (temporaries_.push_back(std::forward<Arrays>(arrays)), ...);
  }

  template <typename F>
  void launch_kernel(F&& fun) {
    launch_kernel_with(std::forward<F>(fun), stream_.schedule_hip_stream());
  }

  template <typename F>
  void launch_kernel_sequentially(F&& fun) {
    launch_kernel_with(std::forward<F>(fun), stream_.last_hip_stream());
  }

  template <typename F>
  void launch_thrust(F&& fun) {
    launch_kernel([&](hipStream_t stream) {
      // Make thrust dispatch work on stream asynchronously.
      auto nosync_exec_policy = rocthrust::hip::par(rocthrust::hip::no_sync).on(stream);
      fun(nosync_exec_policy);
    });
  }

  Device& device() {
    return device_;
  }

  DeviceStream& stream() {
    return stream_;
  }

 private:
  template <typename F>
  void launch_kernel_with(F&& fun, hipStream_t stream) {
    device_.make_current();
    fun(stream);
    check_hip_error("kernel launch", hipGetLastError());
    if (!temporaries_.empty()) {
      stream_.add_host_callback([temporaries = std::move(temporaries_)]() {});
    }
  }

  void prefetch_memory(const array& arr);

  Device& device_;
  DeviceStream& stream_;
  std::vector<array> temporaries_;
};

Device& device(mlx::core::Device device);
DeviceStream& get_stream(Stream stream);
CommandEncoder& get_command_encoder(Stream stream);

} // namespace mlx::core::mxrocm