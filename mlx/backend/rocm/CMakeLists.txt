# Filename rules in ROCm backend:
#
# * Use .hip for device code (instead of .cu), and .cpp/.h if not.
# * Device-only kernel code should be put in kernels/ subdir.
# * Files in kernels/ subdir should not include files outside.
target_sources(
        mlx
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/binary.hip
        ${CMAKE_CURRENT_SOURCE_DIR}/copy.hip
        ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/event.hip.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/fence.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/metal.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/primitives.hip
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce.hip
        ${CMAKE_CURRENT_SOURCE_DIR}/slicing.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/unary.hip
        ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp)

target_compile_definitions(mlx PUBLIC MLX_USE_ROCM)

# Enable defining device lambda functions.
target_compile_options(mlx PRIVATE "$<$<COMPILE_LANGUAGE:HIP>:--extended-lambda>")

# Set architectures for AMD GPUs - adjust based on your target hardware
set(MLX_ROCM_ARCHITECTURES
        "gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100"
        CACHE STRING "ROCm architectures")
message(STATUS "ROCm architectures: ${MLX_ROCM_ARCHITECTURES}")

# Set HIP compiler flags for target architectures
foreach(arch ${MLX_ROCM_ARCHITECTURES})
    list(APPEND HIP_ARCH_FLAGS "--amdgpu-target=${arch}")
endforeach()
set_target_properties(mlx PROPERTIES HIP_ARCHITECTURES "${MLX_ROCM_ARCHITECTURES}")

# Find required packages
find_package(hip REQUIRED)
find_package(rocblas REQUIRED)
find_package(rocprim REQUIRED)
find_package(hipthrust REQUIRED)
find_package(hipcub REQUIRED)

# Include directories
target_include_directories(mlx PRIVATE ${HIP_INCLUDE_DIRS})

# Link against ROCm libraries
target_link_libraries(mlx PRIVATE roc::rocblas hip::hipblas)

# Suppress HIP warnings if needed
target_compile_options(mlx PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-Wno-deprecated-declarations>)