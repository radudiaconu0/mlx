# Filename rules in ROCm backend:
#
# * Use .cpp for device code files that contain kernels
# * Use .cpp/.h for host-only code
# * Device-only kernel code should be in kernels/ subdir.
# * Files in kernels/ subdir should not include files outside.

# Set HIP as the language for .cpp files
set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/binary.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/copy.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/event.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/unary.cpp
        PROPERTIES LANGUAGE HIP
)

target_sources(
        mlx
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/binary.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/copy.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/event.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/fence.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/metal.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/slicing.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/unary.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp)

target_compile_definitions(mlx PUBLIC MLX_USE_ROCM)

# Enable defining device lambda functions
target_compile_options(mlx PRIVATE "$<$<COMPILE_LANGUAGE:HIP>:--extended-lambda>")

# Suppress some HIP warnings if needed
target_compile_options(mlx PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-Wno-deprecated-declarations>)

# Links against ROCm libraries
target_link_libraries(mlx PRIVATE roc::rocblas hip::hipblas)