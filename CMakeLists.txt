# CMakeLists.txt
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(RayTracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architecture (based on RTX 2060's compute capability of 7.5)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/main.cpp
    src/raytracer.cu
    src/Sphere.cu
)

# Add executable
add_executable(RayTracer ${SOURCES})

# Specify CUDA properties
set_target_properties(RayTracer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link CURAND library directly by specifying the library name
target_link_libraries(RayTracer PRIVATE curand)

# Link additional libraries if necessary (e.g., OpenGL, GLFW)
# target_link_libraries(RayTracer PRIVATE glfw OpenGL::GL)
