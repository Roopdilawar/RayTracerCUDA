# CMakeLists.txt
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(RayTracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architecture (replace 75 with your GPU's compute capability)
# You can find your GPU's compute capability here: https://developer.nvidia.com/cuda-gpus
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

# Link libraries if necessary (e.g., OpenGL, GLFW)
# target_link_libraries(RayTracer PRIVATE glfw OpenGL::GL)