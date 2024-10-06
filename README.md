# CUDA Ray Tracer with Bilateral Denoising

<img src="build\output.png" alt="Ray Tracing" width="500"/>


## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Building the Project](#building-the-project)
- [Running the Executable](#running-the-executable)
- [Output](#output)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

Welcome to the **CUDA Ray Tracer with Bilateral Denoising**! This project is a GPU-accelerated ray tracing application developed using CUDA, designed to render high-quality 3D scenes efficiently. To enhance image quality, a bilateral filter denoiser is integrated, reducing noise while preserving essential edges and details.

## Features

- **Real-Time Ray Tracing:** Leverages CUDA for parallel processing, enabling efficient rendering of complex 3D scenes.
- **Multiple Samples Per Pixel (Anti-Aliasing):** Reduces noise and aliasing artifacts by averaging multiple ray samples per pixel.
- **Bilateral Filter Denoising:** Smoothens the final image by applying a CUDA-implemented bilateral filter, maintaining edge integrity.
- **Gamma Correction:** Ensures realistic color representation by adjusting luminance based on gamma values.
- **Flexible Scene Configuration:** Easily add or modify objects and materials within the scene.
- **Cross-Platform Compatibility:** Developed and tested on Windows but can be adapted for other platforms supporting CUDA.

## Technologies Used

- **CUDA C++:** Core language for GPU acceleration and parallel processing.
- **CMake:** Build system generator to streamline the compilation process.
- **CURAND:** CUDA library for random number generation, essential for stochastic sampling in ray tracing.
- **OpenGL & GLFW (Optional):** Placeholder for potential future integrations like real-time display windows.

## Project Structure

```
C:\UALBERTA\ray_tracing_project\RayTracer\
│
├── assets
│   └── textures
├── include
│   ├── Camera.h
│   ├── HitRecord.h
│   ├── Hittable.h
│   ├── Material.h
│   ├── MaterialType.h
│   ├── Ray.h
│   ├── Sphere.h
│   ├── Utils.h
│   └── Vector3.h
├── src
│   ├── main.cpp
│   ├── raytracer.cu
│   ├── Sphere.cu
│   └── shaders
│       ├── fragment_shader.glsl
│       └── vertex_shader.glsl
├── build
│   └── Debug
│       └── RayTracer.exe
├── CMakeLists.txt
└── README.md
```

- **assets/textures:** Directory for storing texture files used in the scene.
- **include:** Header files defining various components like vectors, rays, materials, and camera.
- **src:** Source files containing the implementation of the ray tracer and its components.
- **build:** Generated directory containing compiled binaries.
- **CMakeLists.txt:** Configuration file for CMake to manage the build process.
- **README.md:** Documentation file (this file).

## Installation

### Prerequisites

Before building the project, ensure that the following software is installed on your system:

1. **CUDA Toolkit:** 
   - Download and install from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).
   - Ensure your GPU is CUDA-compatible.

2. **CMake (Version 3.18 or higher):**
   - Download and install from [CMake Official Website](https://cmake.org/download/).

3. **C++ Compiler:**
   - A compatible C++ compiler that works with CUDA (e.g., MSVC on Windows).

4. **NVIDIA Drivers:**
   - Updated drivers supporting the installed CUDA version.

### Optional

- **OpenGL & GLFW:** If you plan to extend the project with real-time rendering features.

## Building the Project

Follow these steps to build the ray tracing application:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Roopdilawar/RayTracerCUDA.git
   ```

2. **Navigate to the Project Directory:**
   ```bash
   cd RayTracer
   ```

3. **Create a Build Directory:**
   ```bash
   mkdir build
   cd build
   ```

4. **Configure the Project with CMake:**
   ```bash
   cmake ..
   ```
   - This command generates the necessary build files based on your system's configuration.

5. **Build the Project:**
   ```bash
   cmake --build . --config Debug
   ```
   - This compiles the source code and generates the executable in the `build/Debug` directory.

## Running the Executable

After successfully building the project, you can run the ray tracer as follows:

1. **Navigate to the Executable Directory:**
   ```bash
   cd build\Debug
   ```

2. **Execute the Ray Tracer:**
   ```bash
   RayTracer.exe
   ```
   - The program will start rendering the scene, apply the bilateral filter denoiser, and generate the output image.

## Output

Upon successful execution, the program produces a denoised image saved in the project root directory:

- **`denoised_output.ppm`**
  - **Format:** PPM (Portable Pixmap)
  - **Description:** Contains the final rendered and denoised image of the scene.

### Viewing the Output

PPM files are simple image files but may not be supported by all image viewers. To view or convert the output:

1. **Using Image Viewer:**
   - Use image viewers that support PPM format, such as **IrfanView** or **GIMP**.

2. **Converting to PNG (Using ImageMagick):**
   - Install [ImageMagick](https://imagemagick.org/index.php).
   - Convert the PPM file to PNG:
     ```bash
     magick convert denoised_output.ppm denoised_output.png
     ```
   - View the PNG file with your preferred image viewer.

## Customization

### Adjusting Rendering Parameters

You can modify various parameters to enhance image quality or optimize performance:

1. **Samples Per Pixel (Anti-Aliasing):**
   - **Location:** `src/main.cpp`
   - **Variable:** `samples_per_pixel`
   - **Description:** Determines the number of ray samples per pixel. Increasing this value reduces noise but increases rendering time.
   - **Default:** `100`
   - **Example:**
     ```cpp
     const int samples_per_pixel = 200; // Increase for higher quality
     ```

2. **Bilateral Filter Parameters:**
   - **Location:** `src/main.cpp`
   - **Variables:**
     - `kernel_radius`: Size of the neighborhood for filtering.
     - `sigma_spatial`: Controls the influence of spatial proximity.
     - `sigma_intensity`: Controls the influence of color similarity.
   - **Default:**
     ```cpp
     int kernel_radius = 5;
     float sigma_spatial = 10.0f;
     float sigma_intensity = 0.2f;
     ```
   - **Example:**
     ```cpp
     int kernel_radius = 7;
     float sigma_spatial = 15.0f;
     float sigma_intensity = 0.3f;
     ```

3. **Maximum Ray Depth:**
   - **Location:** `src/raytracer.cu`
   - **Variable:** `MAX_DEPTH`
   - **Description:** Limits the number of ray bounces. Higher values capture more complex lighting but increase computation.
   - **Default:** `50`
   - **Example:**
     ```cpp
     #define MAX_DEPTH 100 // Increase for more reflections/refractions
     ```

4. **Camera Configuration:**
   - **Location:** `src/main.cpp`
   - **Variables:**
     - `camera.origin`: Position of the camera.
     - `camera.lower_left_corner`, `camera.horizontal`, `camera.vertical`: Define the viewing frustum.
   - **Example:**
     ```cpp
     camera.origin = Vector3(1.0f, 1.0f, 3.0f);
     camera.lower_left_corner = Vector3(-2.5f, -1.5f, -1.0f);
     camera.horizontal = Vector3(5.0f, 0.0f, 0.0f);
     camera.vertical = Vector3(0.0f, 3.0f, 0.0f);
     ```

### Adding New Objects

To add more spheres or other hittable objects:

1. **Define the Object:**
   - Create a new instance of the `Sphere` struct in `src/main.cpp`.
   - Specify its position, radius, material type, and relevant properties.

2. **Update the Scene:**
   - Modify the `h_spheres` vector to include the new object.

3. **Rebuild the Project:**
   - After making changes, rebuild the project to incorporate the new objects.

### Changing Materials

Modify material properties to achieve different visual effects:

1. **Location:** `src/main.cpp`
2. **Variables:**
   - `MaterialType`: Choose between `LAMBERTIAN`, `METAL`, or `DIELECTRIC`.
   - `albedo`: Color of the material.
   - `fuzz`: Roughness factor for metals.
   - `ir`: Index of refraction for dielectrics.
3. **Example:**
   ```cpp
   // Transparent Sphere - Glass
   h_material_types.push_back(DIELECTRIC);
   h_albedos.push_back(Vector3(1.0f, 1.0f, 1.0f)); // White glass
   h_fuzzes.push_back(0.0f); // Not used for Dielectrics
   h_irs.push_back(1.5f);    // Glass refraction index
   Vector3 glass_center(1.0f, 0.5f, -1.0f);
   float glass_radius = 0.5f;
   h_spheres.emplace_back(glass_center, glass_radius, h_material_types.back(), h_albedos.back(), h_fuzzes.back(), h_irs.back());
   ```

## Troubleshooting

- **CUDA Errors:**
  - Ensure that your GPU is CUDA-compatible and that the latest NVIDIA drivers are installed.
  - Verify that the CUDA Toolkit version matches your system's requirements.

- **CMake Configuration Issues:**
  - Make sure CMake is updated to version 3.18 or higher.
  - Check that all required dependencies are correctly installed and accessible.

- **Insufficient Memory:**
  - Rendering high-resolution images with many samples per pixel can consume significant memory. Adjust `image_width`, `image_height`, or `samples_per_pixel` if you encounter memory issues.

- **Performance Bottlenecks:**
  - Optimize CUDA kernels for better memory access patterns.
  - Reduce `MAX_DEPTH` or `samples_per_pixel` to improve rendering times.

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please create a new pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- **NVIDIA CUDA:** For providing the platform that powers high-performance GPU computing.
- **CMake:** For simplifying the build configuration process.
- **OpenGL & GLFW:** For their foundational graphics capabilities (if utilized in future expansions).
- **Curand Library:** For efficient random number generation essential in ray tracing.
- **Community Contributors:** For their valuable feedback and contributions to the project.

