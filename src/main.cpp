// src/main.cpp

#include "Vector3.h"
#include "Ray.h"
#include "Camera.h"
#include "Hittable.h"
#include "Sphere.h"
#include "HitRecord.h"
#include "Utils.h"
#include "MaterialType.h"

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib> // For rand and srand
#include <ctime>   // For time
#include <cmath>   // For powf

// Custom clamp function for C++14
template <typename T>
__host__ __device__ T clamp_val(T value, T min_val, T max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

// Forward declarations of CUDA functions
extern "C" void launch_raytracer(Vector3* framebuffer, int image_width, int image_height, Camera camera, Sphere* d_spheres, int num_spheres, curandState* d_states, int samples_per_pixel);
extern "C" void initialize_curand(curandState* d_states, int image_width, int image_height, unsigned long seed);
extern "C" void launch_bilateral_filter(const Vector3* d_input, Vector3* d_output, int image_width, int image_height, int kernel_radius, float sigma_spatial, float sigma_intensity);

int main() {
    // Seed the random number generator (not used anymore, but kept for potential future use)
    std::srand(static_cast<unsigned int>(time(0)));

    // Image dimensions
    const int image_width = 1600;  // Increased for better quality
    const int image_height = 1200;
    const int num_pixels = image_width * image_height;

    // Samples per pixel
    const int samples_per_pixel = 1000; // Increased from 1 to 100

    // Allocate framebuffer
    Vector3* framebuffer;
    cudaError_t err = cudaMallocManaged(&framebuffer, num_pixels * sizeof(Vector3));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate framebuffer: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    // Allocate denoised framebuffer
    Vector3* denoised_framebuffer;
    err = cudaMallocManaged(&denoised_framebuffer, num_pixels * sizeof(Vector3));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate denoised framebuffer: " << cudaGetErrorString(err) << "\n";
        cudaFree(framebuffer);
        return 1;
    }

    // Define camera
    Camera camera;
    camera.origin = Vector3(0.0f, 0.0f, 2.0f);  // Moved the camera back for better view
    camera.lower_left_corner = Vector3(-2.0f, -1.5f, -1.0f);
    camera.horizontal = Vector3(4.0f, 0.0f, 0.0f);
    camera.vertical = Vector3(0.0f, 3.0f, 0.0f);

    // Define spheres
    int num_spheres = 2; // 1 ground + 1 visible sphere
    std::vector<Sphere> h_spheres;
    std::vector<MaterialType> h_material_types;
    std::vector<Vector3> h_albedos;
    std::vector<float> h_fuzzes;
    std::vector<float> h_irs;

    // Ground Sphere - Silver Metal
    h_material_types.push_back(METAL);
    h_albedos.push_back(Vector3(0.8f, 0.8f, 0.8f)); // Silver color (gray)
    h_fuzzes.push_back(0.1f); // Slight fuzz for a metallic reflection
    h_irs.push_back(1.0f);    // Not used for Metal
    Vector3 ground_center(0.0f, -1000.0f, -5.0f); // Positioned below the camera
    float ground_radius = 1000.0f;
    h_spheres.emplace_back(ground_center, ground_radius, h_material_types.back(), h_albedos.back(), h_fuzzes.back(), h_irs.back());

    // Visible Sphere 1 - Solid Red
    h_material_types.push_back(LAMBERTIAN);
    h_albedos.push_back(Vector3(1.0f, 0.0f, 0.0f)); // Solid red
    h_fuzzes.push_back(0.0f); // Not used for Lambertian
    h_irs.push_back(1.0f);    // Not used for Lambertian
    Vector3 sphere1_center(0.0f, 0.5f, -1.0f);
    float sphere1_radius = 0.5f;
    h_spheres.emplace_back(sphere1_center, sphere1_radius, h_material_types.back(), h_albedos.back(), h_fuzzes.back(), h_irs.back());

    // Allocate device memory for spheres
    Sphere* d_spheres;
    err = cudaMalloc(&d_spheres, num_spheres * sizeof(Sphere));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for spheres: " << cudaGetErrorString(err) << "\n";
        cudaFree(framebuffer);
        cudaFree(denoised_framebuffer);
        return 1;
    }

    // Copy spheres to device
    err = cudaMemcpy(d_spheres, h_spheres.data(), num_spheres * sizeof(Sphere), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy spheres to device: " << cudaGetErrorString(err) << "\n";
        cudaFree(framebuffer);
        cudaFree(denoised_framebuffer);
        cudaFree(d_spheres);
        return 1;
    }

    // Allocate and initialize CURAND states
    curandState* d_states;
    err = cudaMalloc(&d_states, num_pixels * sizeof(curandState));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate CURAND states: " << cudaGetErrorString(err) << "\n";
        cudaFree(framebuffer);
        cudaFree(denoised_framebuffer);
        cudaFree(d_spheres);
        return 1;
    }

    unsigned long seed = 1234UL; // Seed for randomness
    initialize_curand(d_states, image_width, image_height, seed);

    // Launch ray tracer with multiple samples per pixel
    launch_raytracer(framebuffer, image_width, image_height, camera, d_spheres, num_spheres, d_states, samples_per_pixel);

    // Parameters for Bilateral Filter
    int kernel_radius = 2;         // Increased from 3 to 5
    float sigma_spatial = 7.0f;   // Increased from 5.0f to 10.0f
    float sigma_intensity = 0.15f;  // Increased from 0.1f to 0.2f

    // Launch Bilateral Filter
    launch_bilateral_filter(framebuffer, denoised_framebuffer, image_width, image_height, kernel_radius, sigma_spatial, sigma_intensity);

    // Write denoised framebuffer to PPM file with Gamma Correction
    std::ofstream ofs("denoised_output.ppm");
    if (!ofs) {
        std::cerr << "Failed to open denoised_output.ppm for writing.\n";
        cudaFree(framebuffer);
        cudaFree(denoised_framebuffer);
        cudaFree(d_spheres);
        cudaFree(d_states);
        return 1;
    }
    ofs << "P3\n" << image_width << " " << image_height << "\n255\n";
    for (int j = image_height - 1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            int pixel_index = j * image_width + i;
            // Apply gamma correction (gamma = 2.2)
            float r = clamp_val(powf(denoised_framebuffer[pixel_index].x, 1.0f / 2.2f), 0.0f, 1.0f);
            float g = clamp_val(powf(denoised_framebuffer[pixel_index].y, 1.0f / 2.2f), 0.0f, 1.0f);
            float b = clamp_val(powf(denoised_framebuffer[pixel_index].z, 1.0f / 2.2f), 0.0f, 1.0f);
            int ir = static_cast<int>(255.99f * r);
            int ig = static_cast<int>(255.99f * g);
            int ib = static_cast<int>(255.99f * b);
            ofs << ir << " " << ig << " " << ib << "\n";
        }
    }
    ofs.close();

    // Free memory
    cudaFree(framebuffer);
    cudaFree(denoised_framebuffer);
    cudaFree(d_spheres);
    cudaFree(d_states);

    std::cout << "Render complete. Denoised image saved to denoised_output.ppm\n";
    return 0;
}
