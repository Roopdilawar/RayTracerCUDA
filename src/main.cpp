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

// Custom clamp function for C++14
template <typename T>
__host__ __device__ T clamp_val(T value, T min_val, T max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

extern "C" void launch_raytracer(Vector3* framebuffer, int image_width, int image_height, Camera camera, Sphere* d_spheres, int num_spheres, curandState* d_states);
extern "C" void initialize_curand(curandState* d_states, int image_width, int image_height, unsigned long seed);

int main() {
    // Seed the random number generator (not used anymore, but kept for potential future use)
    std::srand(static_cast<unsigned int>(time(0)));

    // Image dimensions
    const int image_width = 1600;  // Reduced for quicker testing
    const int image_height = 1200;
    const int num_pixels = image_width * image_height;

    // Allocate framebuffer
    Vector3* framebuffer;
    cudaError_t err = cudaMallocManaged(&framebuffer, num_pixels * sizeof(Vector3));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate framebuffer: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    // Define camera
    Camera camera;
    camera.origin = Vector3(0.0f, 0.0f, 0.5f);  // Camera at origin
    camera.lower_left_corner = Vector3(-2.0f, -1.5f, -1.0f);
    camera.horizontal = Vector3(4.0f, 0.0f, 0.0f);
    camera.vertical = Vector3(0.0f, 3.0f, 0.0f);

    // Define spheres
    int num_spheres = 2; // 1 ground + 3 visible spheres
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

    // // Visible Sphere 2
    // h_material_types.push_back(DIELECTRIC);
    // h_albedos.push_back(Vector3(1.0f, 1.0f, 1.0f)); // Not used for Dielectric
    // h_fuzzes.push_back(0.0f); // Not used for Dielectric
    // h_irs.push_back(1.5f);    // Glass-like
    // Vector3 sphere2_center(2.0f, 0.0f, -6.0f);
    // float sphere2_radius = 1.0f;
    // h_spheres.emplace_back(sphere2_center, sphere2_radius, h_material_types.back(), h_albedos.back(), h_fuzzes.back(), h_irs.back());

    // // Visible Sphere 3
    // h_material_types.push_back(LAMBERTIAN);
    // h_albedos.push_back(Vector3(0.1f, 0.2f, 0.5f)); // Solid blue
    // h_fuzzes.push_back(0.0f); // Not used for Lambertian
    // h_irs.push_back(1.0f);    // Not used for Lambertian
    // Vector3 sphere3_center(-2.0f, 0.0f, -6.0f);
    // float sphere3_radius = 1.0f;
    // h_spheres.emplace_back(sphere3_center, sphere3_radius, h_material_types.back(), h_albedos.back(), h_fuzzes.back(), h_irs.back());

    // // Print the generated spheres for debugging
    // std::cout << "Generated spheres:\n";
    // for (int i = 0; i < num_spheres; ++i) {
    //     std::cout << "Sphere " << i << ": Center(" << h_spheres[i].center.x << ", " 
    //               << h_spheres[i].center.y << ", " << h_spheres[i].center.z 
    //               << "), Radius: " << h_spheres[i].radius 
    //               << ", Material Type: " << h_material_types[i];
    //     if (h_material_types[i] == LAMBERTIAN || h_material_types[i] == METAL)
    //         std::cout << ", Albedo: (" << h_albedos[i].x << ", " << h_albedos[i].y << ", " << h_albedos[i].z << ")";
    //     if (h_material_types[i] == METAL)
    //         std::cout << ", Fuzz: " << h_fuzzes[i];
    //     if (h_material_types[i] == DIELECTRIC)
    //         std::cout << ", IR: " << h_irs[i];
    //     std::cout << "\n";
    // }

    // Allocate device memory for spheres
    Sphere* d_spheres;
    err = cudaMalloc(&d_spheres, num_spheres * sizeof(Sphere));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for spheres: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    // Copy spheres to device
    err = cudaMemcpy(d_spheres, h_spheres.data(), num_spheres * sizeof(Sphere), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy spheres to device: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    // Allocate and initialize CURAND states
    curandState* d_states;
    err = cudaMalloc(&d_states, num_pixels * sizeof(curandState));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate CURAND states: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    unsigned long seed = 1234UL; // Seed for randomness
    initialize_curand(d_states, image_width, image_height, seed);

    // Launch ray tracer
    launch_raytracer(framebuffer, image_width, image_height, camera, d_spheres, num_spheres, d_states);

    // Write framebuffer to PPM file
    std::ofstream ofs("output.ppm");
    if (!ofs) {
        std::cerr << "Failed to open output.ppm for writing.\n";
        return 1;
    }
    ofs << "P3\n" << image_width << " " << image_height << "\n255\n";
    for (int j = image_height - 1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            int pixel_index = j * image_width + i;
            float r = clamp_val(framebuffer[pixel_index].x, 0.0f, 1.0f);
            float g = clamp_val(framebuffer[pixel_index].y, 0.0f, 1.0f);
            float b = clamp_val(framebuffer[pixel_index].z, 0.0f, 1.0f);
            int ir = static_cast<int>(255.99f * r);
            int ig = static_cast<int>(255.99f * g);
            int ib = static_cast<int>(255.99f * b);
            ofs << ir << " " << ig << " " << ib << "\n";
        }
    }
    ofs.close();

    // Free memory
    cudaFree(framebuffer);
    cudaFree(d_spheres);
    cudaFree(d_states);

    std::cout << "Render complete. Image saved to output.ppm\n";
    return 0;
}
