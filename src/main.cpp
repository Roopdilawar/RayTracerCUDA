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
#include <curand_kernel.h> // Include CURAND
#include <vector>
#include <cstdlib> // For rand and srand
#include <ctime>   // For time
#include <random>  // For better random number generation

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
    // Seed the random number generator
    std::mt19937 rng(static_cast<unsigned int>(time(0)));
    std::uniform_real_distribution<float> dist_x(-2.0f, 2.0f);      // Visible range in x
    std::uniform_real_distribution<float> dist_y(-1.0f, 1.0f);      // Visible range in y
    std::uniform_real_distribution<float> dist_z(-10.0f, -2.0f);    // Visible range in z (in front of the camera)
    std::uniform_real_distribution<float> dist_radius(1.0f, 2.0f);  // Sphere size
    std::uniform_real_distribution<float> dist_color(0.0f, 1.0f);
    std::uniform_real_distribution<float> dist_fuzz(0.0f, 0.5f);
    std::uniform_real_distribution<float> dist_ir(1.3f, 2.5f);
    std::uniform_int_distribution<int> dist_material(0, 2);

    // Image dimensions
    const int image_width = 1600;
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
    camera.origin = Vector3(0.0f, 0.0f, 0.0f);  // Camera at origin
    camera.lower_left_corner = Vector3(-2.0f, -1.5f, -1.0f);
    camera.horizontal = Vector3(4.0f, 0.0f, 0.0f);
    camera.vertical = Vector3(0.0f, 3.0f, 0.0f);

    // Define multiple spheres
    int num_spheres = 4; // Adjust the number as needed for testing
    std::vector<Sphere> h_spheres;
    std::vector<MaterialType> h_material_types;
    std::vector<Vector3> h_albedos;
    std::vector<float> h_fuzzes;
    std::vector<float> h_irs;

    // Pre-allocate materials
    for (int i = 0; i < num_spheres; ++i) {
        // Randomly choose a material type
        int material_type = dist_material(rng); // 0: LAMBERTIAN, 1: METAL, 2: DIELECTRIC
        h_material_types.push_back(static_cast<MaterialType>(material_type));

        if (material_type == 0) {
            // Lambertian with random albedo
            Vector3 albedo = Vector3(dist_color(rng), dist_color(rng), dist_color(rng));
            h_albedos.push_back(albedo);
            h_fuzzes.push_back(0.0f); // Not used
            h_irs.push_back(1.0f);    // Not used
        }
        else if (material_type == 1) {
            // Metal with random albedo and fuzz
            Vector3 albedo = Vector3(0.5f + 0.5f * dist_color(rng), 
                                     0.5f + 0.5f * dist_color(rng), 
                                     0.5f + 0.5f * dist_color(rng));
            float fuzz = dist_fuzz(rng);
            h_albedos.push_back(albedo);
            h_fuzzes.push_back(fuzz);
            h_irs.push_back(1.0f);    // Not used
        }
        else {
            // Dielectric with random index of refraction between 1.3 and 2.5
            float ir = dist_ir(rng);
            h_albedos.push_back(Vector3(1.0f, 1.0f, 1.0f)); // Not used
            h_fuzzes.push_back(0.0f); // Not used
            h_irs.push_back(ir);
        }

        // Place spheres within a visible range from the camera
        Vector3 center = Vector3(
            dist_x(rng),   // x between -2 and 2
            dist_y(rng),   // y between -1 and 1
            dist_z(rng)    // z between -10 and -2 (in front of the camera)
        );

        // Random radius between 0.2 and 1.0
        float radius = dist_radius(rng);

        h_spheres.emplace_back(center, radius, h_material_types[i], h_albedos[i], h_fuzzes[i], h_irs[i]);
    }

    // Print the generated spheres for debugging
    std::cout << "Generated spheres:\n";
    for (int i = 0; i < num_spheres; ++i) {
        std::cout << "Sphere " << i << ": Center(" << h_spheres[i].center.x << ", " 
                  << h_spheres[i].center.y << ", " << h_spheres[i].center.z 
                  << "), Radius: " << h_spheres[i].radius 
                  << ", Material Type: " << h_material_types[i];
        if (h_material_types[i] == LAMBERTIAN || h_material_types[i] == METAL)
            std::cout << ", Albedo: (" << h_albedos[i].x << ", " << h_albedos[i].y << ", " << h_albedos[i].z << ")";
        if (h_material_types[i] == METAL)
            std::cout << ", Fuzz: " << h_fuzzes[i];
        if (h_material_types[i] == DIELECTRIC)
            std::cout << ", IR: " << h_irs[i];
        std::cout << "\n";
    }

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
