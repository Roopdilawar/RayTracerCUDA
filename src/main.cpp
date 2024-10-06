// src/main.cpp

#include "Vector3.h"
#include "Ray.h"
#include "Camera.h"
#include "Hittable.h"
#include "Sphere.h"
#include "HitRecord.h"

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>

// Custom clamp function for C++14
template <typename T>
__host__ __device__ T clamp_val(T value, T min_val, T max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

extern "C" void launch_raytracer(Vector3* framebuffer, int image_width, int image_height, Camera camera, Sphere* d_spheres, int num_spheres);

int main() {
    // Image dimensions
    const int image_width = 1600;
    const int image_height = 1200;
    const int num_pixels = image_width * image_height;

    // Allocate framebuffer
    Vector3* framebuffer;
    cudaMallocManaged(&framebuffer, num_pixels * sizeof(Vector3));

    // Define camera
    Camera camera;
    camera.origin = Vector3(0.0f, 0.0f, 0.0f); // Camera at origin
    camera.lower_left_corner = Vector3(-2.0f, -1.5f, -1.0f);
    camera.horizontal = Vector3(4.0f, 0.0f, 0.0f);
    camera.vertical = Vector3(0.0f, 3.0f, 0.0f);

    // Define spheres
    int num_spheres = 2;
    Sphere* h_spheres = new Sphere[num_spheres];
    h_spheres[0] = Sphere(Vector3(0.0f, 0.0f, -1.0f), 0.5f);
    h_spheres[1] = Sphere(Vector3(0.0f, -100.5f, -1.0f), 100.0f);

    // Allocate device memory for spheres
    Sphere* d_spheres;
    cudaMalloc(&d_spheres, num_spheres * sizeof(Sphere));
    cudaMemcpy(d_spheres, h_spheres, num_spheres * sizeof(Sphere), cudaMemcpyHostToDevice);

    // Launch ray tracer
    launch_raytracer(framebuffer, image_width, image_height, camera, d_spheres, num_spheres);

    // Write framebuffer to PPM file
    std::ofstream ofs("output.ppm");
    ofs << "P3\n" << image_width << " " << image_height << "\n255\n";
    for (int j = image_height - 1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            int pixel_index = j * image_width + i;
            int ir = static_cast<int>(255.99f * clamp_val(framebuffer[pixel_index].x, 0.0f, 1.0f));
            int ig = static_cast<int>(255.99f * clamp_val(framebuffer[pixel_index].y, 0.0f, 1.0f));
            int ib = static_cast<int>(255.99f * clamp_val(framebuffer[pixel_index].z, 0.0f, 1.0f));
            ofs << ir << " " << ig << " " << ib << "\n";
        }
    }
    ofs.close();

    // Free memory
    cudaFree(framebuffer);
    cudaFree(d_spheres);
    delete[] h_spheres;

    std::cout << "Render complete. Image saved to output.ppm\n";
    return 0;
}
