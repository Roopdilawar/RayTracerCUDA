// src/raytracer.cu

#include "Vector3.h"
#include "Ray.h"
#include "Camera.h"
#include "Sphere.h"
#include "HitRecord.h"
#include "Utils.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>

#define MAX_DEPTH 500 // You can adjust this as needed

// Forward declaration of hit_world
__device__ bool hit_world(const Ray& r, float t_min, float t_max, HitRecord& rec, Sphere* spheres, int num_spheres);

// Iterative ray_color function with unique seed per pixel
__device__ Vector3 ray_color_iterative(Ray r, Sphere* spheres, int num_spheres, int pixel_index) {
    Vector3 color(1.0f, 1.0f, 1.0f); // Initialize color
    unsigned int seed = pixel_index; // Unique seed per pixel

    for (int depth = 0; depth < MAX_DEPTH; ++depth) {
        HitRecord rec;
        if (hit_world(r, 0.001f, FLT_MAX, rec, spheres, num_spheres)) {
            // Generate a unique seed for each reflection based on depth
            unsigned int reflection_seed = seed * (depth + 1);
            Vector3 target = rec.point + rec.normal + random_unit_vector(reflection_seed);
            Vector3 attenuation(0.8f, 0.8f, 0.8f); // Attenuation factor
            color *= attenuation; // Apply attenuation
            r = Ray(rec.point, target - rec.point); // Update the ray
        } else {
            // Background gradient
            Vector3 unit_direction = r.direction.normalized();
            float t = 0.5f * (unit_direction.y + 1.0f);
            Vector3 bg_color = (1.0f - t) * Vector3(1.0f, 1.0f, 1.0f) + t * Vector3(0.5f, 0.7f, 1.0f);
            color *= bg_color;
            break; // Ray has left the scene
        }
    }
    return color;
}

__device__ bool hit_world(const Ray& r, float t_min, float t_max, HitRecord& rec, Sphere* spheres, int num_spheres) {
    HitRecord temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    for (int i = 0; i < num_spheres; ++i) {
        if (spheres[i].hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

__global__ void render_kernel(Vector3* framebuffer, int image_width, int image_height, Camera camera, Sphere* spheres, int num_spheres) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= image_width || j >= image_height) return;

    int pixel_index = j * image_width + i;

    float u = static_cast<float>(i) / (image_width - 1);
    float v = static_cast<float>(j) / (image_height - 1);
    Ray r = camera.get_ray(u, v);
    framebuffer[pixel_index] = ray_color_iterative(r, spheres, num_spheres, pixel_index);
}

// Host function to launch the kernel
extern "C" void launch_raytracer(Vector3* framebuffer, int image_width, int image_height, Camera camera, Sphere* d_spheres, int num_spheres) {
    dim3 threads(16, 16);
    dim3 blocks((image_width + threads.x - 1) / threads.x,
                (image_height + threads.y - 1) / threads.y);

    render_kernel<<<blocks, threads>>>(framebuffer, image_width, image_height, camera, d_spheres, num_spheres);
    cudaError_t err = cudaGetLastError();
   
    cudaDeviceSynchronize();
}
