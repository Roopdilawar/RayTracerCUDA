// src/raytracer.cu

#include <curand_kernel.h>
#include "Vector3.h"
#include "Ray.h"
#include "Camera.h"
#include "Sphere.h"
#include "HitRecord.h"
#include "Utils.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>

#define MAX_DEPTH 50 // Adjust as needed

// Forward declaration of hit_world
__device__ bool hit_world(const Ray& r, float t_min, float t_max, HitRecord& rec, Sphere* spheres, int num_spheres);

// Initialize CURAND states kernel
__global__ void init_curand_kernel(curandState* states, int image_width, int image_height, unsigned long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= image_width || j >= image_height) return;

    int idx = j * image_width + i;
    // Initialize CURAND state with a unique seed per thread
    curand_init(seed, idx, 0, &states[idx]);
}

// Iterative ray_color function with CURAND
__device__ Vector3 ray_color_iterative(Ray r, Sphere* spheres, int num_spheres, curandState* states, int pixel_index) {
    Vector3 color(1.0f, 1.0f, 1.0f); // Initialize color
    curandState local_state = states[pixel_index]; // Load CURAND state
    for (int depth = 0; depth < MAX_DEPTH; ++depth) {
        HitRecord rec;
        if (hit_world(r, 0.001f, FLT_MAX, rec, spheres, num_spheres)) {
            // Generate a random scatter direction using hemisphere sampling
            Vector3 scatter_direction = rec.normal + random_unit_vector(&local_state);

            // Update the ray
            r = Ray(rec.point, scatter_direction - rec.point);

            // Attenuation factor
            Vector3 attenuation(0.8f, 0.8f, 0.8f);
            color *= attenuation;

            // Update the state
            states[pixel_index] = local_state;
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

__global__ void render_kernel(Vector3* framebuffer, int image_width, int image_height, Camera camera, Sphere* spheres, int num_spheres, curandState* states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= image_width || j >= image_height) return;

    int pixel_index = j * image_width + i;

    float u = static_cast<float>(i) / (image_width - 1);
    float v = static_cast<float>(j) / (image_height - 1);
    Ray r = camera.get_ray(u, v);
    framebuffer[pixel_index] = ray_color_iterative(r, spheres, num_spheres, states, pixel_index);
}

// Host function to launch the ray tracing kernel
extern "C" void launch_raytracer(Vector3* framebuffer, int image_width, int image_height, Camera camera, Sphere* d_spheres, int num_spheres, curandState* d_states) {
    dim3 threads(16, 16);
    dim3 blocks((image_width + threads.x - 1) / threads.x,
                (image_height + threads.y - 1) / threads.y);

    render_kernel<<<blocks, threads>>>(framebuffer, image_width, image_height, camera, d_spheres, num_spheres, d_states);
    cudaError_t err = cudaGetLastError();


    cudaDeviceSynchronize();
}

extern "C" void initialize_curand(curandState* d_states, int width, int height, unsigned long seed) {
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    // Launch kernel to initialize CURAND states
    init_curand_kernel<<<blocks, threads>>>(d_states, width, height, seed);
    cudaDeviceSynchronize();
}