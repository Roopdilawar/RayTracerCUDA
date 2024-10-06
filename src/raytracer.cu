// src/raytracer.cu

// Include statements
#include "Vector3.h"
#include "Ray.h"
#include "Camera.h"
#include "Sphere.h"
#include "HitRecord.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>

#define MAX_DEPTH 5

// Forward declaration of hit_world
__device__ bool hit_world(const Ray& r, float t_min, float t_max, HitRecord& rec, Sphere* spheres, int num_spheres);

__device__ Vector3 ray_color(const Ray& r, Sphere* spheres, int num_spheres) {
    HitRecord rec;
    if (hit_world(r, 0.001f, FLT_MAX, rec, spheres, num_spheres)) {
        return 0.5f * (rec.normal + Vector3(1.0f, 1.0f, 1.0f));
    }
    Vector3 unit_direction = r.direction.normalized();
    float t = 0.5f * (unit_direction.y + 1.0f);
    return (1.0f - t) * Vector3(1.0f, 1.0f, 1.0f) + t * Vector3(0.5f, 0.7f, 1.0f);
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

    float u = float(i) / (image_width - 1);
    float v = float(j) / (image_height - 1);
    Ray r = camera.get_ray(u, v);
    framebuffer[pixel_index] = ray_color(r, spheres, num_spheres);
}

// Host function to launch the kernel
extern "C" void launch_raytracer(Vector3* framebuffer, int image_width, int image_height, Camera camera, Sphere* d_spheres, int num_spheres) {
    dim3 threads(16, 16);
    dim3 blocks((image_width + threads.x - 1) / threads.x,
                (image_height + threads.y - 1) / threads.y);

    render_kernel<<<blocks, threads>>>(framebuffer, image_width, image_height, camera, d_spheres, num_spheres);
    cudaDeviceSynchronize();
}

