// src/raytracer.cu

#include <curand_kernel.h>
#include "Vector3.h"
#include "Ray.h"
#include "Camera.h"
#include "Sphere.h"
#include "HitRecord.h"
#include "Utils.h"
#include "MaterialType.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>

#define MAX_DEPTH 5000

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

// Scatter function based on material type
__device__ bool scatter_ray(const Ray& r_in, const HitRecord& rec, Vector3& attenuation, Ray& scattered, curandState* state) {
    switch (rec.type) {
        case LAMBERTIAN: {
            Vector3 scatter_direction = rec.normal + random_unit_vector(state);
            // Catch degenerate scatter direction
            if (scatter_direction.length() < 1e-8f) {
                scatter_direction = rec.normal;
            }
            scattered = Ray(rec.point, scatter_direction - rec.point);
            attenuation = rec.albedo;
            return true;
        }
        case METAL: {
            Vector3 reflected = reflect(r_in.direction.normalized(), rec.normal);
            scattered = Ray(rec.point, reflected + rec.fuzz * random_in_unit_sphere(state));
            attenuation = rec.albedo;
            return (Vector3::dot(scattered.direction, rec.normal) > 0.0f);
        }
        case DIELECTRIC: {
            attenuation = Vector3(1.0f, 1.0f, 1.0f);
            float refraction_ratio = (Vector3::dot(r_in.direction, rec.normal) < 0.0f) ? (1.0f / rec.ir) : rec.ir;

            Vector3 unit_direction = r_in.direction.normalized();
            float cos_theta = fmin(Vector3::dot(-unit_direction, rec.normal), 1.0f);
            float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

            bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
            Vector3 direction;

            if (cannot_refract || reflectance(cos_theta, refraction_ratio) > curand_uniform(state))
                direction = reflect(unit_direction, rec.normal);
            else
                direction = refract(unit_direction, rec.normal, refraction_ratio);

            scattered = Ray(rec.point, direction);
            return true;
        }
        default:
            return false;
    }
}

// Iterative ray_color function with CURAND
__device__ Vector3 ray_color_iterative(Ray r, Sphere* spheres, int num_spheres, curandState* states, int pixel_index) {
    Vector3 color(1.0f, 1.0f, 1.0f); // Initialize color
    curandState local_state = states[pixel_index]; // Load CURAND state
    for (int depth = 0; depth < MAX_DEPTH; ++depth) {
        HitRecord rec;
        if (hit_world(r, 0.001f, FLT_MAX, rec, spheres, num_spheres)) {
            Ray scattered;
            Vector3 attenuation;
            if (scatter_ray(r, rec, attenuation, scattered, &local_state)) {
                color *= attenuation;
                r = scattered;
            }
            else {
                // No scatter, terminate the ray
                break;
            }
        }
        else {
            // Background gradient
            Vector3 unit_direction = r.direction.normalized();
            float t = 0.5f * (unit_direction.y + 1.0f);
            Vector3 bg_color = (1.0f - t) * Vector3(1.0f, 1.0f, 1.0f) + t * Vector3(0.5f, 0.7f, 1.0f);
            color *= bg_color;
            break; // Ray has left the scene
        }
    }
    states[pixel_index] = local_state; // Save the updated state
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

__global__ void render_kernel(Vector3* framebuffer, int image_width, int image_height, Camera camera, Sphere* spheres, int num_spheres, curandState* states, int samples_per_pixel) {      
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= image_width || j >= image_height) return;

    int pixel_index = j * image_width + i;
    Vector3 accumulated_color(0.0f, 0.0f, 0.0f);

    curandState local_state = states[pixel_index]; // Load CURAND state

    for (int s = 0; s < samples_per_pixel; ++s) {
        // Generate random offsets for anti-aliasing
        float u = (static_cast<float>(i) + curand_uniform(&local_state)) / (image_width - 1);
        float v = (static_cast<float>(j) + curand_uniform(&local_state)) / (image_height - 1);
        Ray r = camera.get_ray(u, v);
        accumulated_color += ray_color_iterative(r, spheres, num_spheres, &local_state, pixel_index);
    }

    states[pixel_index] = local_state; // Save the updated state

    // Average the accumulated color
    framebuffer[pixel_index] = accumulated_color / static_cast<float>(samples_per_pixel);
}

// Bilateral Filter Kernel
__global__ void bilateral_filter_kernel(const Vector3* input, Vector3* output, int width, int height, int kernel_radius, float sigma_spatial, float sigma_intensity) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    Vector3 center = input[idx];
    Vector3 sum(0.0f, 0.0f, 0.0f);
    float weight_sum = 0.0f;
    
    for (int ky = -kernel_radius; ky <= kernel_radius; ++ky) {
        for (int kx = -kernel_radius; kx <= kernel_radius; ++kx) {
            int nx = min(max(x + kx, 0), width - 1);
            int ny = min(max(y + ky, 0), height - 1);
            int n_idx = ny * width + nx;
            Vector3 neighbor = input[n_idx];
            
            float spatial_dist = sqrtf(float(kx * kx + ky * ky));
            float spatial_weight = expf(-(spatial_dist * spatial_dist) / (2.0f * sigma_spatial * sigma_spatial));
            
            Vector3 diff = neighbor - center;
            float intensity_dist = diff.length();
            float intensity_weight = expf(-(intensity_dist * intensity_dist) / (2.0f * sigma_intensity * sigma_intensity));
            
            float weight = spatial_weight * intensity_weight;
            sum += neighbor * weight;
            weight_sum += weight;
        }
    }
    
    output[idx] = sum / weight_sum;
}

// Host function to launch the ray tracing kernel
extern "C" void launch_raytracer(Vector3* framebuffer, int image_width, int image_height, Camera camera, Sphere* d_spheres, int num_spheres, curandState* d_states, int samples_per_pixel) {
    dim3 threads(16, 16);
    dim3 blocks((image_width + threads.x - 1) / threads.x,
                (image_height + threads.y - 1) / threads.y);

    render_kernel<<<blocks, threads>>>(framebuffer, image_width, image_height, camera, d_spheres, num_spheres, d_states, samples_per_pixel);
    cudaError_t err = cudaGetLastError();


    cudaDeviceSynchronize();
}

// Host function to launch the bilateral filter kernel
extern "C" void launch_bilateral_filter(const Vector3* d_input, Vector3* d_output, int image_width, int image_height, int kernel_radius, float sigma_spatial, float sigma_intensity) {
    dim3 threads(16, 16);
    dim3 blocks((image_width + threads.x - 1) / threads.x,
                (image_height + threads.y - 1) / threads.y);
    
    bilateral_filter_kernel<<<blocks, threads>>>(d_input, d_output, image_width, image_height, kernel_radius, sigma_spatial, sigma_intensity);
    cudaError_t err = cudaGetLastError();


    cudaDeviceSynchronize();
}

extern "C" void initialize_curand(curandState* d_states, int width, int height, unsigned long seed) {
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    // Launch kernel to initialize CURAND states
    init_curand_kernel<<<blocks, threads>>>(d_states, width, height, seed);
    cudaError_t err = cudaGetLastError();


    cudaDeviceSynchronize();
}
