// include/Utils.h

#ifndef UTILS_H
#define UTILS_H

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "Vector3.h"
#include "Ray.h"
#include "MaterialType.h"
#include <curand_kernel.h>
#include <cmath>

// Clamp function
__device__ __inline__ float clamp_val(float x, float min_val, float max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

// Generate a random unit vector
__device__ __inline__ Vector3 random_unit_vector(curandState* state) {
    float a = curand_uniform(state) * 2.0f * M_PI;
    float z = curand_uniform(state) * 2.0f - 1.0f;
    float r = sqrtf(1.0f - z * z);
    return Vector3(r * cosf(a), r * sinf(a), z);
}

// Generate a random point inside a unit sphere
__device__ __inline__ Vector3 random_in_unit_sphere(curandState* state) {
    Vector3 p;
    do {
        p = 2.0f * Vector3(curand_uniform(state), curand_uniform(state), curand_uniform(state)) - Vector3(1.0f, 1.0f, 1.0f);
    } while (p.length() >= 1.0f);
    return p;
}

// Reflect function
__device__ __inline__ Vector3 reflect(const Vector3& v, const Vector3& n) {
    return v - 2.0f * Vector3::dot(v, n) * n;
}

// Refract function
__device__ __inline__ Vector3 refract(const Vector3& uv, const Vector3& n, float etai_over_etat) {
    float cos_theta = Vector3::dot(-uv, n);
    Vector3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    Vector3 r_out_parallel = -sqrtf(fabsf(1.0f - r_out_perp.length() * r_out_perp.length())) * n;
    return r_out_perp + r_out_parallel;
}

// Schlick's approximation
__device__ __inline__ float reflectance(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
}

#endif // UTILS_H
