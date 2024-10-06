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
#include <curand_kernel.h>
#include <cmath>

// Device function to clamp a value between min and max
__device__ float clamp_val(float x, float min_val, float max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

// Device function to generate a random unit vector inside a hemisphere
__device__ Vector3 random_unit_vector(curandState* state) {
    float a = curand_uniform(state) * 2.0f * M_PI;
    float z = curand_uniform(state) * 2.0f - 1.0f;
    float r = sqrtf(1.0f - z * z);
    return Vector3(r * cosf(a), r * sinf(a), z);
}

#endif // UTILS_H
