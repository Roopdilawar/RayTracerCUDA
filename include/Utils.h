// include/Utils.h

#ifndef UTILS_H
#define UTILS_H

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

#include "Vector3.h"

// Xorshift RNG
__device__ unsigned int xorshift(unsigned int seed) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return seed;
}

// Device function to generate a random float in [0,1) using Xorshift
__device__ float random_float(unsigned int &seed) {
    seed = xorshift(seed);
    return (seed & 0x00FFFFFF) / 16777216.0f; // 24-bit mantissa
}

// Device function to clamp a value between min and max
__device__ float clamp_val(float x, float min_val, float max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

// Device function to generate a random unit vector inside a sphere with a max iteration limit
__device__ Vector3 random_unit_vector(unsigned int &seed) {
    Vector3 p;
    int max_iterations = 100000; // Prevent infinite loops
    for (int i = 0; i < max_iterations; ++i) {
        // Generate three independent random numbers for x, y, z
        p.x = 2.0f * random_float(seed) - 1.0f;
        p.y = 2.0f * random_float(seed) - 1.0f;
        p.z = 2.0f * random_float(seed) - 1.0f;
        if (p.length() < 1.0f) {
            return p.normalized();
        }
    }
    // Fallback to a default direction if no valid p is found
    return Vector3(1.0f, 0.0f, 0.0f);
}

#endif // UTILS_H
