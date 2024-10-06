// include/Utils.h
#ifndef UTILS_H
#define UTILS_H

#ifndef __CUDACC__
#define __host__
#define __device__
#endif


#include "Vector3.h"

// Device function to generate a random float in [0,1)
__device__ float random_float() {
    // Simple hash-based pseudo-random number generator
    unsigned int seed = threadIdx.x + blockIdx.x * blockDim.x + threadIdx.y + blockIdx.y * blockDim.y;
    seed ^= (seed << 13);
    seed ^= (seed >> 17);
    seed ^= (seed << 5);
    return static_cast<float>(seed & 0x00FFFFFF) / static_cast<float>(0x01000000);
}

// Device function to clamp a value between min and max
__device__ float clamp(float x, float min_val, float max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

#endif // UTILS_H