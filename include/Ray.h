// include/Ray.h
#ifndef RAY_H
#define RAY_H

#ifndef __CUDACC__
#define __host__
#define __device__
#endif


#include "Vector3.h"

struct Ray {
    Vector3 origin;
    Vector3 direction;

    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const Vector3& o, const Vector3& d) : origin(o), direction(d) {}

    __host__ __device__ Vector3 at(float t) const {
        return origin + t * direction;
    }
};

#endif // RAY_H