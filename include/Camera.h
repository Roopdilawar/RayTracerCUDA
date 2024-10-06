// include/Camera.h

#ifndef CAMERA_H
#define CAMERA_H

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

#include "Vector3.h"
#include "Ray.h"

struct Camera {
    Vector3 origin;
    Vector3 lower_left_corner;
    Vector3 horizontal;
    Vector3 vertical;

    __host__ __device__ Camera() {}

    __host__ __device__ Ray get_ray(float s, float t) const {
        return Ray(origin, lower_left_corner + s * horizontal + t * vertical - origin);
    }
};

#endif // CAMERA_H
