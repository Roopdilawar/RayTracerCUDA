// include/Sphere.h
#ifndef SPHERE_H
#define SPHERE_H

#ifndef __CUDACC__
#define __host__
#define __device__
#endif


#include "Vector3.h"
#include "Ray.h"
#include "HitRecord.h"

struct Sphere {
    Vector3 center;
    float radius;

    __host__ __device__ Sphere() {}
    __host__ __device__ Sphere(Vector3 cen, float r) : center(cen), radius(r) {}

    __host__ __device__ bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const;
};

#endif // SPHERE_H

