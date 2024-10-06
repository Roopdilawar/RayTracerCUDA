// include/Hittable.h
#ifndef HITTABLE_H
#define HITTABLE_H

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

#include "Ray.h"
#include "HitRecord.h"

struct Hittable {
    __host__ __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const = 0;
};

#endif // HITTABLE_H
