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
#include "MaterialType.h"

struct Sphere {
    Vector3 center;
    float radius;
    MaterialType type;
    Vector3 albedo; // Used for LAMBERTIAN and METAL
    float fuzz;     // Used for METAL
    float ir;       // Used for DIELECTRIC

    __host__ __device__ Sphere() 
        : center(Vector3()), radius(1.0f), type(LAMBERTIAN), albedo(Vector3(1.0f, 1.0f, 1.0f)), fuzz(0.0f), ir(1.0f) {}
    
    __host__ __device__ Sphere(Vector3 cen, float r, MaterialType m, Vector3 a = Vector3(1.0f, 1.0f, 1.0f), float f = 0.0f, float index = 1.0f)
        : center(cen), radius(r), type(m), albedo(a), fuzz(f), ir(index) {}
    
    __host__ __device__ bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const;
};

#endif // SPHERE_H
