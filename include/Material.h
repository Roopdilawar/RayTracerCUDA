// include/Material.h
#ifndef MATERIAL_H
#define MATERIAL_H

#ifndef __CUDACC__
#define __host__
#define __device__
#endif


#include "Ray.h"
#include "Vector3.h"
#include "HitRecord.h" // Ensure HitRecord.h is included before usage
#include "Utils.h"

#include <ctime> // For clock()


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Material {
    __device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vector3& attenuation, Ray& scattered) const = 0;
};

struct Lambertian : public Material {
    Vector3 albedo;

    __device__ Lambertian(Vector3 a) : albedo(a) {}

    __device__ bool scatter(const Ray& r_in, const HitRecord& rec, Vector3& attenuation, Ray& scattered) const override {
        Vector3 scatter_direction = rec.normal + random_unit_vector();
        // Catch degenerate scatter direction
        if (scatter_direction.length() < 1e-8f) {
            scatter_direction = rec.normal;
        }
        scattered = Ray(rec.point, scatter_direction);
        attenuation = albedo;
        return true;
    }

};

#endif // MATERIAL_H