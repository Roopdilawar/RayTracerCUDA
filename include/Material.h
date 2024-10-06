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

    __device__ Vector3 random_unit_vector() const {
        // Simple approximation for random unit vector
        float a = random_float();
        float b = random_float();
        float z = 2.0f * b - 1.0f;
        float r = sqrtf(1.0f - z*z);
        float phi = 2.0f * M_PI * a;
        float x = r * cosf(phi);
        float y = r * sinf(phi);
        return Vector3(x, y, z);
    }

    __device__ float random_float() const {
        // Simple pseudo-random generator; replace with better RNG if needed
        return fractf(sinf(clock()) * 43758.5453f);
    }

    __device__ float fractf(float x) const {
        return x - floorf(x);
    }
};

#endif // MATERIAL_H