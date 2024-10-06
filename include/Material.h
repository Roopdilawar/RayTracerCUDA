// include/Material.h
#ifndef MATERIAL_H
#define MATERIAL_H

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

#include "Ray.h"
#include "Vector3.h"
#include "HitRecord.h" // Include HitRecord.h for HitRecord struct

#include "Utils.h"

#include <ctime> // For clock()

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Material {
    __device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vector3& attenuation, Ray& scattered, curandState* state) const = 0;
};

struct Lambertian : public Material {
    Vector3 albedo;

    __device__ Lambertian(Vector3 a) : albedo(a) {}

    __device__ bool scatter(const Ray& r_in, const HitRecord& rec, Vector3& attenuation, Ray& scattered, curandState* state) const override {
        Vector3 scatter_direction = rec.normal + random_unit_vector(state);
        // Catch degenerate scatter direction
        if (scatter_direction.length() < 1e-8f) {
            scatter_direction = rec.normal;
        }
        scattered = Ray(rec.point, scatter_direction - rec.point);
        attenuation = albedo;
        return true;
    }
};

struct Metal : public Material {
    Vector3 albedo;
    float fuzz;

    __device__ Metal(Vector3 a, float f) : albedo(a), fuzz(f < 1.0f ? f : 1.0f) {}

    __device__ bool scatter(const Ray& r_in, const HitRecord& rec, Vector3& attenuation, Ray& scattered, curandState* state) const override {
        Vector3 reflected = reflect(r_in.direction.normalized(), rec.normal);
        scattered = Ray(rec.point, reflected + fuzz * random_in_unit_sphere(state));
        attenuation = albedo;
        return (Vector3::dot(scattered.direction, rec.normal) > 0.0f);
    }

    __device__ Vector3 reflect(const Vector3& v, const Vector3& n) const {
        return v - 2.0f * Vector3::dot(v, n) * n;
    }

    __device__ Vector3 random_in_unit_sphere(curandState* state) const {
        Vector3 p;
        do {
            p = 2.0f * Vector3(curand_uniform(state), curand_uniform(state), curand_uniform(state)) - Vector3(1.0f, 1.0f, 1.0f);
        } while (p.length() >= 1.0f);
        return p;
    }
};

struct Dielectric : public Material {
    float ir; // Index of Refraction

    __device__ Dielectric(float index_of_refraction) : ir(index_of_refraction) {}

    __device__ bool scatter(const Ray& r_in, const HitRecord& rec, Vector3& attenuation, Ray& scattered, curandState* state) const override {
        attenuation = Vector3(1.0f, 1.0f, 1.0f);
        float refraction_ratio = rec.normal.y > 0.0f ? (1.0f / ir) : ir;

        Vector3 unit_direction = r_in.direction.normalized();
        float cos_theta = fmin(Vector3::dot(-unit_direction, rec.normal), 1.0f);
        float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
        Vector3 direction;

        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > curand_uniform(state))
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, refraction_ratio);

        scattered = Ray(rec.point, direction);
        return true;
    }

    __device__ Vector3 reflect(const Vector3& v, const Vector3& n) const {
        return v - 2.0f * Vector3::dot(v, n) * n;
    }

    __device__ Vector3 refract(const Vector3& uv, const Vector3& n, float etai_over_etat) const {
        float cos_theta = Vector3::dot(-uv, n);
        Vector3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
        Vector3 r_out_parallel = -sqrtf(fabsf(1.0f - r_out_perp.length() * r_out_perp.length())) * n;
        return r_out_perp + r_out_parallel;
    }

    __device__ float reflectance(float cosine, float ref_idx) const {
        // Use Schlick's approximation for reflectance
        float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
        r0 = r0 * r0;
        return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
    }
};

#endif // MATERIAL_H
