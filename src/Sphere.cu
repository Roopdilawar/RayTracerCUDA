// src/Sphere.cu

#include "Sphere.h"

__host__ __device__ bool Sphere::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
    Vector3 oc = r.origin - center;
    float a = Vector3::dot(r.direction, r.direction);
    float half_b = Vector3::dot(oc, r.direction);
    float c = Vector3::dot(oc, oc) - radius * radius;
    float discriminant = half_b * half_b - a * c;

    if (discriminant > 0.0f) {
        float sqrt_d = sqrtf(discriminant);
        float root = (-half_b - sqrt_d) / a;
        if (root < t_max && root > t_min) {
            rec.t = root;
            rec.point = r.at(rec.t);
            rec.normal = (rec.point - center) / radius;
            return true;
        }
        root = (-half_b + sqrt_d) / a;
        if (root < t_max && root > t_min) {
            rec.t = root;
            rec.point = r.at(rec.t);
            rec.normal = (rec.point - center) / radius;
            return true;
        }
    }
    return false;
}
