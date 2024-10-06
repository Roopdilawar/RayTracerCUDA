// include/Vector3.h
#ifndef VECTOR3_H
#define VECTOR3_H

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

#include <cmath>

struct Vector3 {
    float x, y, z;

    __host__ __device__ Vector3() : x(0.0f), y(0.0f), z(0.0f) {}
    __host__ __device__ Vector3(float e0, float e1, float e2) : x(e0), y(e1), z(e2) {}

    // Unary minus operator
    __host__ __device__ Vector3 operator-() const { return Vector3(-x, -y, -z); }

    // Compound assignment operators
    __host__ __device__ Vector3& operator+=(const Vector3& v) {
        x += v.x; y += v.y; z += v.z; return *this;
    }
    __host__ __device__ Vector3& operator-=(const Vector3& v) {
        x -= v.x; y -= v.y; z -= v.z; return *this;
    }
    __host__ __device__ Vector3& operator*=(const float t) {
        x *= t; y *= t; z *= t; return *this;
    }
    __host__ __device__ Vector3& operator/=(const float t) {
        return *this *= 1.0f / t;
    }
    __host__ __device__ Vector3& operator*=(const Vector3& v) {
        x *= v.x; y *= v.y; z *= v.z; return *this;
    }

    // Member operator overloads
    __host__ __device__ Vector3 operator+(const Vector3& v) const {
        return Vector3(x + v.x, y + v.y, z + v.z);
    }
    __host__ __device__ Vector3 operator-(const Vector3& v) const {
        return Vector3(x - v.x, y - v.y, z - v.z);
    }
    __host__ __device__ Vector3 operator*(const Vector3& v) const {
        return Vector3(x * v.x, y * v.y, z * v.z);
    }
    __host__ __device__ Vector3 operator*(float t) const {
        return Vector3(x * t, y * t, z * t);
    }
    __host__ __device__ Vector3 operator/(float t) const {
        return Vector3(x / t, y / t, z / t);
    }

    // Length and normalization
    __host__ __device__ float length() const {
        return sqrtf(x * x + y * y + z * z);
    }
    __host__ __device__ Vector3 normalized() const {
        float len = length();
        return len > 0 ? *this / len : Vector3(0.0f, 0.0f, 0.0f);
    }

    // Dot and cross products
    __host__ __device__ static float dot(const Vector3& a, const Vector3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    __host__ __device__ static Vector3 cross(const Vector3& a, const Vector3& b) {
        return Vector3(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        );
    }

    // Friend function for scalar multiplication from the left
    friend __host__ __device__ Vector3 operator*(float t, const Vector3& v) {
        return v * t;
    }
};

#endif // VECTOR3_H

