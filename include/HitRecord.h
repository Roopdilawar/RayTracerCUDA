// include/HitRecord.h

#ifndef HITRECORD_H
#define HITRECORD_H

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

#include "Vector3.h"
#include "Ray.h"
#include "MaterialType.h"

struct HitRecord {
    float t;
    Vector3 point;
    Vector3 normal;
    MaterialType type;
    Vector3 albedo; // Used for LAMBERTIAN and METAL
    float fuzz;     // Used for METAL
    float ir;       // Used for DIELECTRIC
};

#endif // HITRECORD_H
