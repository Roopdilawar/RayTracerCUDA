// include/HitRecord.h
#ifndef HITRECORD_H
#define HITRECORD_H

#ifndef __CUDACC__
#define __host__
#define __device__
#endif


#include "Vector3.h"

struct HitRecord {
    float t;
    Vector3 point;
    Vector3 normal;
};

#endif // HITRECORD_H