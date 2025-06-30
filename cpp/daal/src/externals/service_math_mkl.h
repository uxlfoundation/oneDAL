/* file: service_math_mkl.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Declaration of math service functions
//--
*/

#ifndef __SERVICE_MATH_MKL_H__
#define __SERVICE_MATH_MKL_H__

#include <math.h>
#include <mkl.h>
#include "src/services/service_defines.h"

#define __DAAL_MKLFN_CALL_MATH(f_name, f_args) \
    f_name f_args;                             \
    return;

namespace daal
{
namespace internal
{
namespace mkl
{
template <typename fpType, CpuType cpu>
struct MklMath
{};

/*
// Double precision functions definition
*/

template <CpuType cpu>
struct MklMath<double, cpu>
{
    typedef size_t SizeType;

    static double sFabs(double in) { return (in >= 0.0f) ? in : -in; }

    static double sMin(double in1, double in2) { return (in1 > in2) ? in2 : in1; }

    static double sMax(double in1, double in2) { return (in1 < in2) ? in2 : in1; }

    static double sSqrt(double in) { return sqrt(in); }

    static double sPowx(double in, double in1)
    {
        double r;
        vPowx(1, &in, in1, &r);
        return r;
    }

    static double sCeil(double in)
    {
        double r;
        vCeil(1, &in, &r);
        return r;
    }

    static double sErfInv(double in)
    {
        double r;
        vErfInv(1, &in, &r);
        return r;
    }

    static double sErf(double in)
    {
        double r;
        vErf(1, &in, &r);
        return r;
    }

    static double sLog(double in)
    {
        double r;
        vLog(1, &in, &r);
        return r;
    }

    static double sCdfNormInv(double in)
    {
        double r;
        vCdfNormInv(1, &in, &r);
        return r;
    }

    static void vAdd(SizeType n, const double * a, const double * b, double * y) { __DAAL_MKLFN_CALL_MATH(vdAdd, (n, a, b, y)); }

    static void vSub(SizeType n, const double * a, const double * b, double * y) { __DAAL_MKLFN_CALL_MATH(vdSub, ((int)n, a, b, y)); }

    static void vPowx(SizeType n, const double * in, double in1, double * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmdPowx, ((int)n, in, in1, out, (VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void vCeil(SizeType n, const double * in, double * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmdCeil, ((int)n, in, out, (VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void vErfInv(SizeType n, const double * in, double * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmdErfInv, ((int)n, in, out, (VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void vErf(SizeType n, const double * in, double * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmdErf, ((int)n, in, out, (VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void vExp(SizeType n, const double * in, double * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmdExp, ((int)n, in, out, (VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static double vExpThreshold() { return -650.0; }

    static void vTanh(SizeType n, const double * in, double * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmdTanh, ((int)n, in, out, (VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void vSqrt(SizeType n, const double * in, double * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmdSqrt, ((int)n, in, out, (VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void vInvSqrtI(SizeType n, const double * a, const SizeType inca, double * b, const SizeType incb)
    {
        __DAAL_MKLFN_CALL_MATH(vmdInvSqrtI, (n, a, inca, b, incb, (VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void vLog(SizeType n, const double * in, double * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmdLn, ((int)n, in, out, (VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void vLog1p(SizeType n, const double * in, double * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmdLog1p, ((int)n, in, out, (VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void vCdfNormInv(SizeType n, const double * in, double * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmdCdfNormInv, ((int)n, in, out, (VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }
};

/*
// Single precision functions definition
*/

template <CpuType cpu>
struct MklMath<float, cpu>
{
    typedef size_t SizeType;

    static float sFabs(float in) { return (in >= 0.0) ? in : -in; }

    static float sMin(float in1, float in2) { return (in1 > in2) ? in2 : in1; }

    static float sMax(float in1, float in2) { return (in1 < in2) ? in2 : in1; }

    static float sSqrt(float in) { return sqrt(in); }

    static float sPowx(float in, float in1)
    {
        float r;
        vPowx(1, &in, in1, &r);
        return r;
    }

    static float sCeil(float in)
    {
        float r;
        vCeil(1, &in, &r);
        return r;
    }

    static float sErfInv(float in)
    {
        float r;
        vErfInv(1, &in, &r);
        return r;
    }

    static float sErf(float in)
    {
        float r;
        vErf(1, &in, &r);
        return r;
    }

    static float sLog(float in)
    {
        float r;
        vLog(1, &in, &r);
        return r;
    }

    static float sCdfNormInv(float in)
    {
        float r;
        vCdfNormInv(1, &in, &r);
        return r;
    }

    static void vAdd(SizeType n, const float * a, const float * b, float * y) { __DAAL_MKLFN_CALL_MATH(vsAdd, (n, a, b, y)); }

    static void vSub(SizeType n, const float * a, const float * b, float * y) { __DAAL_MKLFN_CALL_MATH(vsSub, ((int)n, a, b, y)); }

    static void vPowx(SizeType n, const float * in, float in1, float * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmsPowx, ((int)n, in, in1, out, (VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void vCeil(SizeType n, const float * in, float * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmsCeil, ((int)n, in, out, (VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void vErfInv(SizeType n, const float * in, float * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmsErfInv, ((int)n, in, out, (VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void vErf(SizeType n, const float * in, float * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmsErf, ((int)n, in, out, (VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void vExp(SizeType n, const float * in, float * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmsExp, ((int)n, in, out, (VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static float vExpThreshold() { return -75.0f; }

    static void vTanh(SizeType n, const float * in, float * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmsTanh, ((int)n, in, out, (VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void vSqrt(SizeType n, const float * in, float * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmsSqrt, ((int)n, in, out, (VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void vInvSqrtI(SizeType n, const float * a, const SizeType inca, float * b, const SizeType incb)
    {
        __DAAL_MKLFN_CALL_MATH(vmsInvSqrtI, (n, a, inca, b, incb, (VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void vLog(SizeType n, const float * in, float * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmsLn, ((int)n, in, out, (VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void vLog1p(SizeType n, const float * in, float * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmsLog1p, ((int)n, in, out, (VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void vCdfNormInv(SizeType n, const float * in, float * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmsCdfNormInv, ((int)n, in, out, (VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }
};

} // namespace mkl
} // namespace internal
} // namespace daal

#endif
