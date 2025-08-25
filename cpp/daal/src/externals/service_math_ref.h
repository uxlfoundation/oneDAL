/* file: service_math_ref.h */
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

#ifndef __SERVICE_MATH_REF_H__
#define __SERVICE_MATH_REF_H__

#include <math.h>
#include <cmath>
#include <limits>
#include "src/services/service_defines.h"
#if defined(TARGET_ARM)
    #if (__CPUID__(DAAL_CPU) == __sve__)
        #include <arm_sve.h>
        #define R_LN2f 1.4426950408889634f       // log2(e)
        #define L2Uf   0.693145751953125f        // high part of ln(2)
        #define L2Lf   1.428606765330187045e-06f //  low part of ln(2); L2Uf+L2Lf ≈ ln(2)
    #endif
#endif

namespace daal
{
namespace internal
{
namespace ref
{
template <typename fpType, CpuType cpu>
struct RefMath
{};

/*
// Double precision functions definition
*/

template <CpuType cpu>
struct RefMath<double, cpu>
{
    typedef size_t SizeType;

    static double sFabs(double in) { return std::abs(in); }

    static double sMin(double in1, double in2) { return (in1 > in2) ? in2 : in1; }

    static double sMax(double in1, double in2) { return (in1 < in2) ? in2 : in1; }

    static double sSqrt(double in) { return sqrt(in); }

    static double sPowx(double in, double in1) { return pow(in, in1); }

    static double sCeil(double in) { return ceil(in); }

    // Not implemented
    static double sErfInv(double in) { return std::numeric_limits<double>::quiet_NaN(); }

    static double sErf(double in) { return erf(in); }

    static double sLog(double in) { return log(in); }

    // Not implemented
    static double sCdfNormInv(double in) { return std::numeric_limits<double>::quiet_NaN(); }

    static void vAdd(SizeType n, const double * a, const double * b, double * y)
    {
#pragma omp simd
        for (SizeType i = 0; i < n; ++i) y[i] = a[i] + b[i];
    }

    static void vSub(SizeType n, const double * a, const double * b, double * y)
    {
#pragma omp simd
        for (SizeType i = 0; i < n; ++i) y[i] = a[i] - b[i];
    }

    static void vPowx(SizeType n, const double * in, double in1, double * out)
    {
#pragma omp simd
        for (SizeType i = 0; i < n; ++i) out[i] = pow(in[i], in1);
    }

    static void vCeil(SizeType n, const double * in, double * out)
    {
#pragma omp simd
        for (SizeType i = 0; i < n; ++i) out[i] = ceil(in[i]);
    }

    // Not implemented
    static void vErfInv(SizeType n, const double * in, double * out)
    {
        for (SizeType i = 0; i < n; ++i) out[i] = std::numeric_limits<double>::quiet_NaN();
    }

    static void vErf(SizeType n, const double * in, double * out)
    {
#pragma omp simd
        for (SizeType i = 0; i < n; ++i) out[i] = erf(in[i]);
    }

    static void vExp(SizeType n, const double * in, double * out)
    {
#pragma omp simd
        for (SizeType i = 0; i < n; ++i) out[i] = exp(in[i]);
    }

    static double vExpThreshold()
    {
        return -650.0;
    }

    static void vTanh(SizeType n, const double * in, double * out)
    {
#pragma omp simd
        for (SizeType i = 0; i < n; ++i) out[i] = tanh(in[i]);
    }

    static void vSqrt(SizeType n, const double * in, double * out)
    {
#pragma omp simd
        for (SizeType i = 0; i < n; ++i) out[i] = sqrt(in[i]);
    }

    static void vInvSqrtI(SizeType n, const double * a, const SizeType inca, double * b, const SizeType incb)
    {
#pragma omp simd
        for (SizeType i = 0; i < n; ++i) b[i * incb] = 1.0 / sqrt(a[i * inca]);
    }

    static void vLog(SizeType n, const double * in, double * out)
    {
#pragma omp simd
        for (SizeType i = 0; i < n; ++i) out[i] = log(in[i]);
    }

    static void vLog1p(SizeType n, const double * in, double * out)
    {
#pragma omp simd
        for (SizeType i = 0; i < n; ++i) out[i] = log1p(in[i]);
    }

    // Not implemented
    static void vCdfNormInv(SizeType n, const double * in, double * out)
    {
        for (SizeType i = 0; i < n; ++i) out[i] = std::numeric_limits<double>::quiet_NaN();
    }
};

/*
// Single precision functions definition
*/

//Sleef implementation of exp function for SVE taken from https://github.com/shibatch/sleef/blob/ae6ad6cdfc1cb837685a8f40bb1447afce49ded2/src/libm/sleefsimdsp.c#L1313
#if defined(TARGET_ARM)
    #if (__CPUID__(DAAL_CPU) == __sve__)
static svfloat32_t exp_vectorized(svfloat32_t d)
{
    svbool_t pg = svptrue_b32();

    // q = (int)(d * R_LN2f)
    svfloat32_t qf = svmul_f32_x(pg, d, svdup_f32(R_LN2f));
    svint32_t q    = svcvt_s32_f32_x(pg, svrintn_f32_x(pg, qf));

    // s = d - q * L2Uf - q * L2Lf
    svfloat32_t s = svmad_f32_x(pg, svcvt_f32_s32_x(pg, q), svdup_f32(-L2Uf), d);
    s             = svmad_f32_x(pg, svcvt_f32_s32_x(pg, q), svdup_f32(-L2Lf), s);

    // Polynomial evaluation
    svfloat32_t u = svdup_f32(0.000198527617612853646278381f);
    u             = svmad_f32_x(pg, u, s, svdup_f32(0.00139304355252534151077271f));
    u             = svmad_f32_x(pg, u, s, svdup_f32(0.00833336077630519866943359f));
    u             = svmad_f32_x(pg, u, s, svdup_f32(0.0416664853692054748535156f));
    u             = svmad_f32_x(pg, u, s, svdup_f32(0.166666671633720397949219f));
    u             = svmad_f32_x(pg, u, s, svdup_f32(0.5f));

    u = svadd_f32_x(pg, svdup_f32(1.0f), svmad_f32_x(pg, svmul_f32_x(pg, s, s), u, s));

    // vldexp2(u, q)
    svint32_t q1     = svasr_n_s32_x(pg, q, 1);
    svint32_t q2     = svsub_s32_x(pg, q, q1);
    svint32_t m1     = svlsl_n_s32_x(pg, svadd_s32_x(pg, q1, svdup_s32(127)), 23);
    svint32_t m2     = svlsl_n_s32_x(pg, svadd_s32_x(pg, q2, svdup_s32(127)), 23);
    svfloat32_t pow1 = svreinterpret_f32_s32(m1);
    svfloat32_t pow2 = svreinterpret_f32_s32(m2);
    u                = svmul_f32_x(pg, svmul_f32_x(pg, u, pow1), pow2);

    // Underflow: x < -104 => 0
    svbool_t mask_lo   = svcmplt_f32(pg, d, svdup_f32(-104.0f));
    svint32_t u_as_int = svreinterpret_s32_f32(u);
    u_as_int           = svsel_s32(mask_lo, svdup_s32(0), u_as_int);
    u                  = svreinterpret_f32_s32(u_as_int);

    // Overflow: x > 100 => +inf
    svbool_t mask_hi = svcmplt_f32(pg, svdup_f32(100.0f), d);
    u                = svsel_f32(mask_hi, svdup_f32(__builtin_inff()), u);

    return u;
}
    #endif
#endif

template <CpuType cpu>
struct RefMath<float, cpu>
{
    typedef size_t SizeType;

    static float sFabs(float in) { return std::abs(in); }

    static float sMin(float in1, float in2) { return (in1 > in2) ? in2 : in1; }

    static float sMax(float in1, float in2) { return (in1 < in2) ? in2 : in1; }

    static float sSqrt(float in) { return sqrt(in); }

    static float sPowx(float in, float in1) { return pow(in, in1); }

    static float sCeil(float in) { return ceil(in); }

    // Not implemented
    static float sErfInv(float in) { return std::numeric_limits<float>::quiet_NaN(); }

    static float sErf(float in) { return erf(in); }

    static float sLog(float in) { return log(in); }

    // Not implemented
    static float sCdfNormInv(float in) { return std::numeric_limits<float>::quiet_NaN(); }

    static void vAdd(SizeType n, const float * a, const float * b, float * y)
    {
#pragma omp simd
        for (SizeType i = 0; i < n; ++i) y[i] = a[i] + b[i];
    }

    static void vSub(SizeType n, const float * a, const float * b, float * y)
    {
#pragma omp simd
        for (SizeType i = 0; i < n; ++i) y[i] = a[i] - b[i];
    }

    static void vPowx(SizeType n, const float * in, float in1, float * out)
    {
#pragma omp simd
        for (SizeType i = 0; i < n; ++i) out[i] = pow(in[i], in1);
    }

    static void vCeil(SizeType n, const float * in, float * out)
    {
#pragma omp simd
        for (SizeType i = 0; i < n; ++i) out[i] = ceil(in[i]);
    }

    // Not implemented
    static void vErfInv(SizeType n, const float * in, float * out)
    {
        for (SizeType i = 0; i < n; ++i) out[i] = std::numeric_limits<float>::quiet_NaN();
    }

    static void vErf(SizeType n, const float * in, float * out)
    {
#pragma omp simd
        for (SizeType i = 0; i < n; ++i) out[i] = erf(in[i]);
    }

    static void vExp(SizeType n, const float * in, float * out)
    {
#if defined(TARGET_ARM) && (__CPUID__(DAAL_CPU) == __sve__)
        const SizeType vectorWidth = svcntw();
        SizeType i                 = 0;
        svbool_t pg_full           = svptrue_b32(); // All lanes active for full vectors

        // Process full vectors
        for (; i + vectorWidth <= n; i += vectorWidth)
        {
            svfloat32_t vin  = svld1(pg_full, &in[i]);
            svfloat32_t vout = exp_vectorized(vin);
            svst1(pg_full, &out[i], vout);
        }

        // Handle remaining tail (if any)
        if (i < n)
        {
            svbool_t pg_tail = svwhilelt_b32(i, n);
            svfloat32_t vin  = svld1(pg_tail, &in[i]);
            svfloat32_t vout = exp_vectorized(vin);
            svst1(pg_tail, &out[i], vout);
        }

        return;
#endif

        // Scalar fallback
#pragma omp simd
        for (SizeType i = 0; i < n; ++i)
        {
            out[i] = exp(in[i]);
        }
    }
    static float vExpThreshold()
    {
        return -75.0f;
    }

    static void vTanh(SizeType n, const float * in, float * out)
    {
#pragma omp simd
        for (SizeType i = 0; i < n; ++i) out[i] = tanh(in[i]);
    }

    static void vSqrt(SizeType n, const float * in, float * out)
    {
#pragma omp simd
        for (SizeType i = 0; i < n; ++i) out[i] = sqrt(in[i]);
    }

    static void vInvSqrtI(SizeType n, const float * a, const SizeType inca, float * b, const SizeType incb)
    {
#pragma omp simd
        for (SizeType i = 0; i < n; ++i) b[i * incb] = 1.0f / sqrtf(a[i * inca]);
    }

    static void vLog(SizeType n, const float * in, float * out)
    {
#pragma omp simd
        for (SizeType i = 0; i < n; ++i) out[i] = log(in[i]);
    }

    static void vLog1p(SizeType n, const float * in, float * out)
    {
#pragma omp simd
        for (SizeType i = 0; i < n; ++i) out[i] = log1p(in[i]);
    }

    // Not implemented
    static void vCdfNormInv(SizeType n, const float * in, float * out)
    {
        for (SizeType i = 0; i < n; ++i) out[i] = std::numeric_limits<float>::quiet_NaN();
    }
};

} // namespace ref
} // namespace internal
} // namespace daal

#endif
