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
#pragma omp simd
        for (SizeType i = 0; i < n; ++i) out[i] = exp(in[i]);
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
