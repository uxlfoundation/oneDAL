/* file: kernel_function_rbf_helper.h */
/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef __KERNEL_FUNCTION_RBF_HELPER_H__
#define __KERNEL_FUNCTION_RBF_HELPER_H__

#include "src/externals/service_math.h"
#if defined(TARGET_ARM)
    #if (__CPUID__(DAAL_CPU) == __sve__)
        #include <arm_sve.h>
        #define R_LN2f          1.4426950408889634f
        #define L2Uf            0.693145751953125f
        #define L2Lf            1.428606765330187045e-06f
        #define SLEEF_INFINITYf __builtin_inff()

    #endif
#endif
namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace rbf
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
struct KernelRBFTask
{
public:
    DAAL_NEW_DELETE();
    algorithmFPType * mklBuff;
    algorithmFPType * sqrDataA1;
    algorithmFPType * sqrDataA2;

    static KernelRBFTask * create(const size_t blockSize, const bool isEqualMatrix)
    {
        auto object = new KernelRBFTask(blockSize, isEqualMatrix);
        if (object && object->isValid()) return object;
        delete object;
        return nullptr;
    }

    bool isValid() const { return _buff.get(); }

private:
    KernelRBFTask(const size_t blockSize, const bool isEqualMatrix)
    {
        const size_t buffASize = isEqualMatrix ? blockSize : 2 * blockSize;
        _buff.reset(blockSize * blockSize + buffASize);

        mklBuff   = &_buff[0];
        sqrDataA1 = &_buff[blockSize * blockSize];
        sqrDataA2 = isEqualMatrix ? sqrDataA1 : &sqrDataA1[blockSize];
    }

    TArrayScalable<algorithmFPType, cpu> _buff;
};

template <typename algorithmFPType, CpuType cpu>
class HelperKernelRBF
{
public:
    static services::Status postGemmPart(algorithmFPType * const mklBuff, const algorithmFPType * const sqrA1i, const algorithmFPType sqrA2i,
                                         const algorithmFPType coeff, const algorithmFPType expExpThreshold, const size_t n,
                                         algorithmFPType * const dataRBlock);
};

template <typename algorithmFPType, CpuType cpu>
services::Status HelperKernelRBF<algorithmFPType, cpu>::postGemmPart(algorithmFPType * const mklBuff, const algorithmFPType * const sqrA1i,
                                                                     const algorithmFPType sqrA2i, const algorithmFPType coeff,
                                                                     const algorithmFPType expExpThreshold, const size_t n,
                                                                     algorithmFPType * const dataRBlock)
{
    const algorithmFPType negTwo = algorithmFPType(-2.0);
    for (size_t i = 0; i < n; ++i)
    {
        const algorithmFPType rbf = (mklBuff[i] * negTwo + sqrA2i + sqrA1i[i]) * coeff;
        mklBuff[i]                = rbf > expExpThreshold ? rbf : expExpThreshold;
    }
    MathInst<algorithmFPType, cpu>::vExp(n, mklBuff, dataRBlock);
    return services::Status();
}

// SVE implementation for RBF kernel post-GEMM part
#if defined(TARGET_ARM)
    #if (__CPUID__(DAAL_CPU) == __sve__)

//SVE implementation for RBF kernel post-GEMM part double data type
template <>
inline services::Status HelperKernelRBF<double, sve>::postGemmPart(double * const mklBuff, const double * const sqrA1i, const double sqrA2i,
                                                                   const double coeff, const double expExpThreshold, const size_t n,
                                                                   double * const dataRBlock)
{
    const size_t step = svcntd();

    svfloat64_t negTwoVec    = svdup_f64(-2.0);
    svfloat64_t sqrA2iVec    = svdup_f64(sqrA2i);
    svfloat64_t coeffVec     = svdup_f64(coeff);
    svfloat64_t thresholdVec = svdup_f64(expExpThreshold);

    svbool_t pg = svptrue_b64();
    size_t i    = 0;

    // Unrolled loop - 3x
    for (; i + 3 * step <= n; i += 3 * step)
    {
        // Block 1
        svfloat64_t mklVec   = svld1(pg, &mklBuff[i]);
        svfloat64_t sqrA1Vec = svld1(pg, &sqrA1i[i]);
        svfloat64_t tmp      = svmul_f64_x(pg, svadd_f64_x(pg, svmla_f64_x(pg, sqrA1Vec, mklVec, negTwoVec), sqrA2iVec), coeffVec);
        svbool_t mask        = svcmpgt_f64(pg, tmp, thresholdVec);
        tmp                  = svsel_f64(mask, tmp, thresholdVec);
        svst1(pg, &mklBuff[i], tmp);

        // Block 2
        mklVec   = svld1(pg, &mklBuff[i + step]);
        sqrA1Vec = svld1(pg, &sqrA1i[i + step]);
        tmp      = svmul_f64_x(pg, svadd_f64_x(pg, svmla_f64_x(pg, sqrA1Vec, mklVec, negTwoVec), sqrA2iVec), coeffVec);
        mask     = svcmpgt_f64(pg, tmp, thresholdVec);
        tmp      = svsel_f64(mask, tmp, thresholdVec);
        svst1(pg, &mklBuff[i + step], tmp);

        // Block 3
        mklVec   = svld1(pg, &mklBuff[i + 2 * step]);
        sqrA1Vec = svld1(pg, &sqrA1i[i + 2 * step]);
        tmp      = svmul_f64_x(pg, svadd_f64_x(pg, svmla_f64_x(pg, sqrA1Vec, mklVec, negTwoVec), sqrA2iVec), coeffVec);
        mask     = svcmpgt_f64(pg, tmp, thresholdVec);
        tmp      = svsel_f64(mask, tmp, thresholdVec);
        svst1(pg, &mklBuff[i + 2 * step], tmp);
    }

    // Tail loop
    for (; i < n; i += step)
    {
        svbool_t tail_pg     = svwhilelt_b64(i, n);
        svfloat64_t mklVec   = svld1(tail_pg, &mklBuff[i]);
        svfloat64_t sqrA1Vec = svld1(tail_pg, &sqrA1i[i]);

        svfloat64_t tmp = svmul_f64_x(tail_pg, svadd_f64_x(tail_pg, svmla_f64_x(tail_pg, sqrA1Vec, mklVec, negTwoVec), sqrA2iVec), coeffVec);

        svbool_t mask = svcmpgt_f64(tail_pg, tmp, thresholdVec);
        tmp           = svsel_f64(mask, tmp, thresholdVec);
        svst1(tail_pg, &mklBuff[i], tmp);
    }

    //exponential function
    MathInst<double, sve>::vExp(n, mklBuff, dataRBlock);

    return services::Status();
}
//Sleef implementation of float exp function for SVE taken from https://github.com/shibatch/sleef/blob/ae6ad6cdfc1cb837685a8f40bb1447afce49ded2/src/libm/sleefsimdsp.c#L1313
inline svfloat32_t exp_sleef(svfloat32_t d)
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
    u                = svsel_f32(mask_hi, svdup_f32(SLEEF_INFINITYf), u);

    return u;
}

//SVE implementation for RBF kernel post-GEMM part float data type
template <>
inline services::Status HelperKernelRBF<float, sve>::postGemmPart(float * const mklBuff, const float * const sqrA1i, const float sqrA2i,
                                                                  const float coeff, const float expExpThreshold, const size_t n,
                                                                  float * const dataRBlock)
{
    const size_t step        = svcntw();
    svfloat32_t negTwoVec    = svdup_f32(-2.0f);
    svfloat32_t sqrA2iVec    = svdup_f32(sqrA2i);
    svfloat32_t coeffVec     = svdup_f32(coeff);
    svfloat32_t thresholdVec = svdup_f32(expExpThreshold);
    svbool_t pg              = svptrue_b32();
    size_t i                 = 0;
    // Unrolled loop - 3x
    for (; i + 3 * step <= n; i += 3 * step)
    {
        // Block 1
        svfloat32_t mklVec = svld1(pg, &mklBuff[i]);
        svfloat32_t sqrVec = svld1(pg, &sqrA1i[i]);

        svfloat32_t tmp = svmul_f32_x(pg, svadd_f32_x(pg, svmla_f32_x(pg, sqrVec, mklVec, negTwoVec), sqrA2iVec), coeffVec);

        svbool_t mask = svcmpgt_f32(pg, tmp, thresholdVec);
        tmp           = svsel_f32(mask, tmp, thresholdVec);
        svst1(pg, &mklBuff[i], tmp);

        svfloat32_t expVal = exp_sleef(tmp);
        svst1(pg, &dataRBlock[i], expVal);

        // Block 2
        mklVec = svld1(pg, &mklBuff[i + step]);
        sqrVec = svld1(pg, &sqrA1i[i + step]);

        tmp = svmul_f32_x(pg, svadd_f32_x(pg, svmla_f32_x(pg, sqrVec, mklVec, negTwoVec), sqrA2iVec), coeffVec);

        mask = svcmpgt_f32(pg, tmp, thresholdVec);
        tmp  = svsel_f32(mask, tmp, thresholdVec);
        svst1(pg, &mklBuff[i + step], tmp);

        expVal = exp_sleef(tmp);
        svst1(pg, &dataRBlock[i + step], expVal);

        // Block 3
        mklVec = svld1(pg, &mklBuff[i + 2 * step]);
        sqrVec = svld1(pg, &sqrA1i[i + 2 * step]);

        tmp = svmul_f32_x(pg, svadd_f32_x(pg, svmla_f32_x(pg, sqrVec, mklVec, negTwoVec), sqrA2iVec), coeffVec);

        mask = svcmpgt_f32(pg, tmp, thresholdVec);
        tmp  = svsel_f32(mask, tmp, thresholdVec);
        svst1(pg, &mklBuff[i + 2 * step], tmp);

        expVal = exp_sleef(tmp);
        svst1(pg, &dataRBlock[i + 2 * step], expVal);
    }

    // Tail loop
    for (; i < n; i += step)
    {
        svbool_t tail_pg   = svwhilelt_b32(i, n);
        svfloat32_t mklVec = svld1(tail_pg, &mklBuff[i]);
        svfloat32_t sqrVec = svld1(tail_pg, &sqrA1i[i]);

        svfloat32_t tmp = svmul_f32_x(tail_pg, svadd_f32_x(tail_pg, svmla_f32_x(tail_pg, sqrVec, mklVec, negTwoVec), sqrA2iVec), coeffVec);

        svbool_t mask = svcmpgt_f32(tail_pg, tmp, thresholdVec);
        tmp           = svsel_f32(mask, tmp, thresholdVec);
        svst1(tail_pg, &mklBuff[i], tmp);

        svfloat32_t expVal = exp_sleef(tmp);
        svst1(tail_pg, &dataRBlock[i], expVal);
    }

    return services::Status();
}

    #endif
#endif

#if defined(__AVX512F__) && defined(DAAL_INTEL_CPP_COMPILER)

template <>
inline services::Status HelperKernelRBF<double, avx512>::postGemmPart(double * const mklBuff, const double * const sqrA1i, const double sqrA2i,
                                                                      const double coeff, const double expExpThreshold, const size_t n,
                                                                      double * const dataRBlock)
{
    const double negTwo              = -2.0;
    const __m512d sqrA2iVec          = _mm512_set1_pd(sqrA2i);
    const __m512d coeffVec           = _mm512_set1_pd(coeff);
    const __m512d negTwoVec          = _mm512_set1_pd(negTwo);
    const __m512d expExpThresholdVec = _mm512_set1_pd(expExpThreshold);

    size_t i = 0;
    for (; (i + 8) < n; i += 8)
    {
        const __m512d sqrDataA1Vec = _mm512_load_pd(&sqrA1i[i]);
        __m512d sqrDataA1CoeffVec  = _mm512_mul_pd(sqrDataA1Vec, coeffVec);
        __m512d mklBuffVec         = _mm512_load_pd(&mklBuff[i]);
        mklBuffVec                 = _mm512_mul_pd(mklBuffVec, negTwoVec);
        __m512d rbfVec             = _mm512_add_pd(mklBuffVec, sqrA2iVec);
        rbfVec                     = _mm512_fmadd_pd(rbfVec, coeffVec, sqrDataA1CoeffVec);
        rbfVec                     = _mm512_max_pd(rbfVec, expExpThresholdVec);

        _mm512_store_pd(&mklBuff[i], rbfVec);
    }
    for (; i < n; i++)
    {
        const double rbf = (mklBuff[i] * negTwo + sqrA2i + sqrA1i[i]) * coeff;
        mklBuff[i]       = rbf > expExpThreshold ? rbf : expExpThreshold;
    }

    MathInst<double, avx512>::vExp(n, mklBuff, mklBuff);
    i = 0;

    const size_t align = ((64 - (reinterpret_cast<size_t>(dataRBlock) & 63)) & 63) >> 3;
    for (; i < align; i++)
    {
        dataRBlock[i] = mklBuff[i];
    }
    for (; (i + 8) < n; i += 8)
    {
        const __m512d mklBuffVec = _mm512_loadu_pd(&mklBuff[i]);
        _mm512_stream_pd(&dataRBlock[i], mklBuffVec);
    }
    for (; i < n; i++)
    {
        dataRBlock[i] = mklBuff[i];
    }
    return services::Status();
}

template <>
inline services::Status HelperKernelRBF<float, avx512>::postGemmPart(float * const mklBuff, const float * const sqrA1i, const float sqrA2i,
                                                                     const float coeff, const float expExpThreshold, const size_t n,
                                                                     float * const dataRBlock)
{
    const float negTwo              = -2.0f;
    const __m512 sqrA2iVec          = _mm512_set1_ps(sqrA2i);
    const __m512 coeffVec           = _mm512_set1_ps(coeff);
    const __m512 negTwoVec          = _mm512_set1_ps(negTwo);
    const __m512 expExpThresholdVec = _mm512_set1_ps(expExpThreshold);

    size_t i = 0;

    for (; (i + 16) < n; i += 16)
    {
        const __m512 sqrDataA1Vec = _mm512_load_ps(&sqrA1i[i]);
        __m512 sqrDataA1CoeffVec  = _mm512_mul_ps(sqrDataA1Vec, coeffVec);
        __m512 mklBuffVec         = _mm512_load_ps(&mklBuff[i]);
        mklBuffVec                = _mm512_mul_ps(mklBuffVec, negTwoVec);
        __m512 rbfVec             = _mm512_add_ps(mklBuffVec, sqrA2iVec);
        rbfVec                    = _mm512_fmadd_ps(rbfVec, coeffVec, sqrDataA1CoeffVec);
        rbfVec                    = _mm512_max_ps(rbfVec, expExpThresholdVec);
        _mm512_store_ps(&mklBuff[i], rbfVec);
    }
    for (; i < n; i++)
    {
        const float rbf = (mklBuff[i] * negTwo + sqrA2i + sqrA1i[i]) * coeff;
        mklBuff[i]      = rbf > expExpThreshold ? rbf : expExpThreshold;
    }

    MathInst<float, avx512>::vExp(n, mklBuff, mklBuff);
    i = 0;

    const size_t align = ((64 - (reinterpret_cast<size_t>(dataRBlock) & 63)) & 63) >> 2;
    for (; i < align; i++)
    {
        dataRBlock[i] = mklBuff[i];
    }
    for (; (i + 16) < n; i += 16)
    {
        const __m512 mklBuffVec = _mm512_loadu_ps(&mklBuff[i]);
        _mm512_stream_ps(&dataRBlock[i], mklBuffVec);
    }
    for (; i < n; i++)
    {
        dataRBlock[i] = mklBuff[i];
    }
    return services::Status();
}

#endif

} // namespace internal
} // namespace rbf
} // namespace kernel_function
} // namespace algorithms
} // namespace daal

#endif
