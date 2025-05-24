/*******************************************************************************
* Copyright contributors to the oneDAL project
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
 * Contains SVE optimizations for sumWithSIMD algorithm.
 */

#ifndef __FINITENESS_CHECKER_SVE_IMPL_I__
#define __FINITENESS_CHECKER_SVE_IMPL_I__

#include <arm_sve.h>

/*
// Computes sum of the elements of input array of type `float` with sve instructions.
*/
template <>
float sumWithSIMD<float, sve>(size_t n, const float * dataPtr)
{
    float sum        = 0.0f;
    svfloat32_t sums = svdup_f32(0.0f); // Vector register initialized to zero

    // Pointer to the data
    const float * ptr = dataPtr;
    size_t i          = 0;

    // Single loop that handles both full and remainder elements
    svbool_t pg = svwhilelt_b32(i, n);
    while (svptest_any(svptrue_b32(), pg))
    {                                                   // Check if there's any active lane
        svfloat32_t data = svld1_f32(pg, &ptr[i]);      // Load elements
        sums             = svadd_f32_x(pg, sums, data); // Vector sum
        i += svcntw();                                  // Advance by number of elements processed
        pg = svwhilelt_b32(i, n);                       // Update predicate for next iteration
    }

    // Horizontal sum
    sum = svaddv_f32(svptrue_b32(), sums);

    return sum;
}

/*
// Computes sum of the elements of input array of type `double` with sve instructions.
*/
template <>
double sumWithSIMD<double, sve>(size_t n, const double * dataPtr)
{
    double sum       = 0.0;
    svfloat64_t sums = svdup_f64(0.0); // Vector register initialized to zero

    // Pointer to the data
    const double * ptr = dataPtr;
    size_t i           = 0;

    // Single loop that handles both full and remainder elements
    svbool_t pg = svwhilelt_b64(i, n);
    while (svptest_any(svptrue_b64(), pg))
    {                                                   // Check if there's any active lane
        svfloat64_t data = svld1_f64(pg, &ptr[i]);      // Load elements
        sums             = svadd_f64_x(pg, sums, data); // Vector sum
        i += svcntd();                                  // Advance by number of elements processed
        pg = svwhilelt_b64(i, n);                       // Update predicate for next iteration
    }

    // Horizontal sum
    sum = svaddv_f64(svptrue_b64(), sums);

    return sum;
}

template <>
float computeSum<float, sve>(size_t nDataPtrs, size_t nElementsPerPtr, const float ** dataPtrs)
{
    // computeSumSIMD defined in finiteness_checker_cpu.cpp
    return computeSumSIMD<float, sve>(nDataPtrs, nElementsPerPtr, dataPtrs);
}

template <>
double computeSum<double, sve>(size_t nDataPtrs, size_t nElementsPerPtr, const double ** dataPtrs)
{
    // computeSumSIMD defined in finiteness_checker_cpu.cpp
    return computeSumSIMD<double, sve>(nDataPtrs, nElementsPerPtr, dataPtrs);
}

template <>
double computeSumSOA<sve>(NumericTable & table, bool & sumIsFinite, services::Status & st)
{
    // computeSumSOASIMD defined in finiteness_checker_cpu.cpp
    return computeSumSOASIMD<sve>(table, sumIsFinite, st);
}

//TODO: Implement checkFinitenessInBlocks()

#endif // __FINITENESS_CHECKER_SVE_IMPL_I__
