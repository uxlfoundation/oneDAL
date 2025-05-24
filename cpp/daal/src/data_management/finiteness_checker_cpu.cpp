/** file: finiteness_checker_cpu.cpp */
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

#include "data_management/data/internal/finiteness_checker.h"
#include "data_management/data/numeric_table.h"
#include "services/env_detect.h"
#include "src/services/service_data_utils.h"
#include "src/externals/service_dispatch.h"
#include "src/threading/threading.h"
#include "service_numeric_table.h"
#include "src/algorithms/service_error_handling.h"
#include "src/data_management/finiteness_checker.h"

#include "finiteness_checker_impl.i"

namespace daal
{
namespace data_management
{
namespace internal
{
using namespace daal::internal;

#if defined(DAAL_INTEL_CPP_COMPILER) || (__CPUID__(DAAL_CPU) == __sve__)

const size_t BLOCK_SIZE       = 8192;
const size_t THREADING_BORDER = 262144;

template <typename DataType>
DataType getInf()
{
    DataType inf;
    if (sizeof(DataType) == 4)
        *((uint32_t *)(&inf)) = floatExpMask;
    else
        *((uint64_t *)(&inf)) = doubleExpMask;
    return inf;
}

// These functions are used for AVX2, AVX512 and SVE
// and are therefore outside of their separate
// implementations

template <typename DataType, daal::CpuType cpu>
DataType sumWithSIMD(size_t n, const DataType * dataPtr);

/*
// Computes multi-threaded sum of a numeric table via summation using SIMD calls
*/
template <typename DataType, daal::CpuType cpu>
DataType computeSumSIMD(size_t nDataPtrs, size_t nElementsPerPtr, const DataType ** dataPtrs)
{
    size_t nBlocksPerPtr = nElementsPerPtr / BLOCK_SIZE;
    if (nBlocksPerPtr == 0) nBlocksPerPtr = 1;
    size_t nElements    = nDataPtrs * nElementsPerPtr;
    bool inParallel     = !(nElements < THREADING_BORDER);
    size_t nPerBlock    = nElementsPerPtr / nBlocksPerPtr;
    size_t nSurplus     = nElementsPerPtr % nBlocksPerPtr;
    size_t nTotalBlocks = nBlocksPerPtr * nDataPtrs;

    daal::services::internal::TArray<DataType, cpu> partialSumsArr(nTotalBlocks);
    DataType * pSums = partialSumsArr.get();
    if (!pSums) return getInf<DataType>();
    for (size_t iBlock = 0; iBlock < nTotalBlocks; ++iBlock) pSums[iBlock] = 0;

    daal::conditional_threader_for(inParallel, nTotalBlocks, [&](size_t iBlock) {
        size_t ptrIdx        = iBlock / nBlocksPerPtr;
        size_t blockIdxInPtr = iBlock - nBlocksPerPtr * ptrIdx;
        size_t start         = blockIdxInPtr * nPerBlock;
        size_t end           = blockIdxInPtr == nBlocksPerPtr - 1 ? start + nPerBlock + nSurplus : start + nPerBlock;

        //sumWithSIMD defined for AVX2 and AVX512 in finiteness_checker_avx2_impl.i and finiteness_checker_avx512_impl.i
        //sumWithSIMD defined for SVE in finiteness_checker_sve_impl.i
        pSums[iBlock] = sumWithSIMD<DataType, cpu>(end - start, dataPtrs[ptrIdx] + start);
    });

    return sumWithSIMD<DataType, cpu>(nTotalBlocks, pSums);
}

/*
// Computes multi-threaded sum of an SOA numeric table via summation using SIMD calls
*/
template <daal::CpuType cpu>
double computeSumSOASIMD(NumericTable & table, bool & sumIsFinite, services::Status & st)
{
    SafeStatus safeStat;
    double sum                                  = 0;
    bool breakFlag                              = false;
    const size_t nRows                          = table.getNumberOfRows();
    const size_t nCols                          = table.getNumberOfColumns();
    NumericTableDictionaryPtr tableFeaturesDict = table.getDictionarySharedPtr();

    daal::TlsMem<double, cpu, services::internal::ScalableCalloc<double, cpu> > tlsSum(1);
    daal::TlsMem<bool, cpu, services::internal::ScalableCalloc<bool, cpu> > tlsNotFinite(1);

    daal::threader_for_break(nCols, nCols, [&](size_t i, bool & needBreak) {
        double * localSum     = tlsSum.local();
        bool * localNotFinite = tlsNotFinite.local();
        DAAL_CHECK_MALLOC_THR(localSum);
        DAAL_CHECK_MALLOC_THR(localNotFinite);

        switch ((*tableFeaturesDict)[i].getIndexType())
        {
        case daal::data_management::features::IndexNumType::DAAL_FLOAT32:
        {
            ReadColumns<float, cpu> colBlock(table, i, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS_THR(colBlock);
            const float * colPtr = colBlock.get();
            // computeSum defined in the finiteness_checker_*impl.i files
            *localSum += static_cast<double>(computeSum<float, cpu>(1, nRows, &colPtr));
            break;
        }
        case daal::data_management::features::IndexNumType::DAAL_FLOAT64:
        {
            ReadColumns<double, cpu> colBlock(table, i, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS_THR(colBlock);
            const double * colPtr = colBlock.get();
            *localSum += computeSum<double, cpu>(1, nRows, &colPtr);
            break;
        }
        default: break;
        }

        *localNotFinite |= valuesAreNotFinite(localSum, 1, false);
        if (*localNotFinite)
        {
            needBreak = true;
            breakFlag = true;
        }
    });

    st |= safeStat.detach();
    if (!st)
    {
        return 0;
    }

    if (breakFlag)
    {
        sum         = getInf<double>();
        sumIsFinite = false;
    }
    else
    {
        tlsSum.reduce([&](double * localSum) { sum += *localSum; });
        tlsNotFinite.reduce([&](bool * localNotFinite) { sumIsFinite &= !*localNotFinite; });
    }

    return sum;
}
    #if defined(TARGET_ARM)
        #if (__CPUID__(DAAL_CPU) == __sve__)

            #include "finiteness_checker_sve_impl.i"

        #endif // __CPUID__(DAAL_CPU) == __sve__
    #endif

#endif

#if defined(DAAL_INTEL_CPP_COMPILER)

template <daal::CpuType cpu>
services::Status checkFinitenessInBlocks(const float ** dataPtrs, bool inParallel, size_t nTotalBlocks, size_t nBlocksPerPtr, size_t nPerBlock,
                                         size_t nSurplus, bool allowNaN, bool & finiteness);

template <daal::CpuType cpu>
services::Status checkFinitenessInBlocks(const double ** dataPtrs, bool inParallel, size_t nTotalBlocks, size_t nBlocksPerPtr, size_t nPerBlock,
                                         size_t nSurplus, bool allowNaN, bool & finiteness);

/*
// Computes finiteness for a numeric table in blocks using SIMD calls
*/
template <typename DataType, daal::CpuType cpu>
bool checkFinitenessSIMD(const size_t nElements, size_t nDataPtrs, size_t nElementsPerPtr, const DataType ** dataPtrs, bool allowNaN)
{
    size_t nBlocksPerPtr = nElementsPerPtr / BLOCK_SIZE;
    if (nBlocksPerPtr == 0) nBlocksPerPtr = 1;
    bool inParallel     = !(nElements < THREADING_BORDER);
    size_t nPerBlock    = nElementsPerPtr / nBlocksPerPtr;
    size_t nSurplus     = nElementsPerPtr % nBlocksPerPtr;
    size_t nTotalBlocks = nBlocksPerPtr * nDataPtrs;

    bool finiteness;
    // defined in finiteness_checker_avx2_impl.i and finiteness_checker_avx512_impl.i for avx2 and avx512 respectively
    checkFinitenessInBlocks<cpu>(dataPtrs, inParallel, nTotalBlocks, nBlocksPerPtr, nPerBlock, nSurplus, allowNaN, finiteness);
    return finiteness;
}

/*
// Computes finiteness for a SOA numeric table by isinf/isnan using SIMD calls
*/
template <daal::CpuType cpu>
bool checkFinitenessSOASIMD(NumericTable & table, bool allowNaN, services::Status & st)
{
    SafeStatus safeStat;
    bool valuesAreFinite                        = true;
    bool breakFlag                              = false;
    const size_t nRows                          = table.getNumberOfRows();
    const size_t nCols                          = table.getNumberOfColumns();
    NumericTableDictionaryPtr tableFeaturesDict = table.getDictionarySharedPtr();

    daal::TlsMem<bool, cpu, services::internal::ScalableCalloc<bool, cpu> > tlsNotFinite(1);

    daal::threader_for_break(nCols, nCols, [&](size_t i, bool & needBreak) {
        bool * localNotFinite = tlsNotFinite.local();
        DAAL_CHECK_MALLOC_THR(localNotFinite);

        switch ((*tableFeaturesDict)[i].getIndexType())
        {
        case daal::data_management::features::IndexNumType::DAAL_FLOAT32:
        {
            ReadColumns<float, cpu> colBlock(table, i, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS_THR(colBlock);
            const float * colPtr = colBlock.get();
            // defined in finiteness_checker_avx2_impl.i and finiteness_checker_avx512_impl.i for avx2 and avx512 respectively
            *localNotFinite |= !checkFiniteness<float, cpu>(nRows, 1, nRows, &colPtr, allowNaN);
            break;
        }
        case daal::data_management::features::IndexNumType::DAAL_FLOAT64:
        {
            ReadColumns<double, cpu> colBlock(table, i, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS_THR(colBlock);
            const double * colPtr = colBlock.get();
            *localNotFinite |= !checkFiniteness<double, cpu>(nRows, 1, nRows, &colPtr, allowNaN);
            break;
        }
        default: break;
        }

        if (*localNotFinite)
        {
            needBreak = true;
            breakFlag = true;
        }
    });

    st |= safeStat.detach();
    if (!st)
    {
        return false;
    }

    if (breakFlag)
    {
        valuesAreFinite = false;
    }
    else
    {
        tlsNotFinite.reduce([&](bool * localNotFinite) { valuesAreFinite &= !*localNotFinite; });
    }

    return valuesAreFinite;
}
    #if (__CPUID__(DAAL_CPU) == __avx512__)

        #include "finiteness_checker_avx512_impl.i"

    #endif // __CPUID__(DAAL_CPU) == __avx512__
    #if (__CPUID__(DAAL_CPU) == __avx2__)

        #include "finiteness_checker_avx2_impl.i"

    #endif // __CPUID__(DAAL_CPU) == __avx2__

#endif

/*
// Check all values of table are finite via summation and if necessary, via isinf/isnan
// if allowNan == True, then NaN is considered a finite value.
*/
template <typename DataType, daal::CpuType cpu>
services::Status allValuesAreFiniteImpl(NumericTable & table, bool allowNaN, bool * finiteness)
{
    services::Status s;
    const size_t nRows    = table.getNumberOfRows();
    const size_t nColumns = table.getNumberOfColumns();
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRows, nColumns);
    const size_t nElements = nRows * nColumns;
    const NTLayout layout  = table.getDataLayout();

    // first stage: compute sum of all values and check its finiteness
    double sum       = 0;
    bool sumIsFinite = true;

    if (layout == NTLayout::soa)
    {
        sum = computeSumSOA<cpu>(table, sumIsFinite, s);
    }
    else
    {
        ReadRows<DataType, cpu> dataBlock(table, 0, nRows);
        DAAL_CHECK_BLOCK_STATUS(dataBlock);
        const DataType * dataPtr = dataBlock.get();

        sum = computeSum<DataType, cpu>(1, nElements, &dataPtr);
    }

    sumIsFinite &= !valuesAreNotFinite(&sum, 1, false);

    if (sumIsFinite)
    {
        *finiteness = true;
        return s;
    }

    // second stage: check finiteness of all values
    bool valuesAreFinite = true;
    if (layout == NTLayout::soa)
    {
        // defined in finiteness_checker_avx2_impl.i and finiteness_checker_avx512_impl.i for avx2 and avx512 respectively
        valuesAreFinite = checkFinitenessSOA<cpu>(table, allowNaN, s);
    }
    else
    {
        ReadRows<DataType, cpu> dataBlock(table, 0, nRows);
        DAAL_CHECK_BLOCK_STATUS(dataBlock);
        const DataType * dataPtr = dataBlock.get();

        valuesAreFinite = checkFiniteness<DataType, cpu>(nElements, 1, nElements, &dataPtr, allowNaN);
    }

    *finiteness = valuesAreFinite;

    return s;
}

template services::Status allValuesAreFiniteImpl<float, DAAL_CPU>(NumericTable & table, bool allowNaN, bool * finiteness);
template services::Status allValuesAreFiniteImpl<double, DAAL_CPU>(NumericTable & table, bool allowNaN, bool * finiteness);

} // namespace internal
} // namespace data_management
} // namespace daal
