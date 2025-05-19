/** file train_test_split.cpp */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "data_management/data/internal/train_test_split.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/data_dictionary.h"
#include "services/env_detect.h"
#include "src/externals/service_dispatch.h"
#include "src/services/service_data_utils.h"
#include "src/services/service_utils.h"
#include "src/services/service_defines.h"
#include "src/threading/threading.h"
#include "src/data_management/service_numeric_table.h"
#include "data_management/features/defines.h"
#include "src/algorithms/service_error_handling.h"
#include "src/externals/service_rng.h"

namespace daal
{
namespace data_management
{
namespace internal
{
typedef daal::data_management::NumericTable::StorageLayout NTLayout;

const size_t BLOCK_CONST      = 2048;
const size_t THREADING_BORDER = 8388608;

const size_t MT19937_NUMBERS        = 624;
const size_t MT19937_SIZE           = 631;
const size_t MT19937_NUMBERS_OFFSET = 5;

size_t genSwapIdx(size_t i, unsigned int * randomNumbers, size_t & rnIdx)
{
    uint32_t bitMask = i;
    bitMask |= bitMask >> 1;
    bitMask |= bitMask >> 2;
    bitMask |= bitMask >> 4;
    bitMask |= bitMask >> 8;
    bitMask |= bitMask >> 16;

    size_t j = randomNumbers[rnIdx] & bitMask;
    while (j > i)
    {
        ++rnIdx;
        j = randomNumbers[rnIdx] & bitMask;
    }
    ++rnIdx;
    return j;
}

template <daal::CpuType cpu>
services::Status generateRandomNumbers(const int * rngState, unsigned int * randomNumbers, const size_t nSkip, const size_t n)
{
    // initialize baseRNG
    daal::internal::BaseRNGsInst<cpu> baseRng(0, __DAAL_BRNG_MT19937);
    // check that it has correct size
    DAAL_CHECK(baseRng.getStateSize() / 4 == MT19937_SIZE, daal::services::ErrorEngineNotSupported);
    daal::services::internal::TArray<unsigned int, cpu> baseRngStateArr(baseRng.getStateSize() / 4);
    unsigned int * baseRngState = baseRngStateArr.get();
    DAAL_CHECK_MALLOC(baseRngState);
    baseRng.saveState(baseRngState);
    // copy input state numbers to baseRNG
    services::internal::daal_memcpy_s(baseRngState + MT19937_NUMBERS_OFFSET, MT19937_NUMBERS * sizeof(unsigned int), rngState,
                                      MT19937_NUMBERS * sizeof(unsigned int));
    baseRng.loadState(baseRngState);
    daal::internal::RNGsInst<unsigned int, cpu> rng;

    if (nSkip != 0) baseRng.skipAhead(nSkip);

    rng.uniformBits32(n, randomNumbers + nSkip, baseRng.getState());

    return services::Status();
}

template <typename IdxType, daal::CpuType cpu>
services::Status generateShuffledIndicesImpl(const NumericTablePtr & idxTable, const NumericTablePtr & rngStateTable)
{
    daal::SafeStatus s;
    const size_t nThreads  = threader_get_threads_number();
    const size_t n         = idxTable->getNumberOfRows();
    const size_t stateSize = rngStateTable->getNumberOfRows();
    // number of generated uints: 2 x n + 32 for reserve
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n + 16, 2);
    const size_t nRandomUInts = 2 * n + 32;

    daal::internal::WriteColumns<IdxType, cpu> idxBlock(*idxTable, 0, 0, n);
    IdxType * idx = idxBlock.get();
    DAAL_CHECK_MALLOC(idx);

    // check that input RNG state has needed quantity of state numbers
    DAAL_CHECK(stateSize == MT19937_NUMBERS, daal::services::ErrorIncorrectSizeOfInputNumericTable);

    daal::internal::ReadColumns<int, cpu> rngStateBlock(*rngStateTable, 0, 0, stateSize);
    const int * rngState = rngStateBlock.get();
    DAAL_CHECK_MALLOC(rngState);

    daal::services::internal::TArray<unsigned int, cpu> randomUIntsArr(nRandomUInts);
    unsigned int * randomUInts = randomUIntsArr.get();
    DAAL_CHECK_MALLOC(randomUInts);

    const size_t rngBlockSize = 16 * BLOCK_CONST;
    const size_t nRngBlocks   = nRandomUInts / rngBlockSize + !!(nRandomUInts % rngBlockSize);

    if (nThreads > 1 && nRandomUInts > THREADING_BORDER / 16)
    {
        daal::threader_for(nRngBlocks, nRngBlocks, [&](size_t iBlock) {
            const size_t start = iBlock * rngBlockSize;
            const size_t end   = daal::services::internal::min<cpu, size_t>(start + rngBlockSize, nRandomUInts);

            s |= generateRandomNumbers<cpu>(rngState, randomUInts, start, end - start);
        });
    }
    else
    {
        s |= generateRandomNumbers<cpu>(rngState, randomUInts, 0, nRandomUInts);
    }

    for (size_t i = 0; i < n; ++i)
    {
        idx[i] = i;
    }

    size_t iIdx = 0;
    for (size_t i = n - 1; i > 0; --i)
    {
        size_t j = genSwapIdx(i, randomUInts, iIdx);
        daal::services::internal::swap<cpu, IdxType>(idx[i], idx[j]);
    }

    return s.detach();
}

template <typename IdxType>
void generateShuffledIndicesDispImpl(const NumericTablePtr & idxTable, const NumericTablePtr & rngStateTable)
{
#define DAAL_GENERATE_INDICES(cpuId, ...) generateShuffledIndicesImpl<IdxType, cpuId>(__VA_ARGS__);
    DAAL_DISPATCH_FUNCTION_BY_CPU(DAAL_GENERATE_INDICES, idxTable, rngStateTable);
#undef DAAL_GENERATE_INDICES
}

template <typename IdxType>
DAAL_EXPORT void generateShuffledIndices(const NumericTablePtr & idxTable, const NumericTablePtr & rngStateTable)
{
    DAAL_SAFE_CPU_CALL((generateShuffledIndicesDispImpl<IdxType>(idxTable, rngStateTable)),
                       (generateShuffledIndicesImpl<IdxType, DAAL_BASE_CPU>(idxTable, rngStateTable)));
}

template DAAL_EXPORT void generateShuffledIndices<int>(const NumericTablePtr & idxTable, const NumericTablePtr & rngStateTable);

template <typename DataType, typename IdxType, daal::CpuType cpu>
services::Status assignColumnValues(const DataType * origDataPtr, const NumericTablePtr & dataTable, const IdxType * idxPtr, const size_t startRow,
                                    const size_t nRows, const size_t iCol)
{
    daal::internal::WriteColumns<DataType, cpu> dataBlock(*dataTable, iCol, startRow, nRows);
    DataType * dataPtr = dataBlock.get();
    DAAL_CHECK_MALLOC(dataPtr);

    PRAGMA_FORCE_SIMD
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < nRows; ++i)
    {
        dataPtr[i] = origDataPtr[idxPtr[i]];
    }

    return services::Status();
}

template <typename DataType, typename IdxType, daal::CpuType cpu>
services::Status assignColumnSubset(const DataType * origDataPtr, const NumericTablePtr & dataTable, const IdxType * idxPtr, const size_t nRows,
                                    const size_t iCol, const size_t nThreads)
{
    if (nRows > THREADING_BORDER && nThreads > 1)
    {
        daal::SafeStatus s;
        const size_t nBlocks = nRows / BLOCK_CONST + !!(nRows % BLOCK_CONST);

        daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock) {
            const size_t start = iBlock * BLOCK_CONST;
            const size_t end   = daal::services::internal::min<cpu, size_t>(start + BLOCK_CONST, nRows);

            s |= assignColumnValues<DataType, IdxType, cpu>(origDataPtr, dataTable, idxPtr, start, end - start, iCol);
        });
        return s.detach();
    }
    else
    {
        return assignColumnValues<DataType, IdxType, cpu>(origDataPtr, dataTable, idxPtr, 0, nRows, iCol);
    }
}

template <typename DataType, typename IdxType, daal::CpuType cpu>
services::Status splitColumn(const NumericTablePtr & inputTable, const NumericTablePtr & trainTable, const NumericTablePtr & testTable,
                             const IdxType * trainIdx, const IdxType * testIdx, const size_t nTrainRows, const size_t nTestRows, const size_t iCol,
                             const size_t nThreads)
{
    services::Status s;
    daal::internal::ReadColumns<DataType, cpu> origDataBlock(*inputTable, iCol, 0, inputTable->getNumberOfRows());
    const DataType * origDataPtr = origDataBlock.get();
    DAAL_CHECK_MALLOC(origDataPtr);

    s |= assignColumnSubset<DataType, IdxType, cpu>(origDataPtr, trainTable, trainIdx, nTrainRows, iCol, nThreads);
    s |= assignColumnSubset<DataType, IdxType, cpu>(origDataPtr, testTable, testIdx, nTestRows, iCol, nThreads);

    return s;
}

template <typename DataType, typename IdxType, daal::CpuType cpu>
services::Status assignRows(const DataType * origDataPtr, const NumericTablePtr & dataTable, const NumericTablePtr & idxTable, const size_t startRow,
                            const size_t nRows, const size_t nColumns)
{
    daal::internal::WriteRows<DataType, cpu> dataBlock(*dataTable, startRow, nRows);
    daal::internal::ReadColumns<IdxType, cpu> idxBlock(*idxTable, 0, startRow, nRows);
    DataType * dataPtr     = dataBlock.get();
    const IdxType * idxPtr = idxBlock.get();
    DAAL_CHECK_MALLOC(dataPtr);
    DAAL_CHECK_MALLOC(idxPtr);

    for (size_t i = 0; i < nRows; ++i)
    {
        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < nColumns; ++j)
        {
            dataPtr[i * nColumns + j] = origDataPtr[idxPtr[i] * nColumns + j];
        }
    }

    return services::Status();
}

template <typename DataType, typename IdxType, daal::CpuType cpu>
services::Status assignRowsSubset(const DataType * origDataPtr, const NumericTablePtr & dataTable, const NumericTablePtr & idxTable,
                                  const size_t nRows, const size_t nColumns, const size_t nThreads, const size_t blockSize)
{
    if (nRows * nColumns > THREADING_BORDER && nThreads > 1)
    {
        daal::SafeStatus s;
        const size_t nBlocks = nRows / blockSize + !!(nRows % blockSize);

        daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock) {
            const size_t start = iBlock * blockSize;
            const size_t end   = daal::services::internal::min<cpu, size_t>(start + blockSize, nRows);

            s |= assignRows<DataType, IdxType, cpu>(origDataPtr, dataTable, idxTable, start, end - start, nColumns);
        });
        return s.detach();
    }
    else
    {
        return assignRows<DataType, IdxType, cpu>(origDataPtr, dataTable, idxTable, 0, nRows, nColumns);
    }
}

template <typename DataType, typename IdxType, daal::CpuType cpu>
services::Status splitRows(const NumericTablePtr & inputTable, const NumericTablePtr & trainTable, const NumericTablePtr & testTable,
                           const NumericTablePtr & trainIdxTable, const NumericTablePtr & testIdxTable, const size_t nTrainRows,
                           const size_t nTestRows, const size_t nColumns, const size_t nThreads)
{
    services::Status s;
    const size_t blockSize = daal::services::internal::max<cpu, size_t>(BLOCK_CONST / nColumns, 1);
    daal::internal::ReadRows<DataType, cpu> origBlock(*inputTable, 0, inputTable->getNumberOfRows());
    const DataType * origDataPtr = origBlock.get();
    DAAL_CHECK_MALLOC(origDataPtr);

    s |= assignRowsSubset<DataType, IdxType, cpu>(origDataPtr, trainTable, trainIdxTable, nTrainRows, nColumns, nThreads, blockSize);
    s |= assignRowsSubset<DataType, IdxType, cpu>(origDataPtr, testTable, testIdxTable, nTestRows, nColumns, nThreads, blockSize);

    return s;
}

template <typename IdxType, daal::CpuType cpu>
services::Status trainTestSplitImpl(const NumericTablePtr & inputTable, const NumericTablePtr & trainTable, const NumericTablePtr & testTable,
                                    const NumericTablePtr & trainIdxTable, const NumericTablePtr & testIdxTable)
{
    const size_t nThreads   = threader_get_threads_number();
    const NTLayout layout   = inputTable->getDataLayout();
    const size_t nColumns   = inputTable->getNumberOfColumns();
    const size_t nInputRows = inputTable->getNumberOfRows();
    const size_t nTrainRows = trainTable->getNumberOfRows();
    const size_t nTestRows  = testTable->getNumberOfRows();
    DAAL_CHECK(nColumns == testTable->getNumberOfColumns(), ErrorInconsistentNumberOfColumns);
    DAAL_CHECK(nColumns == trainTable->getNumberOfColumns(), ErrorInconsistentNumberOfColumns);
    DAAL_CHECK(nTrainRows == trainIdxTable->getNumberOfRows(), ErrorInconsistentNumberOfRows);
    DAAL_CHECK(nTestRows == testIdxTable->getNumberOfRows(), ErrorInconsistentNumberOfRows);
    DAAL_CHECK(nTrainRows + nTestRows <= nInputRows, ErrorInconsistentNumberOfRows);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nTrainRows + nTestRows, nColumns);

    NumericTableDictionaryPtr tableFeaturesDict = inputTable->getDictionarySharedPtr();

    if (layout == NTLayout::soa)
    {
        daal::SafeStatus s;
        daal::internal::ReadColumns<IdxType, cpu> trainIdxBlock(*trainIdxTable, 0, 0, nTrainRows);
        daal::internal::ReadColumns<IdxType, cpu> testIdxBlock(*testIdxTable, 0, 0, nTestRows);
        const IdxType * trainIdx = trainIdxBlock.get();
        const IdxType * testIdx  = testIdxBlock.get();
        DAAL_CHECK_MALLOC(trainIdx);
        DAAL_CHECK_MALLOC(testIdx);

        daal::conditional_threader_for(
            nColumns > 1 && nColumns * (nTrainRows + nTestRows) > THREADING_BORDER && nThreads > 1, nColumns, [&](size_t iCol) {
                switch ((*tableFeaturesDict)[iCol].getIndexType())
                {
                case daal::data_management::features::IndexNumType::DAAL_FLOAT32:
                    s |=
                        splitColumn<float, IdxType, cpu>(inputTable, trainTable, testTable, trainIdx, testIdx, nTrainRows, nTestRows, iCol, nThreads);
                    break;
                case daal::data_management::features::IndexNumType::DAAL_FLOAT64:
                    s |= splitColumn<double, IdxType, cpu>(inputTable, trainTable, testTable, trainIdx, testIdx, nTrainRows, nTestRows, iCol,
                                                           nThreads);
                    break;
                default:
                    s |= splitColumn<int, IdxType, cpu>(inputTable, trainTable, testTable, trainIdx, testIdx, nTrainRows, nTestRows, iCol, nThreads);
                }
            });
        return s.detach();
    }
    else
    {
        switch ((*tableFeaturesDict)[0].getIndexType())
        {
        case daal::data_management::features::IndexNumType::DAAL_FLOAT32:
            return splitRows<float, IdxType, cpu>(inputTable, trainTable, testTable, trainIdxTable, testIdxTable, nTrainRows, nTestRows, nColumns,
                                                  nThreads);
            break;
        case daal::data_management::features::IndexNumType::DAAL_FLOAT64:
            return splitRows<double, IdxType, cpu>(inputTable, trainTable, testTable, trainIdxTable, testIdxTable, nTrainRows, nTestRows, nColumns,
                                                   nThreads);
            break;
        default:
            return splitRows<int, IdxType, cpu>(inputTable, trainTable, testTable, trainIdxTable, testIdxTable, nTrainRows, nTestRows, nColumns,
                                                nThreads);
        }
    }
}

template <typename IdxType>
void trainTestSplitDispImpl(const NumericTablePtr & inputTable, const NumericTablePtr & trainTable, const NumericTablePtr & testTable,
                            const NumericTablePtr & trainIdxTable, const NumericTablePtr & testIdxTable)
{
#define DAAL_TRAIN_TEST_SPLIT(cpuId, ...) trainTestSplitImpl<IdxType, cpuId>(__VA_ARGS__);
    DAAL_DISPATCH_FUNCTION_BY_CPU(DAAL_TRAIN_TEST_SPLIT, inputTable, trainTable, testTable, trainIdxTable, testIdxTable);
#undef DAAL_TRAIN_TEST_SPLIT
}

template <typename IdxType>
DAAL_EXPORT void trainTestSplit(const NumericTablePtr & inputTable, const NumericTablePtr & trainTable, const NumericTablePtr & testTable,
                                const NumericTablePtr & trainIdxTable, const NumericTablePtr & testIdxTable)
{
    DAAL_SAFE_CPU_CALL((trainTestSplitDispImpl<IdxType>(inputTable, trainTable, testTable, trainIdxTable, testIdxTable)),
                       (trainTestSplitImpl<IdxType, DAAL_BASE_CPU>(inputTable, trainTable, testTable, trainIdxTable, testIdxTable)));
}

template DAAL_EXPORT void trainTestSplit<int>(const NumericTablePtr & inputTable, const NumericTablePtr & trainTable,
                                              const NumericTablePtr & testTable, const NumericTablePtr & trainIdxTable,
                                              const NumericTablePtr & testIdxTable);

} // namespace internal
} // namespace data_management
} // namespace daal
