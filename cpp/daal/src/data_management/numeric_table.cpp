/** file numeric_table.cpp */
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

#include "algorithms/algorithm_types.h"
#include "data_management/data/numeric_table.h"

#include "data_management/data/homogen_numeric_table.h"
#include "data_management/data/merged_numeric_table.h"
#include "data_management/data/row_merged_numeric_table.h"
#include "data_management/data/aos_numeric_table.h"
#include "data_management/data/csr_numeric_table.h"
#include "data_management/data/soa_numeric_table.h"
#include "data_management/data/data_collection.h"
#include "data_management/data/memory_block.h"
#include "data_management/data/internal/base_arrow_numeric_table.h"
#include "src/data_management/service_numeric_table.h"
#include "src/services/service_defines.h"

using namespace daal::services;

/**
 * Checks the correctness of this numeric table
 * \param[in]  nt                The numeric table to check
 * \param[out] errors            The collection of errors
 * \param[in]  description       Additional information about error
 * \param[in]  unexpectedLayouts The bit mask of invalid layouts for this numeric table.
 * \param[in]  expectedLayouts   The bit mask of valid layouts for this numeric table.
 * \param[in]  nColumns          Required number of columns.
 *                               nColumns = 0 means that required number of columns is not specified.
 * \param[in]  nRows             Required number of rows.
 *                               nRows = 0 means that required number of rows is not specified.
 * \return                       Check status: True if the table satisfies the requirements, false otherwise.
 */
Status daal::data_management::checkNumericTable(const NumericTable * nt, const char * description, const int unexpectedLayouts,
                                                const int expectedLayouts, size_t nColsValid, size_t nRowsValid, bool checkDataAllocation)
{
    using namespace daal::services;

    if (nt == 0) return Status(Error::create(ErrorNullNumericTable, ArgumentName, description));

    size_t nColumns = nt->getNumberOfColumns();
    size_t nRows    = nt->getNumberOfRows();

    if (unexpectedLayouts != 0)
    {
        const int state = (int)nt->getDataLayout() & unexpectedLayouts;
        if (state != 0) return Status(Error::create(ErrorIncorrectTypeOfNumericTable, ArgumentName, description));
    }

    if (expectedLayouts != 0)
    {
        const int state = (int)nt->getDataLayout() & expectedLayouts;
        if (state == 0) return Status(Error::create(ErrorIncorrectTypeOfNumericTable, ArgumentName, description));
    }

    if (nColsValid != 0 && nColumns != nColsValid) return Status(Error::create(ErrorIncorrectNumberOfColumns, ArgumentName, description));

    if (nRowsValid != 0 && nRows != nRowsValid)
    {
        auto e = Error::create(ErrorIncorrectNumberOfRows, ArgumentName, description);
        e->addIntDetail(ExpectedValue, (int)nRowsValid);
        e->addIntDetail(ActualValue, (int)nRows);
        return Status(e);
    }

    return nt->check(description, checkDataAllocation);
}

Status daal::data_management::NumericTable::allocateBasicStatistics()
{
    size_t ncols = getNumberOfColumns();

    Status status;

    if (basicStatistics.get(NumericTable::minimum).get() == NULL || basicStatistics.get(NumericTable::minimum)->getNumberOfColumns() != ncols)
    {
        basicStatistics.set(NumericTable::minimum, data_management::HomogenNumericTable<DAAL_SUMMARY_STATISTICS_TYPE>::create(
                                                       ncols, 1, NumericTableIface::doAllocate, &status));
    }
    if (basicStatistics.get(NumericTable::maximum).get() == NULL || basicStatistics.get(NumericTable::maximum)->getNumberOfColumns() != ncols)
    {
        basicStatistics.set(NumericTable::maximum, data_management::HomogenNumericTable<DAAL_SUMMARY_STATISTICS_TYPE>::create(
                                                       ncols, 1, NumericTableIface::doAllocate, &status));
    }
    if (basicStatistics.get(NumericTable::sum).get() == NULL || basicStatistics.get(NumericTable::sum)->getNumberOfColumns() != ncols)
    {
        basicStatistics.set(NumericTable::sum, data_management::HomogenNumericTable<DAAL_SUMMARY_STATISTICS_TYPE>::create(
                                                   ncols, 1, NumericTableIface::doAllocate, &status));
    }
    if (basicStatistics.get(NumericTable::sumSquares).get() == NULL || basicStatistics.get(NumericTable::sumSquares)->getNumberOfColumns() != ncols)
    {
        basicStatistics.set(NumericTable::sumSquares, data_management::HomogenNumericTable<DAAL_SUMMARY_STATISTICS_TYPE>::create(
                                                          ncols, 1, NumericTableIface::doAllocate, &status));
    }
    return status;
}

namespace daal
{
namespace data_management
{
#define DAAL_IMPL_CONVERTTOHOMOGEN_FAST(T)                                                                                                   \
    template <>                                                                                                                              \
    DAAL_EXPORT daal::data_management::NumericTablePtr convertToHomogen<T>(NumericTable & src, daal::MemType type)                           \
    {                                                                                                                                        \
        using namespace daal::services;                                                                                                      \
                                                                                                                                             \
        size_t ncols                                 = src.getNumberOfColumns();                                                             \
        size_t nrows                                 = src.getNumberOfRows();                                                                \
        NumericTableDictionaryPtr srcDict            = src.getDictionarySharedPtr();                                                         \
        DictionaryIface::FeaturesEqual featuresEqual = srcDict->getFeaturesEqual();                                                          \
                                                                                                                                             \
        SharedPtr<HomogenNumericTable<T> > dst = HomogenNumericTable<T>::create(featuresEqual, ncols, nrows, NumericTableIface::doAllocate); \
        NumericTableDictionaryPtr dstDict      = dst->getDictionarySharedPtr();                                                              \
                                                                                                                                             \
        if (featuresEqual == DictionaryIface::equal)                                                                                         \
        {                                                                                                                                    \
            dstDict->setFeature(srcDict->operator[](0), 0);                                                                                  \
        }                                                                                                                                    \
        else                                                                                                                                 \
        {                                                                                                                                    \
            for (size_t i = 0; i < ncols; i++)                                                                                               \
            {                                                                                                                                \
                dstDict->setFeature(srcDict->operator[](i), i);                                                                              \
            }                                                                                                                                \
        }                                                                                                                                    \
                                                                                                                                             \
        BlockDescriptor<T> block;                                                                                                            \
        src.getBlockOfRows(0, nrows, readOnly, block);                                                                                       \
        daal::services::internal::daal_memcpy_s(dst->getArray(), nrows * ncols * sizeof(T), block.getBlockPtr(), nrows * ncols * sizeof(T)); \
        src.releaseBlockOfRows(block);                                                                                                       \
                                                                                                                                             \
        return dst;                                                                                                                          \
    }

#define DAAL_IMPL_CONVERTTOHOMOGEN_SLOW(T)                                                                                                   \
    template <>                                                                                                                              \
    DAAL_EXPORT daal::data_management::NumericTablePtr convertToHomogen<T>(NumericTable & src, daal::MemType type)                           \
    {                                                                                                                                        \
        using namespace daal::services;                                                                                                      \
                                                                                                                                             \
        size_t ncols                                 = src.getNumberOfColumns();                                                             \
        size_t nrows                                 = src.getNumberOfRows();                                                                \
        NumericTableDictionaryPtr srcDict            = src.getDictionarySharedPtr();                                                         \
        DictionaryIface::FeaturesEqual featuresEqual = srcDict->getFeaturesEqual();                                                          \
                                                                                                                                             \
        SharedPtr<HomogenNumericTable<T> > dst = HomogenNumericTable<T>::create(featuresEqual, ncols, nrows, NumericTableIface::doAllocate); \
        NumericTableDictionaryPtr dstDict      = dst->getDictionarySharedPtr();                                                              \
                                                                                                                                             \
        if (featuresEqual == DictionaryIface::equal)                                                                                         \
        {                                                                                                                                    \
            dstDict->setFeature(srcDict->operator[](0), 0);                                                                                  \
        }                                                                                                                                    \
        else                                                                                                                                 \
        {                                                                                                                                    \
            for (size_t i = 0; i < ncols; i++)                                                                                               \
            {                                                                                                                                \
                dstDict->setFeature(srcDict->operator[](i), i);                                                                              \
            }                                                                                                                                \
        }                                                                                                                                    \
                                                                                                                                             \
        BlockDescriptor<> block;                                                                                                             \
        src.getBlockOfRows(0, nrows, readOnly, block);                                                                                       \
        float * srcData = block.getBlockPtr();                                                                                               \
        T * dstData     = dst->getArray();                                                                                                   \
        for (size_t i = 0; i < ncols * nrows; i++)                                                                                           \
        {                                                                                                                                    \
            dstData[i] = (T)srcData[i];                                                                                                      \
        }                                                                                                                                    \
        src.releaseBlockOfRows(block);                                                                                                       \
                                                                                                                                             \
        return dst;                                                                                                                          \
    }

#define DAAL_INSTANTIATE_FAST(T) DAAL_IMPL_CONVERTTOHOMOGEN_FAST(T)

#define DAAL_INSTANTIATE_SLOW(T) DAAL_IMPL_CONVERTTOHOMOGEN_SLOW(T)

DAAL_INSTANTIATE_FAST(float)
DAAL_INSTANTIATE_FAST(double)
DAAL_INSTANTIATE_FAST(int)
DAAL_INSTANTIATE_SLOW(unsigned int)
DAAL_INSTANTIATE_SLOW(DAAL_INT64)
DAAL_INSTANTIATE_SLOW(DAAL_UINT64)
DAAL_INSTANTIATE_SLOW(char)
DAAL_INSTANTIATE_SLOW(unsigned char)
DAAL_INSTANTIATE_SLOW(short)
DAAL_INSTANTIATE_SLOW(unsigned short)
DAAL_INSTANTIATE_SLOW(unsigned long)
DAAL_INSTANTIATE_SLOW(long)

IMPLEMENT_SERIALIZABLE_TAG(SOANumericTable, SERIALIZATION_SOA_NT_ID)
IMPLEMENT_SERIALIZABLE_TAG(BaseArrowImmutableNumericTable, SERIALIZATION_ARROW_IMMUTABLE_NT_ID)
IMPLEMENT_SERIALIZABLE_TAG(CSRNumericTable, SERIALIZATION_CSR_NT_ID)
IMPLEMENT_SERIALIZABLE_TAG(AOSNumericTable, SERIALIZATION_AOS_NT_ID)
IMPLEMENT_SERIALIZABLE_TAG(MergedNumericTable, SERIALIZATION_MERGE_NT_ID)
IMPLEMENT_SERIALIZABLE_TAG(RowMergedNumericTable, SERIALIZATION_ROWMERGE_NT_ID)
IMPLEMENT_SERIALIZABLE_TAG(DataCollection, SERIALIZATION_DATACOLLECTION_ID)
IMPLEMENT_SERIALIZABLE_TAG(MemoryBlock, SERIALIZATION_MEMORY_BLOCK_ID)

namespace interface1
{
IMPLEMENT_SERIALIZABLE_TAG1T_SPECIALIZATION(SerializableKeyValueCollection, SerializationIface, SERIALIZATION_KEYVALUEDATACOLLECTION_ID)

#define DAAL_INSTANTIATE_SER_TAG(T)                                                                                                                 \
    IMPLEMENT_SERIALIZABLE_TAG1T(HomogenNumericTable, T, SERIALIZATION_HOMOGEN_NT_ID)

DAAL_INSTANTIATE_SER_TAG(float)
DAAL_INSTANTIATE_SER_TAG(double)
DAAL_INSTANTIATE_SER_TAG(int)
DAAL_INSTANTIATE_SER_TAG(unsigned int)
DAAL_INSTANTIATE_SER_TAG(DAAL_INT64)
DAAL_INSTANTIATE_SER_TAG(DAAL_UINT64)
DAAL_INSTANTIATE_SER_TAG(char)
DAAL_INSTANTIATE_SER_TAG(unsigned char)
DAAL_INSTANTIATE_SER_TAG(short)
DAAL_INSTANTIATE_SER_TAG(unsigned short)
DAAL_INSTANTIATE_SER_TAG(unsigned long)
DAAL_INSTANTIATE_SER_TAG(long)

Status RowMergedNumericTable::setNumberOfColumnsImpl(size_t ncols)
{
    for (size_t i = 0; i < _tables->size(); i++)
    {
        NumericTable * nt = (NumericTable *)(_tables->operator[](i).get());
        nt->setNumberOfColumns(ncols);
    }
    return NumericTable::setNumberOfColumnsImpl(ncols);
}

Status RowMergedNumericTable::allocateDataMemoryImpl(daal::MemType type)
{
    for (size_t i = 0; i < _tables->size(); i++)
    {
        NumericTable * nt = (NumericTable *)(_tables->operator[](i).get());
        nt->allocateDataMemory(type);
    }
    return Status();
}

void RowMergedNumericTable::freeDataMemoryImpl()
{
    for (size_t i = 0; i < _tables->size(); i++)
    {
        NumericTable * nt = (NumericTable *)(_tables->operator[](i).get());
        nt->freeDataMemory();
    }
}

Status MergedNumericTable::setNumberOfRowsImpl(size_t nrow)
{
    return resize(nrow);
}

Status MergedNumericTable::allocateDataMemoryImpl(daal::MemType type)
{
    for (size_t i = 0; i < _tables->size(); i++)
    {
        NumericTable * nt = (NumericTable *)(_tables->operator[](i).get());
        nt->allocateDataMemory(type);
    }
    return Status();
}

void MergedNumericTable::freeDataMemoryImpl()
{
    for (size_t i = 0; i < _tables->size(); i++)
    {
        NumericTable * nt = (NumericTable *)(_tables->operator[](i).get());
        nt->freeDataMemory();
    }
}

} // namespace interface1
} // namespace data_management
} // namespace daal

namespace daal
{
namespace internal
{
using namespace daal::services;
using namespace daal::data_management;

template <typename algorithmFPType>
Status createSparseTableImpl(const NumericTablePtr & inTable, CSRNumericTablePtr & resTable)
{
    DAAL_CHECK(inTable, ErrorNullNumericTable);

    const size_t nFeatures     = inTable->getNumberOfColumns();
    const size_t nObservations = inTable->getNumberOfRows();

    CSRNumericTableIfacePtr inputTable = dynamicPointerCast<CSRNumericTableIface, NumericTable>(inTable);
    DAAL_CHECK(inputTable, ErrorNullNumericTable);

    size_t * resColIndices    = NULL;
    size_t * resRowIndices    = NULL;
    algorithmFPType * resData = NULL;
    const size_t dataSize     = inputTable->getDataSize();

    Status s;
    CSRBlockDescriptor<algorithmFPType> inputBlock;
    DAAL_CHECK_STATUS(s, inputTable->getSparseBlock(0, nObservations, readOnly, inputBlock));

    resTable =
        CSRNumericTable::create(resData, resColIndices, resRowIndices, nFeatures, nObservations, CSRNumericTableIface::CSRIndexing::oneBased, &s);
    DAAL_CHECK_STATUS_VAR(s);

    DAAL_CHECK_STATUS(s, resTable->allocateDataMemory(dataSize));
    resTable->getArrays<algorithmFPType>(&resData, &resColIndices, &resRowIndices);

    size_t * inColIndices = inputBlock.getBlockColumnIndicesPtr();
    size_t * inRowIndices = inputBlock.getBlockRowIndicesPtr();

    for (size_t i = 0; i < dataSize; i++)
    {
        resColIndices[i] = inColIndices[i];
    }
    for (size_t i = 0; i < nObservations + 1; i++)
    {
        resRowIndices[i] = inRowIndices[i];
    }

    s = inputTable->releaseSparseBlock(inputBlock);

    return s;
}

template <>
Status createSparseTable<double>(const NumericTablePtr & inputTable, CSRNumericTablePtr & resTable)
{
    return createSparseTableImpl<double>(inputTable, resTable);
}

template <>
Status createSparseTable<float>(const NumericTablePtr & inputTable, CSRNumericTablePtr & resTable)
{
    return createSparseTableImpl<float>(inputTable, resTable);
}

} // namespace internal

IMPLEMENT_SERIALIZABLE_TAG(algorithms::OptionalArgument, SERIALIZATION_OPTIONAL_RESULT_ID)

} // namespace daal
