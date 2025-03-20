/** file: finiteness_checker_impl.i */
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

#ifndef __FINITENESS_CHECKER_IMPL_I__
#define __FINITENESS_CHECKER_IMPL_I__

namespace daal
{
namespace data_management
{
namespace internal
{
using namespace daal::internal;
/*
// Computes sum of the elements of input array
*/
template <typename DataType, daal::CpuType cpu>
DataType computeSum(size_t nDataPtrs, size_t nElementsPerPtr, const DataType ** dataPtrs)
{
    DataType sum = 0;
    for (size_t ptrIdx = 0; ptrIdx < nDataPtrs; ++ptrIdx)
        for (size_t i = 0; i < nElementsPerPtr; ++i) sum += dataPtrs[ptrIdx][i];

    return sum;
}

/*
// Computes sum of the elements of an input SOA array
*/
template <daal::CpuType cpu>
double computeSumSOA(NumericTable & table, bool & sumIsFinite, services::Status & st)
{
    double sum                                  = 0;
    const size_t nRows                          = table.getNumberOfRows();
    const size_t nCols                          = table.getNumberOfColumns();
    NumericTableDictionaryPtr tableFeaturesDict = table.getDictionarySharedPtr();

    for (size_t i = 0; (i < nCols) && sumIsFinite; ++i)
    {
        switch ((*tableFeaturesDict)[i].getIndexType())
        {
        case daal::data_management::features::IndexNumType::DAAL_FLOAT32:
        {
            ReadColumns<float, cpu> colBlock(table, i, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS(colBlock);
            const float * colPtr = colBlock.get();
            sum += static_cast<double>(computeSum<float, cpu>(1, nRows, &colPtr));
            break;
        }
        case daal::data_management::features::IndexNumType::DAAL_FLOAT64:
        {
            ReadColumns<double, cpu> colBlock(table, i, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS(colBlock);
            const double * colPtr = colBlock.get();
            sum += computeSum<double, cpu>(1, nRows, &colPtr);
            break;
        }
        default: break;
        }
        sumIsFinite &= !valuesAreNotFinite(&sum, 1, false);
    }

    return sum;
}

/*
// Check finiteness per element using bitwise isnan isinf for a DAAL table
*/
template <typename DataType, daal::CpuType cpu>
bool checkFiniteness(const size_t nElements, size_t nDataPtrs, size_t nElementsPerPtr, const DataType ** dataPtrs, bool allowNaN)
{
    bool notFinite = false;
    // valuesAreNotFinite is defined in finiteness_checker.cpp
    for (size_t ptrIdx = 0; ptrIdx < nDataPtrs; ++ptrIdx) notFinite = notFinite || valuesAreNotFinite(dataPtrs[ptrIdx], nElementsPerPtr, allowNaN);

    return !notFinite;
}

/*
// Check finiteness per element using bitwise isnan isinf for an DAAL SOA table
*/
template <daal::CpuType cpu>
bool checkFinitenessSOA(NumericTable & table, bool allowNaN, services::Status & st)
{
    bool valuesAreFinite                        = true;
    const size_t nRows                          = table.getNumberOfRows();
    const size_t nCols                          = table.getNumberOfColumns();
    NumericTableDictionaryPtr tableFeaturesDict = table.getDictionarySharedPtr();

    for (size_t i = 0; (i < nCols) && valuesAreFinite; ++i)
    {
        switch ((*tableFeaturesDict)[i].getIndexType())
        {
        case daal::data_management::features::IndexNumType::DAAL_FLOAT32:
        {
            ReadColumns<float, cpu> colBlock(table, i, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS(colBlock);
            const float * colPtr = colBlock.get();
            valuesAreFinite &= checkFiniteness<float, cpu>(nRows, 1, nRows, &colPtr, allowNaN);
            break;
        }
        case daal::data_management::features::IndexNumType::DAAL_FLOAT64:
        {
            ReadColumns<double, cpu> colBlock(table, i, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS(colBlock);
            const double * colPtr = colBlock.get();
            valuesAreFinite &= checkFiniteness<double, cpu>(nRows, 1, nRows, &colPtr, allowNaN);
            break;
        }
        default: break;
        }
    }

    return valuesAreFinite;
}

} // namespace internal
} // namespace data_management
} // namespace daal

#endif // __FINITENESS_CHECKER_IMPL_I__
