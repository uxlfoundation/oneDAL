/* file: outlierdetection_univariate_kernel.h */
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
//  Declaration of template structs for Outliers Detection.
//--
*/

#ifndef __UNIVAR_OUTLIERDETECTION_KERNEL_H__
#define __UNIVAR_OUTLIERDETECTION_KERNEL_H__

#include "algorithms/outlier_detection/outlier_detection_univariate.h"
#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"
#include "src/externals/service_math.h"
#include "src/externals/service_memory.h"
#include "src/data_management/service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace univariate_outlier_detection
{
namespace internal
{
using namespace daal::internal;
using namespace daal::internal;
using namespace daal::services;

template <typename algorithmFPType, Method method, CpuType cpu>
struct OutlierDetectionKernel : public Kernel
{
    static const size_t blockSize = 1000;

    /** \brief Detect outliers in the data from input table
               and store resulting weights into output table */
    inline static Status computeInternal(size_t nFeatures, size_t nVectors, NumericTable & dataTable, NumericTable & resultTable,
                                         const algorithmFPType * location, const algorithmFPType * scatter, algorithmFPType * invScatter,
                                         const algorithmFPType * threshold)
    {
        ReadRows<algorithmFPType, cpu> dataBlock(dataTable);
        WriteOnlyRows<algorithmFPType, cpu> resultBlock(resultTable);

        const algorithmFPType zero(0.0);
        const algorithmFPType one(1.0);

        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < nFeatures; j++)
        {
            invScatter[j] = one;
            if (scatter[j] != zero)
            {
                invScatter[j] = one / scatter[j];
            }
        }

        size_t nBlocks = nVectors / blockSize;
        if (nBlocks * blockSize < nVectors)
        {
            nBlocks++;
        }

        /* Process input data table in blocks */
        for (size_t iBlock = 0; iBlock < nBlocks; iBlock++)
        {
            size_t startRow     = iBlock * blockSize;
            size_t nRowsInBlock = blockSize;
            if (startRow + nRowsInBlock > nVectors)
            {
                nRowsInBlock = nVectors - startRow;
            }

            const algorithmFPType * data = dataBlock.next(startRow, nRowsInBlock);
            algorithmFPType * weight     = resultBlock.next(startRow, nRowsInBlock);
            DAAL_CHECK_BLOCK_STATUS(dataBlock);
            DAAL_CHECK_BLOCK_STATUS(resultBlock);

            const algorithmFPType * dataPtr = data;
            algorithmFPType * weightPtr     = weight;
            algorithmFPType diff;
            for (size_t i = 0; i < nRowsInBlock; i++, dataPtr += nFeatures, weightPtr += nFeatures)
            {
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j < nFeatures; j++)
                {
                    weightPtr[j] = one;
                    diff         = daal::internal::MathInst<algorithmFPType, cpu>::sFabs(dataPtr[j] - location[j]);
                    if (scatter[j] != zero)
                    {
                        /* Here if scatter is greater than zero */
                        if (diff * invScatter[j] > threshold[j])
                        {
                            weightPtr[j] = zero;
                        }
                    }
                    else
                    {
                        /* Here if scatter is equal to zero */
                        if (diff > zero)
                        {
                            weightPtr[j] = zero;
                        }
                    }
                }
            }
        }
        return Status();
    }

    /** \brief Detect outliers in the data from input numeric table
               and store resulting weights into output numeric table */
    Status compute(NumericTable & dataTable, NumericTable & resultTable, NumericTable * locationTable, NumericTable * scatterTable,
                   NumericTable * thresholdTable);

    void defaultInitialization(algorithmFPType * location, algorithmFPType * scatter, algorithmFPType * threshold, const size_t nFeatures);
};

} // namespace internal

} // namespace univariate_outlier_detection

} // namespace algorithms

} // namespace daal

#endif
