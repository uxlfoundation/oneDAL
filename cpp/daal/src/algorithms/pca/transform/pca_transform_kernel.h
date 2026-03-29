/* file: pca_transform_kernel.h */
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
//  Declaration of template function that computes pca transformation
//--
*/

#ifndef __PCA_TRANSFORM_KERNEL_H__
#define __PCA_TRANSFORM_KERNEL_H__

#include "algorithms/pca/transform/pca_transform_batch.h"
#include "src/externals/service_memory.h"
#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"
#include "src/externals/service_blas.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace transform
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::internal;

template <typename algorithmFPType, transform::Method method, CpuType cpu>
class TransformKernel : public Kernel
{
public:
    /**
     *  \brief Compute PCA transformation.
     *
     *  \param data[in]             Matrix of input vectors X
     *  \param eigenvectors[in]     PCA eigenvectors
     *  \param means[in]            PCA means
     *  \param variances[in]        PCA variances
     *  \param eigenvalues[in]        PCA eigenvalues
     *  \param transformedData[out] Transformed data
     */
    services::Status compute(NumericTable & data, NumericTable & eigenvectors, NumericTable * pMeans, NumericTable * pVariances,
                             NumericTable * pEigenvalues, NumericTable & transformedData);

    /**
    *  \brief Function that computes PCA transformation
    *         for a block of input data rows
    *
    *  \param numRows[in]          Number of input data rows
    *  \param numFeatures[in]      Number of features in input data row
    *  \param numComponents[in]    Number of components
    *  \param dataBlock[in]        Block of input data rows
    *  \param eigenvectors[in]     Eigenvectors
    *  \param resultBlock[out]     Resulting block of responses
    */
    void computeTransformedBlock(DAAL_INT * numRows, DAAL_INT * numFeatures, DAAL_INT * numComponents, const algorithmFPType * dataBlock,
                                 const algorithmFPType * eigenvectors, algorithmFPType * resultBlock);

    static const size_t _numRowsInBlock = 256;
};
} // namespace internal
} // namespace transform
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
