/* file: pca_dense_svd_base.h */
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
//  Declaration of template structs that calculate PCA SVD.
//--
*/

#ifndef __PCA_DENSE_SVD_BASE_H__
#define __PCA_DENSE_SVD_BASE_H__

#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/pca/pca_dense_base.h"
#include "src/externals/service_math.h"
#include "src/externals/service_memory.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_error_handling.h"
#include "src/threading/threading.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{
using namespace daal::services::internal;
using namespace daal::data_management;
using namespace daal::internal;
enum InputDataType
{
    nonNormalizedDataset = 0, /*!< Original, non-normalized data set */
    normalizedDataset    = 1, /*!< Normalized data set whose feature vectors have zero average and unit variance */
    correlation          = 2  /*!< Correlation matrix */
};

template <typename algorithmFPType, CpuType cpu>
class PCASVDKernelBase : public PCADenseBase<algorithmFPType, cpu>
{
public:
    PCASVDKernelBase() {}
    virtual ~PCASVDKernelBase() {}

protected:
    services::Status computeEigenValues(const data_management::NumericTable & eigenvalues, data_management::NumericTable & singular_values,
                                        size_t nRows);
    services::Status computeExplainedVariancesRatio(const data_management::NumericTable & eigenvalues,
                                                    const data_management::NumericTable & variances,
                                                    data_management::NumericTable & explained_variances_ratio);
    services::Status scaleSingularValues(data_management::NumericTable & eigenvaluesTable, size_t nVectors);
};

template <typename algorithmFPType, CpuType cpu>
services::Status PCASVDKernelBase<algorithmFPType, cpu>::computeEigenValues(const data_management::NumericTable & singular_values,
                                                                            data_management::NumericTable & eigenvalues, size_t nRows)
{
    const size_t nComponents = singular_values.getNumberOfColumns();
    ReadRows<algorithmFPType, cpu> SingularValuesBlock(const_cast<data_management::NumericTable &>(singular_values), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(SingularValuesBlock);
    const algorithmFPType * const SingularValuesArray = SingularValuesBlock.get();
    WriteRows<algorithmFPType, cpu> EigenValuesBlock(eigenvalues, 0, 1);
    DAAL_CHECK_MALLOC(EigenValuesBlock.get());
    algorithmFPType * EigenValuesArray = EigenValuesBlock.get();

    for (size_t i = 0; i < nComponents; i++)
    {
        EigenValuesArray[i] = SingularValuesArray[i] * SingularValuesArray[i] / (nRows - 1);
    }
    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status PCASVDKernelBase<algorithmFPType, cpu>::computeExplainedVariancesRatio(const data_management::NumericTable & eigenvalues,
                                                                                        const data_management::NumericTable & variances,
                                                                                        data_management::NumericTable & explained_variances_ratio)
{
    const size_t nComponents = eigenvalues.getNumberOfColumns();
    ReadRows<algorithmFPType, cpu> eigenValuesBlock(const_cast<data_management::NumericTable &>(eigenvalues), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(eigenValuesBlock);
    const algorithmFPType * const eigenValuesArray = eigenValuesBlock.get();
    WriteRows<algorithmFPType, cpu> explainedVariancesRatioBlock(explained_variances_ratio, 0, 1);
    DAAL_CHECK_MALLOC(explainedVariancesRatioBlock.get());
    algorithmFPType * explainedVariancesRatioArray = explainedVariancesRatioBlock.get();
    algorithmFPType sum                            = 0;
    for (size_t i = 0; i < nComponents; i++)
    {
        sum += eigenValuesArray[i];
    }
    for (size_t i = 0; i < nComponents; i++)
    {
        explainedVariancesRatioArray[i] = eigenValuesArray[i] / sum;
    }
    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status PCASVDKernelBase<algorithmFPType, cpu>::scaleSingularValues(NumericTable & eigenvaluesTable, size_t nVectors)
{
    const size_t nFeatures = eigenvaluesTable.getNumberOfColumns();
    daal::internal::WriteRows<algorithmFPType, cpu> block(eigenvaluesTable, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(block);
    algorithmFPType * eigenvalues = block.get();

    for (size_t i = 0; i < nFeatures; i++)
    {
        eigenvalues[i] = eigenvalues[i] * eigenvalues[i] / (nVectors - 1);
    }
    return services::Status();
}

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal
#endif
