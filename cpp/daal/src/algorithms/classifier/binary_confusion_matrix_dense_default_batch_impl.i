/* file: binary_confusion_matrix_dense_default_batch_impl.i */
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
//  Declaration of template class that computes binary confusion matrix.
//--
*/

#ifndef __BINARY_CONFUSION_MATRIX_DEFAULT_IMPL_I__
#define __BINARY_CONFUSION_MATRIX_DEFAULT_IMPL_I__

#include "src/externals/service_memory.h"
#include "src/data_management/service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace classifier
{
namespace quality_metric
{
namespace binary_confusion_matrix
{
namespace internal
{
using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status BinaryConfusionMatrixKernel<method, algorithmFPType, cpu>::compute(const NumericTable * predictedLabelsTable,
                                                                                    const NumericTable * groundTruthLabelsTable,
                                                                                    NumericTable * confusionMatrixTable,
                                                                                    NumericTable * accuracyMeasuresTable,
                                                                                    const binary_confusion_matrix::Parameter * parameter)
{
    const algorithmFPType zero = 0.0;
    const size_t nVectors      = predictedLabelsTable->getNumberOfRows();

    /* Get input data */
    ReadColumns<algorithmFPType, cpu> mtPredictedLabels(*const_cast<NumericTable *>(predictedLabelsTable), 0, 0, nVectors);
    DAAL_CHECK_BLOCK_STATUS(mtPredictedLabels);
    ReadColumns<algorithmFPType, cpu> mtGroundTruthLabels(*const_cast<NumericTable *>(groundTruthLabelsTable), 0, 0, nVectors);
    DAAL_CHECK_BLOCK_STATUS(mtGroundTruthLabels);

    /* Get memory to write the results */
    const size_t nClasses = 2;
    WriteOnlyRows<int, cpu> mtConfusionMatrix(confusionMatrixTable, 0, nClasses);
    DAAL_CHECK_BLOCK_STATUS(mtConfusionMatrix);
    WriteOnlyRows<algorithmFPType, cpu> mtAccuracyMeasures(accuracyMeasuresTable, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(mtAccuracyMeasures);

    const algorithmFPType * predictedLabelsData   = mtPredictedLabels.get();
    const algorithmFPType * groundTruthLabelsData = mtGroundTruthLabels.get();
    int * confusionMatrixData                     = mtConfusionMatrix.get();
    algorithmFPType * accuracyMeasuresData        = mtAccuracyMeasures.get();

    algorithmFPType beta  = parameter->beta;
    algorithmFPType beta2 = beta * beta;
    service_memset<int, cpu>(confusionMatrixData, 0, nClasses * nClasses);

    /* Compute confusion matrix for two-class classifier */

    PRAGMA_OMP_SIMD
    for (size_t i = 0; i < nVectors; i++)
    {
        const int predictedLabel   = ((predictedLabelsData[i] > zero) ? 0 : 1);
        const int groundTruthLabel = ((groundTruthLabelsData[i] > zero) ? 0 : 1);
        ++confusionMatrixData[groundTruthLabel * 2 + predictedLabel];
    }

    const algorithmFPType tp(confusionMatrixData[0]);
    const algorithmFPType fn(confusionMatrixData[1]);
    const algorithmFPType fp(confusionMatrixData[2]);
    const algorithmFPType tn(confusionMatrixData[3]);

    const algorithmFPType invNVectors = 1.0 / algorithmFPType(nVectors);
    /* Accuracy */
    accuracyMeasuresData[0] = (tp + tn) * invNVectors;
    /* Precision */
    accuracyMeasuresData[1] = tp / (tp + fp);
    /* Recall */
    accuracyMeasuresData[2] = tp / (tp + fn);
    /* F-score */
    accuracyMeasuresData[3] = ((beta2 + 1.0) * tp) / ((beta2 + 1.0) * tp + beta2 * fn + fp);
    /* Specificity */
    accuracyMeasuresData[4] = tn / (fp + tn);
    /* AUC (ability to avoid false classification) */
    accuracyMeasuresData[5] = 0.5 * (accuracyMeasuresData[2] + accuracyMeasuresData[4]);
    return Status();
}

} // namespace internal
} // namespace binary_confusion_matrix
} // namespace quality_metric
} // namespace classifier
} // namespace algorithms
} // namespace daal

#endif
