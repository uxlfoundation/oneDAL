/* file: roc_auc_score.h */
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

#ifndef __DATA_MANAGEMENT_DATA_INTERNAL_ROC_AUC_SCORE_H__
#define __DATA_MANAGEMENT_DATA_INTERNAL_ROC_AUC_SCORE_H__

#include "data_management/data/numeric_table.h"

namespace daal
{
namespace data_management
{
namespace internal
{
// Default template to double for backwards compatability in daal4py (<2025.5)
template <typename DataType = double>
DAAL_EXPORT double rocAucScore(const NumericTablePtr & truePrediction, const NumericTablePtr & testPrediction);
} // namespace internal
} // namespace data_management
} // namespace daal

#endif
