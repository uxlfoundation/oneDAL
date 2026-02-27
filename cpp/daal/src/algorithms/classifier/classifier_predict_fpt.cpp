/* file: classifier_predict_fpt.cpp */
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
//  Implementation of classifier prediction Result.
//--
*/

#include "algorithms/classifier/classifier_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace classifier
{
namespace prediction
{
namespace dm  = daal::data_management;
namespace dmi = daal::data_management::internal;

namespace interface2
{
/**
 * Allocates memory for storing prediction results of the classification algorithm
 * \tparam  algorithmFPType     Data type for storing prediction results
 * \param[in] input     Pointer to the input objects of the classification algorithm
 * \param[in] parameter Pointer to the parameters of the classification algorithm
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    services::Status st;
    const Parameter * par = static_cast<const Parameter *>(parameter);
    DAAL_CHECK(par, services::ErrorNullParameterNotSupported);

    const size_t nRows    = (static_cast<const InputIface *>(input))->getNumberOfRows();
    const size_t nClasses = par->nClasses;

    if (par->resultsToEvaluate & computeClassLabels)
    {
        dm::NumericTablePtr nt;
        nt = dm::HomogenNumericTable<algorithmFPType>::create(1, nRows, dm::NumericTableIface::doAllocate, &st);

        set(prediction, nt);
    }
    if (par->resultsToEvaluate & computeClassProbabilities)
    {
        dm::NumericTablePtr nt;
        nt = dm::HomogenNumericTable<algorithmFPType>::create(nClasses, nRows, dm::NumericTableIface::doAllocate, &st);

        set(probabilities, nt);
    }
    if (par->resultsToEvaluate & computeClassLogProbabilities)
    {
        dm::NumericTablePtr nt;
        nt = dm::HomogenNumericTable<algorithmFPType>::create(nClasses, nRows, dm::NumericTableIface::doAllocate, &st);

        set(logProbabilities, nt);
    }
    return st;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par,
                                                                    const int method);

} // namespace interface2

} // namespace prediction
} // namespace classifier
} // namespace algorithms
} // namespace daal
