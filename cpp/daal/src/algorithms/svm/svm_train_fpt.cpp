/* file: svm_train_fpt.cpp */
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
//  Implementation of cholesky algorithm and types methods.
//--
*/

#include "algorithms/svm/svm_train_types.h"
#include "src/algorithms/svm/svm_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace interface1
{
/**
 * Allocates memory for storing SVM training results
 * \param[in] input     Pointer to input structure
 * \param[in] parameter Pointer to parameter structure
 * \param[in] method    Algorithm method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    const classifier::training::Input * algInput = static_cast<const classifier::training::Input *>(input);

    services::Status st;
    set(classifier::training::model, daal::algorithms::svm::ModelPtr(new svm::internal::ModelImpl(algorithmFPType(0), 2 /* number of classes */, algInput->get(classifier::training::data)->getNumberOfColumns(),
                                                                     algInput->get(classifier::training::data)->getDataLayout(), st)));
    return st;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                    const daal::algorithms::Parameter * parameter, const int method);

} // namespace interface1
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal
