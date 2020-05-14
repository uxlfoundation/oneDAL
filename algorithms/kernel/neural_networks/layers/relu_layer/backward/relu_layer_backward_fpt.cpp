/* file: relu_layer_backward_fpt.cpp */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
//  Implementation of relu calculation algorithm and types methods.
//--
*/

#include "relu_layer_backward_types.h"
#include "relu_layer_types.h"

#include "service_mkl_tensor.h"
#include "tensor.h"
#include "service_tensor.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace relu
{
namespace backward
{
namespace interface1
{
/**
* Allocates memory to store the result of the backward relu layer
 * \param[in] input     Pointer to an object containing the input data
 * \param[in] method    Computation method for the algorithm
 * \param[in] parameter %Parameter of the backward relu layer
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    using daal::data_management::Tensor;
    using daal::data_management::TensorPtr;
    using daal::internal::MklTensor;

    const Parameter * param = static_cast<const Parameter *>(parameter);
    if (!param->propagateGradient)
    {
        return services::Status();
    }

    const Input * in = static_cast<const Input *>(input);

    TensorPtr valueTable = in->get(auxData);

    if (valueTable == 0)
    {
        return services::Status(services::ErrorNullInputNumericTable);
    }

    if (!get(layers::backward::gradient))
    {
        auto inTensor                                            = in->get(layers::backward::inputGradient);
        data_management::HomogenTensor<algorithmFPType> * inHomo = dynamic_cast<data_management::HomogenTensor<algorithmFPType> *>(inTensor.get());

        if (inHomo && param->allowInplaceComputation)
        {
            set(layers::backward::gradient, inTensor);
        }
        else
        {
            set(layers::backward::gradient, TensorPtr(new MklTensor<algorithmFPType>(valueTable->getDimensions())));
        }
    }
    return services::Status();
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                    const daal::algorithms::Parameter * parameter, const int method);

} // namespace interface1
} // namespace backward
} // namespace relu
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
