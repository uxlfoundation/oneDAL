/* file: bernoulli.h */
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
//  Implementation of the bernoulli distribution
//--
*/

#ifndef __BERNOULLI_H__
#define __BERNOULLI_H__

#include "src/algorithms/distributions/distribution.h"
#include "src/algorithms/distributions/bernoulli/bernoulli_types.h"

namespace daal
{
namespace algorithms
{
namespace distributions
{
namespace bernoulli
{
namespace internal
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DISTRIBUTIONS__BERNOULLI__BATCH"></a>
 * \brief Provides methods for bernoulli distribution computations in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of bernoulli distribution, double or float
 * \tparam method           Computation method of the distribution, bernoulli::Method
 *
 * \par Enumerations
 *      - bernoulli::Method          Computation methods for the bernoulli distribution
 *
 * \par References
 *      - \ref distributions::internal::Input "distributions::Input" class
 *      - \ref distributions::internal::Result "distributions::Result" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class Batch : public distributions::BatchBase
{
public:
    typedef distributions::BatchBase super;

    typedef typename super::InputType InputType;
    typedef algorithms::distributions::bernoulli::Parameter<algorithmFPType> ParameterType;
    typedef typename super::ResultType ResultType;

    /**
     * Constructs bernoulli distribution
     *  \param[in] p     Success probability of a trial, value from [0.0; 1.0]
     */
    Batch(algorithmFPType p);

    /**
     * Constructs bernoulli distribution by copying input objects and parameters of another bernoulli distribution
     * \param[in] other Bernoulli distribution
     */
    Batch(const Batch<algorithmFPType, method> & other);

    /**
     * Returns method of the distribution
     * \return Method of the distribution
     */
    int getMethod() const override { return (int)method; }

    /**
     * Returns the structure that contains results of bernoulli distribution
     * \return Structure that contains results of bernoulli distribution
     */
    ResultPtr getResult() { return _result; }

    /**
     * Registers user-allocated memory to store results of bernoulli distribution
     * \param[in] result  Structure to store results of bernoulli distribution
     *
     * \return Status of computations
     */
    services::Status setResult(const ResultPtr & result)
    {
        DAAL_CHECK(result, services::ErrorNullResult)
        _result = result;
        _res    = _result.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated bernoulli distribution
     * with a copy of input objects and parameters of this bernoulli distribution
     * \return Pointer to the newly allocated distribution
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

    /**
     * Allocates memory to store the result of the bernoulli distribution
     *
     * \return Status of computations
     */
    services::Status allocateResult() override
    {
        _par               = &parameter;
        services::Status s = this->_result->template allocate<algorithmFPType>(&(this->input), &parameter, (int)method);
        this->_res         = this->_result.get();
        return s;
    }

    Parameter<algorithmFPType> parameter; /*!< %Parameters of the bernoulli distribution */

protected:
    Batch<algorithmFPType, method> * cloneImpl() const override { return new Batch<algorithmFPType, method>(*this); }

    void initialize();

private:
    ResultPtr _result;

    Batch & operator=(const Batch &);
};

} // namespace internal
} // namespace bernoulli
} // namespace distributions
} // namespace algorithms
} // namespace daal
#endif
