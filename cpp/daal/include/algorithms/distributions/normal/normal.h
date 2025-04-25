/* file: normal.h */
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
//  Implementation of the normal distribution
//--
*/

#ifndef __NORMAL_H__
#define __NORMAL_H__

#include "algorithms/distributions/distribution.h"
#include "algorithms/distributions/normal/normal_types.h"

namespace daal
{
namespace algorithms
{
namespace distributions
{
namespace normal
{
/**
 * @defgroup distributions_normal_batch Batch
 * @ingroup distributions_normal
 * @{
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DISTRIBUTIONS__NORMAL__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the normal distribution.
 *        This class is associated with the \ref normal::interface1::Batch "normal::Batch" class
 *        and supports the method of normal distribution computation in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of normal distribution, double or float
 * \tparam method           Computation method of the distribution, normal::Method
 * \tparam cpu              Version of the cpu-specific implementation of the distribution, daal::CpuType
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the normal distribution with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    ~BatchContainer();
    /**
     * Computes the result of the normal distribution in the batch processing mode
     *
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DISTRIBUTIONS__NORMAL__BATCH"></a>
 * \brief Provides methods for normal distribution computations in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of normal distribution, double or float
 * \tparam method           Computation method of the distribution, normal::Method
 *
 * \par Enumerations
 *      - normal::Method          Computation methods for the normal distribution
 *
 * \par References
 *      - \ref distributions::interface1::Input "distributions::Input" class
 *      - \ref distributions::interface1::Result "distributions::Result" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public distributions::BatchBase
{
public:
    typedef distributions::BatchBase super;

    typedef typename super::InputType InputType;
    typedef algorithms::distributions::normal::Parameter<algorithmFPType> ParameterType;
    typedef typename super::ResultType ResultType;

    /**
     * Constructs normal distribution
     *  \param[in] a     Mean
     *  \param[in] sigma standard deviation
     */
    Batch(algorithmFPType a = 0.0, algorithmFPType sigma = 1.0);

    /**
     * Constructs normal distribution by copying input objects and parameters of another normal distribution
     * \param[in] other Normal distribution
     */
    Batch(const Batch<algorithmFPType, method> & other);

    /**
     * Returns method of the distribution
     * \return Method of the distribution
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains results of normal distribution
     * \return Structure that contains results of normal distribution
     */
    ResultPtr getResult() { return _result; }

    /**
     * Registers user-allocated memory to store results of normal distribution
     * \param[in] result  Structure to store results of normal distribution
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
     * Returns a pointer to the newly allocated normal distribution
     * with a copy of input objects and parameters of this normal distribution
     * \return Pointer to the newly allocated distribution
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

    /**
     * Allocates memory to store the result of the normal distribution
     *
     * \return Status of computations
     */
    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        _par               = &parameter;
        services::Status s = this->_result->template allocate<algorithmFPType>(&(this->input), &parameter, (int)method);
        this->_res         = this->_result.get();
        return s;
    }

    Parameter<algorithmFPType> parameter; /*!< %Parameters of the normal distribution */

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    void initialize()
    {
        Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in                  = &input;
        _par                 = &parameter;
        _result.reset(new ResultType());
    }

private:
    ResultPtr _result;

    Batch & operator=(const Batch &);
};

} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;
/** @} */
} // namespace normal
} // namespace distributions
} // namespace algorithms
} // namespace daal
#endif
