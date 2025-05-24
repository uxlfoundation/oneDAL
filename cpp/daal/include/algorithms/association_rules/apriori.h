/* file: apriori.h */
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
//  Implementation of the interface for the association rules algorithm
//--
*/

#ifndef __APRIORI_H__
#define __APRIORI_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/association_rules/apriori_types.h"

namespace daal
{
namespace algorithms
{
namespace association_rules
{
namespace interface1
{
/**
 * @defgroup association_rules_batch Batch
 * @ingroup association_rules
 * @{
 */
/**
* <a name="DAAL-CLASS-ALGORITHMS__ASSOCIATION_RULES__BATCHCONTAINER"></a>
* \brief Provides methods to run implementations of the association rules algorithm.
*        This class is associated with daal::algorithms::association_rules::Batch class.
*
* \tparam algorithmFPType  Data type to use in intermediate computations for the association rules algorithm, double or float
* \tparam method           Association rules algorithm computation method, \ref daal::algorithms::association_rules::Method
*/
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the association rules algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the association rules algorithm in the batch processing mode
     */
    virtual services::Status compute();
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ASSOCIATION_RULES__BATCH"></a>
 * \brief Computes the result of the association rules algorithm in the batch processing mode.
 * <!-- \n<a href="DAAL-REF-ASSOCIATION_RULES-ALGORITHM">Association rules algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the association rules algorithm, double or float
 * \tparam method           Association rules algorithm computation method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method    Association rules computation methods
 *      - \ref InputId   Identifiers of input objects for the association rules algorithm
 *      - \ref ResultId  %Result identifiers for the association rules algorithm
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = apriori>
class DAAL_EXPORT Batch : public daal::algorithms::Analysis<batch>
{
public:
    typedef algorithms::association_rules::Input InputType;
    typedef algorithms::association_rules::Parameter ParameterType;
    typedef algorithms::association_rules::Result ResultType;

    /** Default constructor */
    // defined elsewhere in order to prevent implicit instantation of AlgorithmBatchContainer
    Batch();

    /**
     * Constructs an association rules algorithm by copying input objects and parameters
     * of another association rules algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other);

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains results of the association rules algorithm
     * \return Structure that contains results of the association rules algorithm
     */
    ResultPtr getResult() { return _result; }

    /**
     * Registers user-allocated memory to store results of the association rules algorithm
     * \param[in] res  Structure to store results of the association rules algorithm
     */
    services::Status setResult(const ResultPtr & res)
    {
        DAAL_CHECK(res, services::ErrorNullResult)
        _result = res;
        _res    = _result.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated association rules algorithm with a copy of input objects
     * and parameters of this association rules algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(&input, &parameter, (int)method);
        _res               = _result.get();
        return s;
    }

    void initialize()
    {
        _ac  = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in  = &input;
        _par = &parameter;
        _result.reset(new ResultType());
    }

public:
    InputType input;         /*!< %Input data structure */
    ParameterType parameter; /*!< %Algorithm parameter */

private:
    ResultPtr _result;

    Batch & operator=(const Batch &);
};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace association_rules
} // namespace algorithms
} // namespace daal
#endif
