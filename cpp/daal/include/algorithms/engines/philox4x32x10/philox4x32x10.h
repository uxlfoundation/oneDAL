/* file: philox4x32x10.h */
/*******************************************************************************
* Copyright 2024 Intel Corporation
* Copyright contributors to the oneDAL project
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
//  Implementation of the Philox4x32-10 engine: a counter-based pseudorandom number generator (PRNG) 
//  that uses 4x32-bit keys and performs 10 rounds of mixing to produce high-quality randomness.
//--
*/

#ifndef __PHILOX4X32X10_H__
#define __PHILOX4X32X10_H__

#include "algorithms/engines/philox4x32x10/philox4x32x10_types.h"
#include "algorithms/engines/engine.h"

namespace daal
{
namespace algorithms
{
namespace engines
{
namespace philox4x32x10
{
/**
 * @defgroup engines_philox4x32x10_batch Batch
 * @ingroup engines_philox4x32x10
 * @{
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__ENGINES__philox4x32x10__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the philox4x32x10 engine.
 *        This class is associated with the \ref philox4x32x10::interface1::Batch "philox4x32x10::Batch" class
 *        and supports the method of philox4x32x10 engine computation in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of philox4x32x10 engine, double or float
 * \tparam method           Computation method of the engine, philox4x32x10::Method
 * \tparam cpu              Version of the cpu-specific implementation of the engine, daal::CpuType
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the philox4x32x10 engine with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    ~BatchContainer();
    /**
     * Computes the result of the philox4x32x10 engine in the batch processing mode
     *
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ENGINES__philox4x32x10__BATCH"></a>
 * \brief Provides methods for philox4x32x10 engine computations in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of philox4x32x10 engine, double or float
 * \tparam method           Computation method of the engine, philox4x32x10::Method
 *
 * \par Enumerations
 *      - philox4x32x10::Method          Computation methods for the philox4x32x10 engine
 *
 * \par References
 *      - \ref engines::interface1::Input  "engines::Input" class
 *      - \ref engines::interface1::Result "engines::Result" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public engines::BatchBase
{
public:
    typedef engines::BatchBase super;

    typedef typename super::InputType InputType;
    typedef typename super::ResultType ResultType;

    /**
     * Creates philox4x32x10 engine
     * \param[in] seed  Initial condition for philox4x32x10 engine
     *
     * \return Pointer to philox4x32x10 engine
     */
    static services::SharedPtr<Batch<algorithmFPType, method> > create(size_t seed = 777);

    /**
     * Returns method of the engine
     * \return Method of the engine
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns the structure that contains results of philox4x32x10 engine
     * \return Structure that contains results of philox4x32x10 engine
     */
    ResultPtr getResult() { return _result; }

    /**
     * Registers user-allocated memory to store results of philox4x32x10 engine
     * \param[in] result  Structure to store results of philox4x32x10 engine
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
     * Returns a pointer to the newly allocated philox4x32x10 engine
     * with a copy of input objects and parameters of this philox4x32x10 engine
     * \return Pointer to the newly allocated engine
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

    /**
     * Allocates memory to store the result of the philox4x32x10 engine
     *
     * \return Status of computations
     */
    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = this->_result->template allocate<algorithmFPType>(&(this->input), NULL, (int)method);
        this->_res         = this->_result.get();
        return s;
    }

protected:
    Batch(size_t seed = 777) { initialize(); }

    Batch(const Batch<algorithmFPType, method> & other) : super(other) { initialize(); }

    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    void initialize()
    {
        Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in                  = &input;
        _result.reset(new ResultType());
    }

private:
    ResultPtr _result;

    Batch & operator=(const Batch &);
};
typedef services::SharedPtr<Batch<> > philox4x32x10Ptr;
typedef services::SharedPtr<const Batch<> > philox4x32x10ConstPtr;

} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;
using interface1::philox4x32x10Ptr;
using interface1::philox4x32x10ConstPtr;
/** @} */
} // namespace philox4x32x10
} // namespace engines
} // namespace algorithms
} // namespace daal
#endif
