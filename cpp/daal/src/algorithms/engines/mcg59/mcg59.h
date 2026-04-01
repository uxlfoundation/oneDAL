/* file: mcg59.h */
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
//  Implementation of the Mersenne Twister engine in the batch processing mode
//--
*/

#ifndef __MCG59_H__
#define __MCG59_H__

#include "src/algorithms/engines/mcg59/mcg59_types.h"
#include "src/algorithms/engines/engine_impl.h"

namespace daal
{
namespace algorithms
{
namespace engines
{
namespace mcg59
{
/**
 * @defgroup engines_mcg59_batch Batch
 * @ingroup engines_mcg59
 * @{
 */
namespace internal
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__ENGINES__MCG59__BATCH"></a>
 * \brief Provides methods for mcg59 engine computations in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of mcg59 engine, double or float
 * \tparam method           Computation method of the engine, mcg59::Method
 *
 * \par Enumerations
 *      - mcg59::Method          Computation methods for the mcg59 engine
 *
 * \par References
 *      - \ref engines::internal::Input  "engines::Input" class
 *      - \ref engines::internal::Result "engines::Result" class
 *
 * \DAAL_DEPRECATED
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class Batch : public engines::BatchBase
{
public:
    typedef engines::BatchBase super;

    typedef typename super::InputType InputType;
    typedef typename super::ResultType ResultType;

    /**
     * Creates mcg59 engine
     * \param[in] seed  Initial condition for mcg59 engine
     *
     * \return Pointer to mcg59 engine
     */
    DAAL_DEPRECATED static services::SharedPtr<Batch<algorithmFPType, method> > create(size_t seed = 777);

    /**
     * Returns method of the engine
     * \return Method of the engine
     */
    int getMethod() const override { return (int)method; }

    /**
     * Returns the structure that contains results of mcg59 engine
     * \return Structure that contains results of mcg59 engine
     */
    ResultPtr getResult() { return _result; }

    /**
     * Registers user-allocated memory to store results of mcg59 engine
     * \param[in] result  Structure to store results of mcg59 engine
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
     * Returns a pointer to the newly allocated mcg59 engine
     * with a copy of input objects and parameters of this mcg59 engine
     * \return Pointer to the newly allocated engine
     */
    /**
     * Allocates memory to store the result of the mcg59 engine
     *
     * \return Status of computations
     */
    services::Status allocateResult() override
    {
        services::Status s = this->_result->template allocate<algorithmFPType>(&(this->input), NULL, (int)method);
        this->_res         = this->_result.get();
        return s;
    }

protected:
    Batch(size_t seed = 777);

    Batch(const Batch<algorithmFPType, method> & other);

    daal::algorithms::Algorithm<batch> * cloneImpl() const override { return new Batch<algorithmFPType, method>(*this); }

    void initialize();

private:
    ResultPtr _result;

    Batch & operator=(const Batch &);
};
typedef services::SharedPtr<Batch<> > mcg59Ptr;
typedef services::SharedPtr<const Batch<> > mcg59ConstPtr;

} // namespace internal
using internal::Batch;
using internal::mcg59Ptr;
using internal::mcg59ConstPtr;
/** @} */
} // namespace mcg59
} // namespace engines
} // namespace algorithms
} // namespace daal
#endif
