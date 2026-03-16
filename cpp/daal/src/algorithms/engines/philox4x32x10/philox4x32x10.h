/* file: philox4x32x10.h */
/*******************************************************************************
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

#include "src/algorithms/engines/philox4x32x10/philox4x32x10_types.h"
#include "src/algorithms/engines/engine.h"

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
namespace internal
{
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
 *      - \ref engines::internal::Input  "engines::Input" class
 *      - \ref engines::internal::Result "engines::Result" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class Batch : public engines::BatchBase
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
    int getMethod() const override { return (int)method; }

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
    services::Status allocateResult() override
    {
        services::Status s = this->_result->template allocate<algorithmFPType>(&(this->input), NULL, (int)method);
        this->_res         = this->_result.get();
        return s;
    }

    ~Batch();

protected:
    Batch(size_t seed = 777);

    Batch(const Batch<algorithmFPType, method> & other);

    Batch<algorithmFPType, method> * cloneImpl() const override { return new Batch<algorithmFPType, method>(*this); }

    void initialize();

private:
    ResultPtr _result;

    Batch & operator=(const Batch &);
};
typedef services::SharedPtr<Batch<> > philox4x32x10Ptr;
typedef services::SharedPtr<const Batch<> > philox4x32x10ConstPtr;

<<<<<<< HEAD
} // namespace interface1
using interface1::Batch;
using interface1::philox4x32x10Ptr;
using interface1::philox4x32x10ConstPtr;
=======
} // namespace internal
using internal::BatchContainer;
using internal::Batch;
using internal::philox4x32x10Ptr;
using internal::philox4x32x10ConstPtr;
>>>>>>> 8de6185d5 (remove deprecated and daal export macros)
/** @} */
} // namespace philox4x32x10
} // namespace engines
} // namespace algorithms
} // namespace daal
#endif
