/* file: mrg32k3a.h */
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
//  Implementation of the MRG32k3a engine: a 32-bit combined multiple recursive generator
//  with two components of order 3, optimized for batch processing.
//--
*/

#ifndef __MRG32K3A_H__
#define __MRG32K3A_H__

#include "src/algorithms/engines/mrg32k3a/mrg32k3a_types.h"
#include "src/algorithms/engines/engine_impl.h"

namespace daal
{
namespace algorithms
{
namespace engines
{
namespace mrg32k3a
{
/**
 * @defgroup engines_mrg32k3a_batch Batch
 * @ingroup engines_mrg32k3a
 * @{
 */
namespace internal
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__ENGINES__mrg32k3a__BATCH"></a>
 * \brief Provides methods for mrg32k3a engine computations in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of mrg32k3a engine, double or float
 * \tparam method           Computation method of the engine, mrg32k3a::Method
 *
 * \par Enumerations
 *      - mrg32k3a::Method          Computation methods for the mrg32k3a engine
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
     * Creates mrg32k3a engine
     * \param[in] seed  Initial condition for mrg32k3a engine
     *
     * \return Pointer to mrg32k3a engine
     */
    DAAL_DEPRECATED static services::SharedPtr<Batch<algorithmFPType, method> > create(size_t seed = 777);

    /**
     * Returns method of the engine
     * \return Method of the engine
     */
    int getMethod() const override { return (int)method; }

    /**
     * Returns the structure that contains results of mrg32k3a engine
     * \return Structure that contains results of mrg32k3a engine
     */
    ResultPtr getResult() { return _result; }

    /**
     * Registers user-allocated memory to store results of mrg32k3a engine
     * \param[in] result  Structure to store results of mrg32k3a engine
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
     * Returns a pointer to the newly allocated mrg32k3a engine
     * with a copy of input objects and parameters of this mrg32k3a engine
     * \return Pointer to the newly allocated engine
     */
    /**
     * Allocates memory to store the result of the mrg32k3a engine
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

    EngineCloneReturnType * cloneImpl() const override { return new Batch<algorithmFPType, method>(*this); }

    void initialize();

private:
    ResultPtr _result;

    Batch & operator=(const Batch &);
};
typedef services::SharedPtr<Batch<> > mrg32k3aPtr;
typedef services::SharedPtr<const Batch<> > mrg32k3aConstPtr;

} // namespace internal
using internal::Batch;
using internal::mrg32k3aPtr;
using internal::mrg32k3aConstPtr;
/** @} */
} // namespace mrg32k3a
} // namespace engines
} // namespace algorithms
} // namespace daal
#endif
