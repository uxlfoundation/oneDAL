/* file: outlier_detection_bacon.h */
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
//  Implementation of the interface for the BACON outlier detection algorithm
//  in the batch processing mode
//--
*/

#ifndef __OUTLIER_DETECTION_BACON_H__
#define __OUTLIER_DETECTION_BACON_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/outlier_detection/outlier_detection_bacon_types.h"

namespace daal
{
namespace algorithms
{
namespace bacon_outlier_detection
{
namespace interface1
{
/**
 * @defgroup bacon_outlier_detection_batch Batch
 * @ingroup bacon_outlier_detection
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__BACON_OUTLIER_DETECTION__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the BACON outlier detection algorithm.
 *        This class is associated with daal::algorithms::bacon_outlier_detection::Batch class
 *        and supports the methods of the BACON outlier detection in the %batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the BACON outlier detection, double or float
 * \tparam method           Computation method of the algorithm, \ref daal::algorithms::bacon_outlier_detection::Method
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the BACON outlier detection algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the BACON outlier detection algorithm in the batch processing mode
     *
     * \return Status of computations
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__BACON_OUTLIER_DETECTION__BATCH"></a>
 * \brief Abstract class that specifies interface of the algorithms for computing BACON outlier detection
 * in the batch processing mode.
 * <!-- \n<a href="DAAL-REF-BACON_OUTLIER_DETECTION-ALGORITHM">BACON outlier detection algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the BACON outlier detection, double or float
 * \tparam method           BACON outlier detection computation method, \ref daal::algorithms::bacon_outlier_detection::Method
 *
 * \par Enumerations
 *      - \ref Method    Computation methods for the BACON outlier detection
 *      - \ref InputId   Identifiers of input objects for the BACON outlier detection
 *      - \ref ResultId  Identifiers of the results of the BACON outlier detection algorithm
 *
 * \par References
 *      - Parameter<defaultDense> class
 *      - Parameter<BACONDense> class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public daal::algorithms::Analysis<batch>
{
public:
    typedef algorithms::bacon_outlier_detection::Input InputType;
    typedef algorithms::bacon_outlier_detection::Parameter ParameterType;
    typedef algorithms::bacon_outlier_detection::Result ResultType;

    /** Default constructor */
    Batch();

    /**
     * Constructs an algorithm for computing BACON outlier detection by copying input objects and parameters
     * of another algorithm for computing BACON outlier detection
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
     * Returns the structure that contains the results of the BACON outlier detection
     * \return Structure that contains the results of the BACON outlier detection algorithm
     */
    ResultPtr getResult() { return _result; }

    /**
     * Registers user-allocated memory to store the results of the BACON outlier detection algorithm
     * \param[in] result  Structure for storing the results of the BACON outlier detection algorithm
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
     * Returns a pointer to the newly allocated algorithm for computing BACON outlier detection
     * with a copy of input objects and parameters of this algorithm for computing BACON outlier detection
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(&input, NULL, (int)method);
        _res               = _result.get();
        return s;
    }

    void initialize()
    {
        Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in                  = &input;
        _par                 = &parameter;
        _result.reset(new ResultType());
    }

public:
    InputType input;         /*!< %Input object */
    ParameterType parameter; /*!< Algorithm parameters */

private:
    ResultPtr _result;
    Batch & operator=(const Batch &);
};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace bacon_outlier_detection
} // namespace algorithms
} // namespace daal
#endif
