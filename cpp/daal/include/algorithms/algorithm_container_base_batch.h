/* file: algorithm_container_base_batch.h */
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

#ifndef __ALGORITHM_CONTAINER_BASE_BATCH_H__
#define __ALGORITHM_CONTAINER_BASE_BATCH_H__

#include "algorithms/algorithm_container_base.h"

namespace daal
{
namespace algorithms
{
/**
 * @addtogroup base_algorithms
 * @{
 */
/**
* \brief Contains version 1.0 of oneAPI Data Analytics Library interface.
*/
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ALGORITHMCONTAINER"></a>
 * \brief Abstract interface class that provides virtual methods to access and run implementations
 *        of the algorithms in %batch mode. It is associated with the Algorithm<batch> class
 *        and supports the methods for computation of the algorithm results.
 *        The methods of the container are defined in derivative containers defined for each algorithm.
 */
template <>
class AlgorithmContainer<batch> : public AlgorithmContainerIfaceImpl
{
public:
    AlgorithmContainer();
    AlgorithmContainer(const AlgorithmContainer &)                                 = delete;
    AlgorithmContainer<batch> & operator=(const AlgorithmContainer<batch> & other) = delete;

    /**
     * Default constructor
     * \param[in] daalEnv   Pointer to the structure that contains information about the environment
     */
    AlgorithmContainer(daal::services::Environment::env * daalEnv) : AlgorithmContainerIfaceImpl(daalEnv) {}

    ~AlgorithmContainer() override {}

    /**
     * Computes final results of the algorithm.
     * This method behaves similarly to compute method of the Algorithm<batch> class.
     */
    virtual services::Status compute() = 0;

    /**
     * Setups internal datastructures for compute function.
     */
    virtual services::Status setupCompute() = 0;

    /**
     * Resets internal datastructures for compute function.
     */
    virtual services::Status resetCompute() = 0;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ALGORITHMCONTAINERIMPL"></a>
 * \brief Abstract interface class that provides virtual methods to access and run implementations
 *        of the algorithms in %batch mode. It is associated with the Algorithm<batch> class
 *        and supports the methods for computation of the algorithm results.
 *        The methods of the container are defined in derivative containers defined for each algorithm.
 */
template <>
class AlgorithmContainerImpl<batch> : public AlgorithmContainer<batch>
{
public:
    DAAL_NEW_DELETE();

    AlgorithmContainerImpl(const AlgorithmContainerImpl &)                                 = delete;
    AlgorithmContainerImpl<batch> & operator=(const AlgorithmContainerImpl<batch> & other) = delete;

    /**
     * Default constructor
     * \param[in] daalEnv   Pointer to the structure that contains information about the environment
     */
    AlgorithmContainerImpl(daal::services::Environment::env * daalEnv = 0) : AlgorithmContainer<batch>(daalEnv), _par(0), _in(0), _res(0) {};

    virtual ~AlgorithmContainerImpl() {}

    /**
     * Sets arguments of the algorithm
     * \param[in] in    Pointer to the input arguments of the algorithm
     * \param[in] res   Pointer to the final results of the algorithm
     * \param[in] par   Pointer to the parameters of the algorithm
     * \param[in] hpar  Pointer to the hyperparameters of the algorithm
     */
    void setArguments(Input * in, Result * res, Parameter * par, const Hyperparameter * hpar)
    {
        _in   = in;
        _par  = par;
        _res  = res;
        _hpar = hpar;
    }

    /**
     * Retrieves final results of the algorithm
     * \return   Pointer to the final results of the algorithm
     */
    Result * getResult() { return _res; }

    services::Status setupCompute() override { return services::Status(); }

    services::Status resetCompute() override { return services::Status(); }

protected:
    const Hyperparameter * _hpar;
    Parameter * _par;
    Input * _in;
    Result * _res;
};

/** @} */
} // namespace interface1
} // namespace algorithms
} // namespace daal
#endif
