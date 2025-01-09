/* file: algorithm_container_base.h */
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
//  Implementation of base classes defining algorithm interface.
//--
*/

#ifndef __ALGORITHM_CONTAINER_BASE_H__
#define __ALGORITHM_CONTAINER_BASE_H__

#include "services/daal_memory.h"
#include "services/internal/daal_kernel_defines.h"
#include "algorithms/algorithm_types.h"
#include "algorithms/algorithm_kernel.h"

namespace daal
{
namespace algorithms
{
/**
 * @addtogroup base_algorithms
 * @{
 */
/**
* \brief Contains version 1.0 of Intel(R) oneAPI Data Analytics Library interface.
*/
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__ALGORITHMCONTAINERIFACE"></a>
 * \brief Implements the abstract interface AlgorithmContainerIface. It is associated with the Algorithm class
 *        and supports the methods for computation and finalization of the algorithm results
 *        in the batch, distributed, and online modes.
 */
class AlgorithmContainerIface
{
public:
    DAAL_NEW_DELETE();

    virtual ~AlgorithmContainerIface() {}
};

/**
 * <a name="ALGORITHMS__ALGORITHMCONTAINERIFACEIMPL"></a>
 * \brief Implements the abstract interface AlgorithmContainerIfaceImpl. It is associated with the Algorithm class
 *        and supports the methods for computation and finalization of the algorithm results
 *        in the batch, distributed, and online modes.
 */
class AlgorithmContainerIfaceImpl : public AlgorithmContainerIface
{
public:
    AlgorithmContainerIfaceImpl();
    AlgorithmContainerIfaceImpl(const AlgorithmContainerIfaceImpl &)                   = delete;
    AlgorithmContainerIfaceImpl & operator=(const AlgorithmContainerIfaceImpl & other) = delete;

    /**
     * Default constructor
     * \param[in] daalEnv   Pointer to the structure that contains information about the environment
     */
    AlgorithmContainerIfaceImpl(daal::services::Environment::env * daalEnv) : _env(daalEnv), _kernel(NULL) {}

    virtual ~AlgorithmContainerIfaceImpl() DAAL_C11_OVERRIDE {}

    /**
     * Sets the information about the environment
     * \param[in] daalEnv   Pointer to the structure that contains information about the environment
     */
    void setEnvironment(daal::services::Environment::env * daalEnv) { _env = daalEnv; }

protected:
    daal::services::Environment::env * _env;
    Kernel * _kernel;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ALGORITHMCONTAINER"></a>
 * \brief Abstract interface class that provides virtual methods to access and run implementations
 *        of the algorithms. It is associated with the Algorithm class
 *        and supports the methods for computation and finalization of the algorithm results
 *        in the batch, distributed, and online modes.
 *        The methods of the container are defined in derivative containers defined for each algorithm.
 * \tparam mode Computation mode of the algorithm, \ref ComputeMode
 */
template <ComputeMode mode>
class AlgorithmContainer : public AlgorithmContainerIfaceImpl
{
public:
    AlgorithmContainer();
    AlgorithmContainer(const AlgorithmContainer &)                               = delete;
    AlgorithmContainer<mode> & operator=(const AlgorithmContainer<mode> & other) = delete;

    /**
     * Default constructor
     * \param[in] daalEnv   Pointer to the structure that contains information about the environment
     */
    AlgorithmContainer(daal::services::Environment::env * daalEnv) : AlgorithmContainerIfaceImpl(daalEnv) {}

    virtual ~AlgorithmContainer() {}

    /**
     * Computes final results of the algorithm in the %batch mode,
     * or partial results of the algorithm in %online and %distributed modes.
     * This method behaves similarly to compute method of the Algorithm class.
     */
    virtual services::Status compute() = 0;

    /**
     * Computes final results of the algorithm using partial results in %online and %distributed modes.
     * This method behaves similarly to finalizeCompute method of the Algorithm class.
     */
    virtual services::Status finalizeCompute() = 0;

    /**
     * Setups internal datastructures for compute function.
     */
    virtual services::Status setupCompute() = 0;

    /**
     * Resets internal datastructures for compute function.
     */
    virtual services::Status resetCompute() = 0;

    /**
     * Setups internal datastructures for finalizeCompute function.
     */
    virtual services::Status setupFinalizeCompute() = 0;

    /**
     * Resets internal datastructures for finalizeCompute function.
     */
    virtual services::Status resetFinalizeCompute() = 0;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ALGORITHMCONTAINERIMPL"></a>
 * \brief Abstract interface class that provides virtual methods to access and run implementations
 *        of the algorithms. It is associated with the Algorithm class
 *        and supports the methods for computation and finalization of the algorithm results
 *        in the batch, distributed, and online modes.
 *        The methods of the container are defined in derivative containers defined for each algorithm.
 * \tparam mode Computation mode of the algorithm, \ref ComputeMode
 */
template <ComputeMode mode>
class AlgorithmContainerImpl : public AlgorithmContainer<mode>
{
public:
    AlgorithmContainerImpl(const AlgorithmContainerImpl &)                       = delete;
    AlgorithmContainer<mode> & operator=(const AlgorithmContainer<mode> & other) = delete;

    /**
     * Default constructor
     * \param[in] daalEnv   Pointer to the structure that contains information about the environment
     */
    AlgorithmContainerImpl(daal::services::Environment::env * daalEnv = 0)
        : AlgorithmContainer<mode>(daalEnv), _in(0), _pres(0), _res(0), _par(0), _hpar(0)
    {}

    virtual ~AlgorithmContainerImpl() {}

    /**
     * Sets arguments of the algorithm
     * \param[in] in    Pointer to the input arguments of the algorithm
     * \param[in] pres  Pointer to the partial results of the algorithm
     * \param[in] par   Pointer to the parameters of the algorithm
     * \param[in] hpar  Pointer to the hyperparameters of the algorithm
     */
    void setArguments(Input * in, PartialResult * pres, Parameter * par, const Hyperparameter * hpar)
    {
        _in   = in;
        _pres = pres;
        _par  = par;
        _hpar = hpar;
    }

    /**
     * Sets partial results of the algorithm
     * \param[in] pres   Pointer to the partial results of the algorithm
     */
    void setPartialResult(PartialResult * pres) { _pres = pres; }

    /**
     * Sets final results of the algorithm
     * \param[in] res   Pointer to the final results of the algorithm
     */
    void setResult(Result * res) { _res = res; }

    /**
     * Retrieves final results of the algorithm
     * \return   Pointer to the final results of the algorithm
     */
    Result * getResult() const { return _res; }

    virtual services::Status setupCompute() DAAL_C11_OVERRIDE { return services::Status(); }

    virtual services::Status resetCompute() DAAL_C11_OVERRIDE { return services::Status(); }

    virtual services::Status setupFinalizeCompute() DAAL_C11_OVERRIDE { return services::Status(); }

    virtual services::Status resetFinalizeCompute() DAAL_C11_OVERRIDE { return services::Status(); }

protected:
    Input * _in;
    PartialResult * _pres;
    Result * _res;
    Parameter * _par;
    const Hyperparameter * _hpar;
};

/** @} */
} // namespace interface1
using interface1::AlgorithmContainerImpl;
using interface1::AlgorithmContainerIface;

} // namespace algorithms
} // namespace daal
#endif
