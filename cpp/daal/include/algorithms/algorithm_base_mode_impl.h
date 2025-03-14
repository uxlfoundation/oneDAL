/* file: algorithm_base_mode_impl.h */
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

#ifndef __ALGORITHM_BASE_MODE_IMPL_H__
#define __ALGORITHM_BASE_MODE_IMPL_H__

#include "services/daal_defines.h"
#include "algorithms/algorithm_base_common.h"
#include "services/daal_memory.h"
#include "services/internal/daal_kernel_defines.h"

#include "services/host_app.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains version 1.0 of oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__ALGORITHMIMPL"></a>
 * \brief Provides implementations of the compute and finalizeCompute methods of the Algorithm class.
 *        The methods of the class support different computation modes: batch, distributed and online(see \ref ComputeMode)
 * \tparam mode Computation mode of the algorithm, \ref ComputeMode
 */
template <ComputeMode mode>
class DAAL_EXPORT AlgorithmImpl : public Algorithm<mode>
{
public:
    /** Deafult constructor */
    AlgorithmImpl() : wasSetup(false), resetFlag(true), wasFinalizeSetup(false), resetFinalizeFlag(true) {}

    AlgorithmImpl(const AlgorithmImpl & /*other*/)
        : Algorithm<mode>(), wasSetup(false), resetFlag(true), wasFinalizeSetup(false), resetFinalizeFlag(true)
    {}

    virtual ~AlgorithmImpl()
    {
        resetCompute();
        resetFinalizeCompute();
    }

    /**
     * Computes final results of the algorithm in the %batch mode,
     * or partial results of the algorithm in %online and %distributed modes without possibility of throwing an exception.
     */
    services::Status computeNoThrow();

    /**
     * Computes final results of the algorithm in the %batch mode,
     * or partial results of the algorithm in %online and %distributed modes.
     */
    services::Status compute()
    {
        this->_status = computeNoThrow();
        return services::throwIfPossible(this->_status);
    }

    /**
     * Computes final results of the algorithm using partial results in %online and %distributed modes.
     */
    services::Status finalizeComputeNoThrow()
    {
        if (this->isChecksEnabled())
        {
            services::Status s = this->checkPartialResult();
            if (!s) return s;
        }

        services::Status s = this->allocateResultMemory();
        if (!s) return s.add(services::ErrorMemoryAllocationFailed);

        this->_ac->setPartialResult(this->_pres);
        this->_ac->setResult(this->_res);

        if (this->isChecksEnabled())
        {
            s = this->checkFinalizeComputeParams();
            if (!s) return s;
        }

        s = setupFinalizeCompute();
        if (s) s |= this->_ac->finalizeCompute();
        if (resetFinalizeFlag) s |= resetFinalizeCompute();
        return s;
    }

    /**
     * Computes final results of the algorithm using partial results in %online and %distributed modes.
     */
    services::Status finalizeCompute()
    {
        this->_status = finalizeComputeNoThrow();
        return services::throwIfPossible(this->_status);
    }

    /**
     * Validates parameters of the compute method
     */
    virtual services::Status checkComputeParams() DAAL_C11_OVERRIDE
    {
        services::Status s;
        if (this->_par) s = this->_par->check();
        return s.add(this->_in->check(this->_par, this->getMethod()));
    }

    /**
     * Validates result parameters of the compute method
     */
    virtual services::Status checkResult() DAAL_C11_OVERRIDE
    {
        return this->_pres ? this->_pres->check(this->_in, this->_par, this->getMethod()) : services::Status(services::ErrorNullPartialResult);
    }

    /**
     * Validates result parameters of the finalizeCompute method
     */
    virtual services::Status checkPartialResult() DAAL_C11_OVERRIDE
    {
        return this->_pres ? this->_pres->check(this->_par, this->getMethod()) : services::Status(services::ErrorNullPartialResult);
    }

    /**
     * Validates parameters of the finalizeCompute method
     */
    virtual services::Status checkFinalizeComputeParams() DAAL_C11_OVERRIDE
    {
        return this->_res ? this->_res->check(this->_pres, this->_par, this->getMethod()) : services::Status();
    }

    services::Status setupCompute()
    {
        services::Status s;
        if (!wasSetup)
        {
            s        = this->_ac->setupCompute();
            wasSetup = true;
        }
        return s;
    }

    services::Status resetCompute()
    {
        services::Status s;
        if (wasSetup)
        {
            s        = this->_ac->resetCompute();
            wasSetup = false;
        }
        return s;
    }

    void enableResetOnCompute(bool flag) { resetFlag = flag; }

    services::Status setupFinalizeCompute()
    {
        services::Status s;
        if (!wasFinalizeSetup)
        {
            s                = this->_ac->setupFinalizeCompute();
            wasFinalizeSetup = true;
        }
        return s;
    }

    services::Status resetFinalizeCompute()
    {
        services::Status s;
        if (wasFinalizeSetup)
        {
            s                = this->_ac->resetFinalizeCompute();
            wasFinalizeSetup = false;
        }
        return s;
    }

    void enableResetOnFinalizeCompute(bool flag) { resetFinalizeFlag = flag; }
    /**
    * Returns HostAppIface used by the class
    * \return HostAppIface used by the class
    */
    services::HostAppIfacePtr hostApp();

    /**
    * Sets HostAppIface to be used by the class
    * \param pHost to be used by the class
    */
    void setHostApp(const services::HostAppIfacePtr & pHost);

private:
    bool wasSetup;
    bool resetFlag;
    bool wasFinalizeSetup;
    bool resetFinalizeFlag;

    AlgorithmImpl & operator=(const AlgorithmImpl &);
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ALGORITHMIMPL_BATCH"></a>
 * \brief Provides implementations of the compute and checkComputeParams methods of the Algorithm<batch> class
 */
template <>
class DAAL_EXPORT AlgorithmImpl<batch> : public Algorithm<batch>
{
public:
    /** Deafult constructor */
    AlgorithmImpl() : wasSetup(false), resetFlag(true) {}

    AlgorithmImpl(const AlgorithmImpl & /*other*/) : Algorithm<batch>(), wasSetup(false), resetFlag(true) {}

    virtual ~AlgorithmImpl() { resetCompute(); }

    /**
     * Computes final results of the algorithm in the %batch mode without possibility of throwing an exception.
     */
    services::Status computeNoThrow();

    /**
     * Computes final results of the algorithm in the %batch mode.
     */
    services::Status compute()
    {
        this->_status = computeNoThrow();
        return services::throwIfPossible(this->_status);
    }

    /**
     * Validates parameters of the compute method
     */
    virtual services::Status checkComputeParams() DAAL_C11_OVERRIDE
    {
        services::Status s;
        if (_par)
        {
            s = _par->check();
            if (!s) return s;
        }

        return _in->check(_par, getMethod());
    }

    /**
     * Validates result parameters of the compute method
     */
    virtual services::Status checkResult() DAAL_C11_OVERRIDE
    {
        if (_res) return _res->check(_in, _par, getMethod());
        return services::Status(services::ErrorNullResult);
    }

    services::Status setupCompute()
    {
        services::Status s;
        if (!wasSetup)
        {
            s        = this->_ac->setupCompute();
            wasSetup = true;
        }
        return s;
    }

    services::Status resetCompute()
    {
        services::Status s;
        if (wasSetup)
        {
            s        = this->_ac->resetCompute();
            wasSetup = false;
        }
        return s;
    }

    void enableResetOnCompute(bool flag) { resetFlag = flag; }

    /**
    * Returns HostAppIface used by the class
    * \return HostAppIface used by the class
    */
    services::HostAppIfacePtr hostApp();

    /**
    * Sets HostAppIface to be used by the class
    * \param pHost to be used by the class
    */
    void setHostApp(const services::HostAppIfacePtr & pHost);

private:
    bool wasSetup;
    bool resetFlag;

    AlgorithmImpl & operator=(const AlgorithmImpl &);
};
/** @} */
} // namespace interface1
using interface1::AlgorithmImpl;

} // namespace algorithms
} // namespace daal
#endif
