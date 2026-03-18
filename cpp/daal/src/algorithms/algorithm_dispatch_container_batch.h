/* file: algorithm_dispatch_container_batch.h */
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
//  Implementation of base classes defining algorithm interface for batch processing mode.
//--
*/

#ifndef __ALGORITHM_DISPATCH_CONTAINER_BATCH_H__
#define __ALGORITHM_DISPATCH_CONTAINER_BATCH_H__

#include "src/algorithms/algorithm_dispatch_container_common.h"

namespace daal
{
namespace algorithms
{
namespace internal
{

/**
 * \brief Implements a container to dispatch batch processing algorithms to CPU-specific implementations.
 *
 * \tparam mode                 Computation mode of the algorithm, \ref ComputeMode
 * \tparam sse42Container       Implementation for Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2)
 * \tparam avx2Container        Implementation for Intel(R) Advanced Vector Extensions 2 (Intel(R) AVX2)
 * \tparam avx512Container      Implementation for Intel(R) Xeon(R) processors based on Intel AVX-512
 * \tparam sve                  Implementation for ARM processors based on Arm Scalable Vector Extension
 */

#if defined(TARGET_X86_64)
template <typename sse42Container DAAL_KERNEL_AVX2_ONLY(typename avx2Container)
              DAAL_KERNEL_AVX512_ONLY(typename avx512Container)>
class AlgorithmDispatchContainer<batch, sse42Container DAAL_KERNEL_AVX2_ONLY(avx2Container)
                                            DAAL_KERNEL_AVX512_ONLY(avx512Container)> : public AlgorithmContainerImpl<batch>
#elif defined(TARGET_ARM)
template <typename SVEContainer DAAL_KERNEL_SVE_ONLY(typename sveContainer)>
class AlgorithmDispatchContainer<batch, SVEContainer DAAL_KERNEL_SVE_ONLY(sveContainer)> : public AlgorithmContainerImpl<batch>
#elif defined(TARGET_RISCV64)
template <typename RV64Container DAAL_KERNEL_RV64_ONLY(typename rv64Container)>
class AlgorithmDispatchContainer<batch, RV64Container DAAL_KERNEL_RV64_ONLY(rv64Container)> : public AlgorithmContainerImpl<batch>
#endif
{
public:
    /**
     * Default constructor
     * \param[in] daalEnv   Pointer to the structure that contains information about the environment
     */
    AlgorithmDispatchContainer(daal::services::Environment::env * daalEnv);

    virtual ~AlgorithmDispatchContainer()
    {
        delete _cntr;
        _cntr = 0;
    }

    services::Status compute() override
    {
        _cntr->setArguments(this->_in, this->_res, this->_par, this->_hpar);
        return _cntr->compute();
    }

    services::Status setupCompute() override
    {
        _cntr->setArguments(this->_in, this->_res, this->_par, this->_hpar);
        return _cntr->setupCompute();
    }

    services::Status resetCompute() override { return _cntr->resetCompute(); }

protected:
    AlgorithmContainerImpl<batch> * _cntr;

private:
    AlgorithmDispatchContainer(const AlgorithmDispatchContainer &);
    AlgorithmDispatchContainer & operator=(const AlgorithmDispatchContainer &);
};

} // namespace internal
} // namespace algorithms
} // namespace daal

#endif
