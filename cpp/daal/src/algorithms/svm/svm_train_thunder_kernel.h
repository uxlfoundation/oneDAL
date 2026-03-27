/* file: svm_train_thunder_kernel.h */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  Declaration of template structs that calculate SVM Training functions.
//--
*/

#ifndef __SVM_TRAIN_THUNDER_KERNEL_H__
#define __SVM_TRAIN_THUNDER_KERNEL_H__

#include "data_management/data/numeric_table.h"
#include "algorithms/model.h"
#include "services/daal_defines.h"
#include "algorithms/svm/svm_train_types.h"
#include "src/algorithms/kernel.h"
#include "src/data_management/service_micro_table.h"

#include "src/algorithms/svm/svm_train_kernel.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace internal
{
using namespace daal::internal;

template <typename algorithmFPType, CpuType cpu>
struct SVMTrainImpl<thunder, algorithmFPType, cpu> : public Kernel
{
    services::Status compute(const data_management::NumericTablePtr & xTable, const data_management::NumericTablePtr & wTable,
                             data_management::NumericTable & yTable, daal::algorithms::Model * r, const KernelParameter & par);

private:
    services::Status classificationInit(NumericTable & yTable, const NumericTablePtr & wTable, const algorithmFPType C, const algorithmFPType nu,
                                        algorithmFPType * y, algorithmFPType * grad, algorithmFPType * alpha, algorithmFPType * cw,
                                        size_t & nNonZeroWeights, const SvmType svmType);

    services::Status regressionInit(NumericTable & yTable, const NumericTablePtr & wTable, const algorithmFPType C, const algorithmFPType nu,
                                    const algorithmFPType epsilon, algorithmFPType * y, algorithmFPType * grad, algorithmFPType * alpha,
                                    algorithmFPType * cw, size_t & nNonZeroWeights, const SvmType svmType);

    services::Status SMOBlockSolver(const algorithmFPType * y, const algorithmFPType * grad, const uint32_t * wsIndices, algorithmFPType ** kernelWS,
                                    const size_t nVectors, const size_t nWS, const algorithmFPType * cw, const double eps, const double tau,
                                    algorithmFPType * buffer, char * I, algorithmFPType * alpha, algorithmFPType * deltaAlpha,
                                    algorithmFPType & localDiff, SvmType svmType) const;

    services::Status updateGrad(algorithmFPType ** kernelWS, const algorithmFPType * deltaalpha, algorithmFPType * grad, const size_t nVectors,
                                const size_t nTrainVectors, const size_t nWS);

    bool checkStopCondition(const algorithmFPType diff, const algorithmFPType diffPrev, const algorithmFPType eps, size_t & sameLocalDiff);

    services::Status initGrad(const NumericTablePtr & xTable, const kernel_function::KernelIfacePtr & kernel, const size_t nVectors,
                              const size_t nTrainVectors, algorithmFPType * const y, algorithmFPType * const alpha, algorithmFPType * grad);

    // One of the conditions for stopping is diff stays unchanged. nNoChanges - number of repetitions
    static const size_t nNoChanges = 5;
    // The maximum numbers of iteration of the subtask is number of observation in WS x cInnerIterations.
    // It's enough to find minimum for subtask.
    static const size_t cInnerIterations = 100;
    // The maximum block size for blocked SMO solver.
    // Need of (maxBlockSize*6 + maxBlockSize*maxBlockSize)*sizeof(algorithmFPType) internal memory.
    // It should fit into the cache L2 including the use of hardware prefetch.
    static const size_t maxBlockSize = 2048;
    // Inner threshold for break from SVM
    static constexpr algorithmFPType accuracyThresholdInner = algorithmFPType(1e-3);

    enum MemSmoId
    {
        alphaBuffID    = 0,
        yBuffID        = 1,
        gradBuffID     = 2,
        kdBuffID       = 3,
        oldAlphaBuffID = 4,
        cwBuffID       = 5,
        latest         = cwBuffID + 1,
    };
};

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
