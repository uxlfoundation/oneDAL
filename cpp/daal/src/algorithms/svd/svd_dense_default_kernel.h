/* file: svd_dense_default_kernel.h */
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
//  Declaration of template function that calculate svds.
//--
*/

#ifndef __SVD_DENSE_DEFAULT_KERNEL_H__
#define __SVD_DENSE_DEFAULT_KERNEL_H__

#include "algorithms/svd/svd_batch.h"
#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace svd
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services;

template <typename algorithmFPType, daal::algorithms::svd::Method method, CpuType cpu>
class SVDBatchKernel : public Kernel
{
public:
    Status compute(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[], const daal::algorithms::Parameter * par = 0);

    Status compute_seq(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[], const Parameter * par = 0);

    Status compute_thr(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[], const Parameter * par = 0);

    Status compute_pcl(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[], const Parameter * par = 0);
};

template <typename algorithmFPType, daal::algorithms::svd::Method method, CpuType cpu>
class SVDOnlineKernel : public Kernel
{
public:
    Status compute(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[], const daal::algorithms::Parameter * par = 0);
    Status finalizeCompute(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[],
                           const daal::algorithms::Parameter * par = 0);
};

template <typename algorithmFPType, daal::algorithms::svd::Method method, CpuType cpu>
class SVDDistributedStep2Kernel : public Kernel
{
public:
    Status compute(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[], const daal::algorithms::Parameter * par = 0);
};

template <typename algorithmFPType, daal::algorithms::svd::Method method, CpuType cpu>
class SVDDistributedStep3Kernel : public Kernel
{
public:
    Status compute(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[], const daal::algorithms::Parameter * par = 0);
};

} // namespace internal
} // namespace svd
} // namespace algorithms
} // namespace daal

#endif
