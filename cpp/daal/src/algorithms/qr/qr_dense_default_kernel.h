/* file: qr_dense_default_kernel.h */
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
//  Declaration of template function that calculate qrs.
//--
*/

#ifndef __QR_DENSE_DEFAULT_KERNEL_H__
#define __QR_DENSE_DEFAULT_KERNEL_H__

#include "algorithms/qr/qr_batch.h"
#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace qr
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::internal;

template <typename algorithmFPType, daal::algorithms::qr::Method method, CpuType cpu>
class QRBatchKernel : public Kernel
{
public:
    services::Status compute(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[],
                             const daal::algorithms::Parameter * par = 0);

    services::Status compute_seq(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[],
                                 const daal::algorithms::Parameter * par = 0);

    services::Status compute_thr(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[],
                                 const daal::algorithms::Parameter * par = 0);

    services::Status compute_pcl(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[],
                                 const daal::algorithms::Parameter * par = 0);
};

template <typename algorithmFPType, daal::algorithms::qr::Method method, CpuType cpu>
class QROnlineKernel : public Kernel
{
public:
    services::Status compute(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[],
                             const daal::algorithms::Parameter * par = 0);
    services::Status finalizeCompute(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[],
                                     const daal::algorithms::Parameter * par = 0);
};

template <typename algorithmFPType, daal::algorithms::qr::Method method, CpuType cpu>
class QRDistributedStep2Kernel : public Kernel
{
public:
    services::Status compute(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r0, NumericTable * r[],
                             const daal::algorithms::Parameter * par = 0, data_management::KeyValueDataCollection * inCollection = NULL);
};

template <typename algorithmFPType, daal::algorithms::qr::Method method, CpuType cpu>
class QRDistributedStep3Kernel : public Kernel
{
public:
    services::Status compute(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[],
                             const daal::algorithms::Parameter * par = 0);
};

} // namespace internal
} // namespace qr
} // namespace algorithms
} // namespace daal

#endif
