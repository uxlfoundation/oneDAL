/* file: cordistance_batch_impl.i */
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
//  Implementation of correlation distance.
//--
*/

#include "services/daal_defines.h"
#include "src/externals/service_math.h"
#include "src/externals/service_blas.h"
#include "src/threading/threading.h"
#include "src/algorithms/service_error_handling.h"
#include "src/data_management/service_numeric_table.h"

static const int blockSizeDefault = 128;
#include "src/algorithms/cordistance/cordistance_full_impl.i"
#include "src/algorithms/cordistance/cordistance_up_impl.i"
#include "src/algorithms/cordistance/cordistance_lp_impl.i"

using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace correlation_distance
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
bool isFull(NumericTableIface::StorageLayout layout);
template <typename algorithmFPType, CpuType cpu>
bool isUpper(NumericTableIface::StorageLayout layout);
template <typename algorithmFPType, CpuType cpu>
bool isLower(NumericTableIface::StorageLayout layout);
/**
 *  \brief Kernel for Correlation distances calculation
 */
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistanceKernel<algorithmFPType, method, cpu>::compute(const size_t na, const NumericTable * const * a, const size_t nr,
                                                                       NumericTable * r[], const daal::algorithms::Parameter * par)
{
    NumericTable * xTable                          = const_cast<NumericTable *>(a[0]); /* x Input data */
    NumericTable * rTable                          = const_cast<NumericTable *>(r[0]); /* Result */
    const NumericTableIface::StorageLayout rLayout = r[0]->getDataLayout();

    if (isFull<algorithmFPType, cpu>(rLayout))
    {
        if (na == 1)
        {
            return corDistanceFull<algorithmFPType, cpu>(xTable, rTable);
        }
        else if (na == 2)
        {
            NumericTable * yTable = const_cast<NumericTable *>(a[1]); /* y Input data */
            return corDistanceFull<algorithmFPType, cpu>(xTable, yTable, rTable);
        }
        else
        {
            return services::Status(services::ErrorIncorrectNumberOfInputNumericTables);
        }
    }
    else
    {
        if (isLower<algorithmFPType, cpu>(rLayout))
        {
            return corDistanceLowerPacked<algorithmFPType, cpu>(xTable, rTable);
        }
        else if (isUpper<algorithmFPType, cpu>(rLayout))
        {
            return corDistanceUpperPacked<algorithmFPType, cpu>(xTable, rTable);
        }
        else
        {
            return services::Status(services::ErrorIncorrectTypeOfOutputNumericTable);
        }
    }
}

template <typename algorithmFPType, CpuType cpu>
bool isFull(NumericTableIface::StorageLayout layout)
{
    int layoutInt = (int)layout;
    if (packed_mask & layoutInt)
    {
        return false;
    }
    return true;
}

template <typename algorithmFPType, CpuType cpu>
bool isUpper(NumericTableIface::StorageLayout layout)
{
    if (layout == NumericTableIface::upperPackedSymmetricMatrix || layout == NumericTableIface::upperPackedTriangularMatrix)
    {
        return true;
    }
    return false;
}

template <typename algorithmFPType, CpuType cpu>
bool isLower(NumericTableIface::StorageLayout layout)
{
    if (layout == NumericTableIface::lowerPackedSymmetricMatrix || layout == NumericTableIface::lowerPackedTriangularMatrix)
    {
        return true;
    }
    return false;
}

} // namespace internal

} // namespace correlation_distance

} // namespace algorithms

} // namespace daal
