/** file finiteness_checker.cpp */
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

#include "data_management/data/internal/finiteness_checker.h"
#include "data_management/data/numeric_table.h"
#include "services/env_detect.h"
#include "src/services/service_data_utils.h"
#include "src/externals/service_dispatch.h"
#include "src/threading/threading.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_error_handling.h"
#include "src/data_management/finiteness_checker.h"

namespace daal
{
namespace data_management
{
namespace internal
{
using namespace daal::internal;

bool valuesAreNotFinite(const float * dataPtr, size_t n, bool allowNaN)
{
    const uint32_t * uint32Ptr = (const uint32_t *)dataPtr;

    for (size_t i = 0; i < n; ++i)
        // check: all value exponent bits are 1 (so, it's inf or nan) and it's not allowed nan
        if (floatExpMask == (uint32Ptr[i] & floatExpMask) && !(floatZeroBits != (uint32Ptr[i] & floatFracMask) && allowNaN)) return true;
    return false;
}

bool valuesAreNotFinite(const double * dataPtr, size_t n, bool allowNaN)
{
    const uint64_t * uint64Ptr = (const uint64_t *)dataPtr;

    for (size_t i = 0; i < n; ++i)
        // check: all value exponent bits are 1 (so, it's inf or nan) and it's not allowed nan
        if (doubleExpMask == (uint64Ptr[i] & doubleExpMask) && !(doubleZeroBits != (uint64Ptr[i] & doubleFracMask) && allowNaN)) return true;
    return false;
}

template <typename DataType>
DAAL_EXPORT bool allValuesAreFinite(NumericTable & table, bool allowNaN)
{
    bool finiteness = false;

#define DAAL_CHECK_FINITENESS(cpuId, ...) allValuesAreFiniteImpl<DataType, cpuId>(__VA_ARGS__);

    DAAL_DISPATCH_FUNCTION_BY_CPU(DAAL_CHECK_FINITENESS, table, allowNaN, &finiteness);

#undef DAAL_CHECK_FINITENESS

    return finiteness;
}

template DAAL_EXPORT bool allValuesAreFinite<float>(NumericTable & table, bool allowNaN);
template DAAL_EXPORT bool allValuesAreFinite<double>(NumericTable & table, bool allowNaN);

} // namespace internal
} // namespace data_management
} // namespace daal
