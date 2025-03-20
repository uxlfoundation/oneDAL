/** file: finiteness_checker.h */
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

#ifndef __KERNEL_DATA_MANAGEMENT_FINITENESS_CHECKER_H__
#define __KERNEL_DATA_MANAGEMENT_FINITENESS_CHECKER_H__

#include "data_management/data/numeric_table.h"

namespace daal
{
namespace data_management
{
namespace internal
{
//using namespace daal::internal;


#if defined(DAAL_INTEL_CPP_COMPILER) || (__CPUID__(DAAL_CPU) == __sve__)

const size_t BLOCK_SIZE       = 8192;
const size_t THREADING_BORDER = 262144;

#endif

typedef daal::data_management::NumericTable::StorageLayout NTLayout;

constexpr uint32_t floatExpMask  = 0x7f800000u;
constexpr uint32_t floatFracMask = 0x007fffffu;
constexpr uint32_t floatZeroBits = 0x00000000u;

constexpr uint64_t doubleExpMask  = 0x7ff0000000000000ull;
constexpr uint64_t doubleFracMask = 0x000fffffffffffffull;
constexpr uint64_t doubleZeroBits = 0x0000000000000000ull;

bool valuesAreNotFinite(const float * dataPtr, size_t n, bool allowNaN);

bool valuesAreNotFinite(const double * dataPtr, size_t n, bool allowNaN);

template <typename DataType, daal::CpuType cpu>
services::Status allValuesAreFiniteImpl(NumericTable & table, bool allowNaN, bool * finiteness);

#ifdef ONEDAL_XSIMD_ENABLED

#if (__CPUID__(DAAL_CPU) != __sse2__)
template <typename DataType, typename XSIMDArch>
bool checkFinitenessXSIMD(const size_t nElements, size_t nDataPtrs, size_t nElementsPerPtr, const DataType ** dataPtrs, bool allowNaN);
#endif

#endif

} // namespace internal
} // namespace data_management
} // namespace daal

#endif
