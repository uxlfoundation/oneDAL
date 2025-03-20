/** file: finiteness_checker_fpt_cpu.cpp */
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

#include "data_management/data/internal/finiteness_checker.h"
#include "services/env_detect.h"
#include "src/services/service_data_utils.h"
#include "src/services/service_arrays.h"    // for TArray
#include "src/externals/service_dispatch.h"
#include "src/threading/threading.h"
#include "src/algorithms/service_error_handling.h"
#include "src/data_management/finiteness_checker.h"

#ifdef ONEDAL_XSIMD_ENABLED

namespace xs = xsimd;

#if (__CPUID__(DAAL_CPU) != __sse2__)

namespace daal
{
namespace data_management
{
namespace internal
{

template <>
bool checkFinitenessXSIMD<DAAL_FPTYPE, ONEDAL_XSIMD_ARCH>(const size_t nElements, size_t nDataPtrs, size_t nElementsPerPtr, const DAAL_FPTYPE ** dataPtrs, bool allowNaN)
{
    size_t nBlocksPerPtr = nElementsPerPtr / BLOCK_SIZE;
    if (nBlocksPerPtr == 0) nBlocksPerPtr = 1;
    const bool inParallel     = !(nElements < THREADING_BORDER);
    const size_t nPerBlock    = nElementsPerPtr / nBlocksPerPtr;
    const size_t nSurplus     = nElementsPerPtr % nBlocksPerPtr;
    const size_t nTotalBlocks = nBlocksPerPtr * nDataPtrs;
    constexpr size_t nPerInstr = xs::batch<DAAL_FPTYPE, ONEDAL_XSIMD_ARCH>::size;

    bool finiteness;

    daal::services::internal::TArray<bool, DAAL_CPU> notFiniteArr(nTotalBlocks);
    bool * notFinitePtr = notFiniteArr.get();
    DAAL_CHECK_MALLOC(notFinitePtr);
    for (size_t iBlock = 0; iBlock < nTotalBlocks; ++iBlock) notFinitePtr[iBlock] = false;

    daal::conditional_threader_for(inParallel, nTotalBlocks, [&](size_t iBlock) {
        const size_t ptrIdx        = iBlock / nBlocksPerPtr;
        const size_t blockIdxInPtr = iBlock - nBlocksPerPtr * ptrIdx;
        const size_t start         = blockIdxInPtr * nPerBlock;
        const size_t end           = blockIdxInPtr == nBlocksPerPtr - 1 ? start + nPerBlock + nSurplus : start + nPerBlock;
        const size_t lcSize        = end - start;

        if (allowNaN)
        {
            for (size_t i = 0; i < lcSize / nPerInstr; ++i)
            {
                // load data
                xs::batch<DAAL_FPTYPE, ONEDAL_XSIMD_ARCH> ptri = xs::load_unaligned(dataPtrs[ptrIdx] + start + i * nPerInstr);

                notFinitePtr[iBlock] |= xs::any(xs::isinf(ptri));
            }
        }
        else
        {
            for (size_t i = 0; i < lcSize / nPerInstr; ++i)
            {
                // load data
                xs::batch<DAAL_FPTYPE, ONEDAL_XSIMD_ARCH> ptri = xs::load_unaligned(dataPtrs[ptrIdx] + start + i * nPerInstr);

                notFinitePtr[iBlock] |= xs::any(xs::isinf(ptri));
                notFinitePtr[iBlock] |= xs::any(xs::isnan(ptri));
            }
        }
        size_t offset = start + (lcSize / nPerInstr) * nPerInstr;
        notFinitePtr[iBlock] |= valuesAreNotFinite(dataPtrs[ptrIdx] + offset, end - offset, allowNaN);
    });

    for (size_t iBlock = 0; iBlock < nTotalBlocks; ++iBlock)
        if (notFinitePtr[iBlock])
        {
            finiteness = false;
            return finiteness;
        }
    finiteness = true;
    return finiteness;
}

}
}
}
#endif // (__CPUID__(DAAL_CPU) != __sse2__)

#endif // ONEDAL_XSIMD_ENABLED
