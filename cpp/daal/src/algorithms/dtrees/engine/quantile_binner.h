/* file: quantile_binner.h */
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
//  Bins the training table once per fit: borders from a sorted uniform row
//  sample, then every row assigned to its bins (design 4.1).
//--
*/

#ifndef __DTREES_ENGINE_QUANTILE_BINNER_H__
#define __DTREES_ENGINE_QUANTILE_BINNER_H__

#include <cstddef>
#include <cstdint>

#include "algorithms/decision_forest/decision_forest_training_parameter.h"
#include "data_management/data/numeric_table.h"
#include "src/algorithms/dtrees/engine/bin_matrix.h"
#include "src/services/service_arrays.h"

namespace daal
{
namespace algorithms
{
namespace dtrees
{
namespace engine
{
namespace internal
{
using daal::internal::CpuType;

// Rows in the binning sample. By the Dvoretzky-Kiefer-Wolfowitz inequality (Massart's constant) a
// sample of m rows has rank error at most sqrt(ln(2 / delta) / (2 m)) for every quantile at once:
// 0.00095 for m = 2^22 and delta = 1e-3, a quarter bin at 256 bins. Tables up to this size bin exactly.
constexpr size_t binSampleRows = size_t(1) << 22;

// Scratch for the sorted sample of one feature group: values, sort payload and merge buffer.
constexpr size_t binSampleScratchBytes = size_t(1) << 30;

// Largest categorical code; codes must be integers in [0, maxCategoryCode].
constexpr double maxCategoryCode = double((uint32_t(1) << 24) - 1);

using decision_forest::training::BinningStrategy;

struct BinningParams
{
    size_t maxBins           = 256;
    size_t minBinSize        = 5;
    BinningStrategy strategy = decision_forest::training::quantiles;
    bool replicate           = false; // one copy of the bins per NUMA node
    size_t sampleRows        = binSampleRows;
};

/// Sample draws from philox4x32x10, read through service_rng: row r's draw is word r of the sequence
/// keyed by seed. Backends without full-counter skip-ahead cannot sample (ErrorMethodNotSupported).
struct PhiloxRowSampler
{
    uint64_t seed;
};

template <typename algorithmFPType, CpuType cpu>
class QuantileBinner
{
public:
    /**
     *  \brief Bins x into out
     *
     *  binned is false, and out unset, when the fit cannot be binned: maxBins above 65536 or a
     *  categorical feature with more than 65535 categories. Such a fit runs the exact phase everywhere.
     */
    template <typename Sampler>
    static services::Status build(const data_management::NumericTable & x, const BinningParams & params, const Sampler & sampler,
                                  BinMatrix<cpu> & out, bool & binned);

    /**
     *  \brief Bins of one feature from its sorted non-NaN sample values s[0, k)
     *
     *  With at most maxDataBins distinct values and allowExact, one bin per value (exact); otherwise
     *  borders from params.strategy, merging adjacent bins whose count scaled by nRows / k is below
     *  minBinSize. borders gets nBins entries, the last one s[k - 1].
     */
    static services::Status computeBorders(const algorithmFPType * s, size_t k, size_t maxDataBins, size_t nRows, const BinningParams & params,
                                           bool allowExact, services::internal::TArray<double, cpu> & borders, size_t & nBins, bool & exact);

    /// bins[i] = number of the nInner sorted borders below x[i * xStride], for non-NaN x.
    static void searchBins(const double * inner, size_t nInner, const algorithmFPType * x, size_t n, size_t xStride, uint32_t * bins);

    /// Sorts values ascending with NaN moved to the end; returns the number of non-NaN values.
    static size_t sortSample(size_t n, algorithmFPType * values, int32_t * scratch);
};

} // namespace internal
} // namespace engine
} // namespace dtrees
} // namespace algorithms
} // namespace daal

#endif
