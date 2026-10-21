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

/// Binning time against the legacy IndexedFeatures, and the rank error of the 2^22-row sample, on a
/// float32 .npy table (Higgs by default; set BINNING_BENCHMARK_NPY). Run explicitly:
///   bazel run --config=host //cpp/daal/src/algorithms/dtrees/engine:binning_benchmark
/// The legacy binner is compiled for the base CpuType in this file, as its sort is scalar code.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "oneapi/dal/test/engine/common.hpp"
#include "data_management/data/homogen_numeric_table.h"
#include "services/env_detect.h"
#include "services/internal/daal_kernel_defines.h"
#include "src/algorithms/dtrees/dtrees_train_data_helper.i" // legacy binner, as the legacy forest includes it
#include "src/algorithms/dtrees/engine/quantile_binner.h"
#include "src/externals/service_rng.h"

namespace daal::algorithms::dtrees::engine::internal::test
{
using namespace daal::data_management;
using daal::internal::CpuType;

constexpr const char * defaultNpy = "/nfs/site/disks/dal_datasets/DAAL_datasets/higgs_10500K_x_train.npy";

/// Reads a 2-D little-endian float32 .npy file into row-major data.
bool loadNpy(const std::string & path, std::vector<float> & data, size_t & nRows, size_t & nCols)
{
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[10];
    if (std::fread(magic, 1, 10, f) != 10 || std::memcmp(magic, "\x93NUMPY", 6) != 0)
    {
        std::fclose(f);
        return false;
    }
    const size_t headerLen = size_t(uint8_t(magic[8])) | (size_t(uint8_t(magic[9])) << 8);
    std::string header(headerLen, ' ');
    if (std::fread(&header[0], 1, headerLen, f) != headerLen || header.find("'<f4'") == std::string::npos)
    {
        std::fclose(f);
        return false;
    }
    const bool fortran = header.find("'fortran_order': True") != std::string::npos;
    const size_t open  = header.find('(');
    if (std::sscanf(header.c_str() + open, "(%zu, %zu)", &nRows, &nCols) != 2)
    {
        std::fclose(f);
        return false;
    }
    std::vector<float> raw(nRows * nCols);
    const bool ok = std::fread(raw.data(), sizeof(float), raw.size(), f) == raw.size();
    std::fclose(f);
    if (!ok) return false;
    if (!fortran)
    {
        data.swap(raw);
        return true;
    }
    data.resize(raw.size());
    daal::threader_for(nCols, 1, [&](size_t c) {
        for (size_t r = 0; r < nRows; ++r) data[r * nCols + c] = raw[c * nRows + r];
    });
    return true;
}

template <typename F>
double seconds(const F & f)
{
    const auto t0 = std::chrono::steady_clock::now();
    f();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

template <CpuType cpu>
double timeNewBinner(const NumericTable & x, BinMatrix<cpu> & bins)
{
    BinningParams params;
    bool binned = false;
    daal::internal::BaseRNGsInst<cpu> engine(777, __DAAL_BRNG_PHILOX4X32X10);
    const double t = seconds([&]() { REQUIRE(QuantileBinner<float, cpu>::build(x, params, engine.getState(), bins, binned).ok()); });
    REQUIRE(binned);
    return t;
}

/// Largest distance from j / B to the interval of full-column ranks a border spans, over the
/// borders of non-exact features with all B bins; borders map one-to-one to j only there.
template <CpuType cpu>
double maxRankError(const std::vector<float> & data, size_t nRows, size_t nCols, const BinMatrix<cpu> & bins, size_t & nChecked)
{
    std::vector<double> perFeature(nCols, 0.0);
    std::vector<size_t> checked(nCols, 0);
    daal::threader_for(nCols, 1, [&](size_t f) {
        if (bins.exactBins(f) || bins.nBins(f) != BinningParams().maxBins) return;
        std::vector<float> col(nRows);
        for (size_t r = 0; r < nRows; ++r) col[r] = data[r * nCols + f];
        std::sort(col.begin(), col.end());
        const size_t B = bins.nBins(f);
        for (size_t j = 1; j < B; ++j)
        {
            const float b      = float(bins.borders(f)[j - 1]);
            const double below = double(std::lower_bound(col.begin(), col.end(), b) - col.begin()) / double(nRows);
            const double upto  = double(std::upper_bound(col.begin(), col.end(), b) - col.begin()) / double(nRows);
            const double t     = double(j) / double(B);
            const double d     = t < below ? below - t : (t > upto ? t - upto : 0.0);
            perFeature[f]      = std::max(perFeature[f], d);
        }
        checked[f] = 1;
    });
    nChecked = 0;
    for (size_t c : checked) nChecked += c;
    return *std::max_element(perFeature.begin(), perFeature.end());
}

TEST("binning benchmark", "[.benchmark]")
{
    const char * env       = std::getenv("BINNING_BENCHMARK_NPY");
    const std::string path = env ? env : defaultNpy;
    std::vector<float> data;
    size_t nRows = 0, nCols = 0;
    SKIP_IF(!loadNpy(path, data, nRows, nCols));
    NumericTablePtr x = HomogenNumericTable<float>::create(data.data(), nCols, nRows);
    std::cout << path << ": " << nRows << " x " << nCols << ", " << services::Environment::getInstance()->getNumberOfThreads() << " threads\n";

    dtrees::internal::FeatureTypes featTypes;
    REQUIRE(featTypes.init(*x));
    double legacy = 0;
    {
        dtrees::internal::BinParams prm(256, 5, decision_forest::training::quantiles);
        dtrees::internal::IndexedFeatures indexed;
        legacy = seconds([&]() { REQUIRE((indexed.init<float, DAAL_BASE_CPU>(*x, &featTypes, &prm)).ok()); });
    }
    std::cout << "legacy IndexedFeatures::init (base CpuType): " << legacy << " s\n";

    BinMatrix<DAAL_BASE_CPU> baseBins;
    std::cout << "new binner, base CpuType: " << timeNewBinner(*x, baseBins) << " s\n";
#if defined(TARGET_X86_64) && defined(DAAL_KERNEL_AVX512)
    if (services::Environment::getInstance()->getCpuId() >= int(daal::internal::avx512))
    {
        BinMatrix<daal::internal::avx512> avxBins;
        timeNewBinner(*x, avxBins); // first run touches the allocator and the arenas
        std::cout << "new binner, avx512: " << timeNewBinner(*x, avxBins) << " s\n";
    }
#endif

    size_t nChecked    = 0;
    const double err   = maxRankError(data, nRows, nCols, baseBins, nChecked);
    const double m     = double(binSampleRows);
    const double bound = std::sqrt(std::log(2 / 1e-3) / (2 * m)) + 1 / m;
    size_t nExact = 0, nU8 = 0;
    for (size_t f = 0; f < nCols; ++f) nExact += baseBins.exactBins(f);
    nU8 = baseBins.binBytes() == 1;
    std::cout << "sample rank error over " << nChecked << " features: " << err << " (DKW bound at delta 1e-3: " << bound << "); " << nExact
              << " exact-bin features; u8 bins: " << (nU8 ? "yes" : "no") << ", row stride " << baseBins.rowStride() << " B\n";
    REQUIRE(err <= bound);
}

} // namespace daal::algorithms::dtrees::engine::internal::test
