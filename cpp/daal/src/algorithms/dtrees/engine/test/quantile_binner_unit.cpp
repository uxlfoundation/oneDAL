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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "oneapi/dal/test/engine/common.hpp"
#include "data_management/data/csr_numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/env_detect.h"
#include "services/internal/daal_kernel_defines.h"
#include "src/algorithms/dtrees/engine/quantile_binner.h"
#include "src/externals/service_rng.h"

namespace daal::algorithms::dtrees::engine::internal::test
{
using namespace daal::data_management;
using daal::internal::CpuType;

constexpr auto baseCpu = DAAL_BASE_CPU;
const double nan       = std::numeric_limits<double>::quiet_NaN();
const double inf       = std::numeric_limits<double>::infinity();

// CpuTypes both built into the library (DAAL_KERNEL_<ISA>) and supported by this CPU.
template <typename F>
void forEachCpu(const F & f)
{
    f(std::integral_constant<CpuType, DAAL_BASE_CPU>());
#if defined(TARGET_X86_64)
    const int detected = services::Environment::getInstance()->getCpuId();
    #if defined(DAAL_KERNEL_AVX2)
    if (detected >= int(daal::internal::avx2)) f(std::integral_constant<CpuType, daal::internal::avx2>());
    #endif
    #if defined(DAAL_KERNEL_AVX512)
    if (detected >= int(daal::internal::avx512)) f(std::integral_constant<CpuType, daal::internal::avx512>());
    #endif
    (void)detected;
#endif
}

/// Row-major table over data, which must outlive it.
NumericTablePtr makeTable(std::vector<double> & data, size_t nCols, const std::vector<size_t> & categorical = {})
{
    NumericTablePtr t = HomogenNumericTable<double>::create(data.data(), nCols, data.size() / nCols);
    for (size_t f : categorical) (*t->getDictionarySharedPtr())[f].featureType = features::DAAL_CATEGORICAL;
    return t;
}

size_t countBelow(const double * sorted, size_t n, double x)
{
    return size_t(std::lower_bound(sorted, sorted + n, x) - sorted);
}

template <CpuType cpu>
struct Built
{
    BinMatrix<cpu> bins;
    bool binned = false;
    services::Status status;
};

// Engines a fit can use; the sample takes its draws from a stream of the user's engine.
const int engines[] = { __DAAL_BRNG_PHILOX4X32X10, __DAAL_BRNG_MT2203, __DAAL_BRNG_MT19937, __DAAL_BRNG_MCG59 };

template <CpuType cpu>
void build(const NumericTable & x, const BinningParams & params, uint64_t seed, Built<cpu> & out, int brng = __DAAL_BRNG_PHILOX4X32X10)
{
    daal::internal::BaseRNGsInst<cpu> engine(unsigned(seed), brng);
    out.status = QuantileBinner<double, cpu>::build(x, params, engine.getState(), out.bins, out.binned);
}

/// The first n uniform draws of a fresh stream of brng.
std::vector<double> streamDraws(uint64_t seed, size_t n, int brng = __DAAL_BRNG_PHILOX4X32X10)
{
    daal::internal::BaseRNGsInst<baseCpu> engine(unsigned(seed), brng);
    daal::internal::RNGsInst<double, baseCpu> rng;
    std::vector<double> draws(n);
    REQUIRE(rng.uniform(n, draws.data(), engine.getState(), 0.0, 1.0) == 0);
    return draws;
}

template <CpuType cpu>
size_t binOf(const BinMatrix<cpu> & m, size_t row, size_t f)
{
    return m.binBytes() == 1 ? m.template row<uint8_t>(0, row)[f] : m.template row<uint16_t>(0, row)[f];
}

/// Every row's bin is the count of the feature's inner borders below its value; NaN is in the missing bin.
template <CpuType cpu>
void requireConsistent(const BinMatrix<cpu> & m, const std::vector<double> & data)
{
    const size_t p = m.nFeatures();
    for (size_t f = 0; f < p; ++f)
    {
        const double * b = m.borders(f);
        for (size_t j = 1; j < m.nBins(f); ++j) REQUIRE(b[j - 1] < b[j]);
        bool sawNaN = false;
        for (size_t r = 0; r < m.nRows(); ++r)
        {
            const double v = data[r * p + f];
            if (std::isnan(v))
            {
                sawNaN = true;
                REQUIRE(binOf(m, r, f) == m.missingBin(f));
                continue;
            }
            REQUIRE(binOf(m, r, f) == countBelow(b, m.nBins(f) - 1, v));
            if (m.exactBins(f)) REQUIRE(b[binOf(m, r, f)] == v);
        }
        REQUIRE(m.hasMissing(f) == sawNaN);
        if (m.nBins(f)) REQUIRE(b[m.nBins(f) - 1] == m.max(f));
    }
}

template <CpuType a, CpuType b>
void requireEqual(const BinMatrix<a> & x, const BinMatrix<b> & y)
{
    REQUIRE(x.nFeatures() == y.nFeatures());
    REQUIRE(x.binBytes() == y.binBytes());
    for (size_t f = 0; f < x.nFeatures(); ++f)
    {
        REQUIRE(x.nBins(f) == y.nBins(f));
        REQUIRE(x.hasMissing(f) == y.hasMissing(f));
        REQUIRE(x.exactBins(f) == y.exactBins(f));
        for (size_t j = 0; j < x.nBins(f); ++j) REQUIRE(x.borders(f)[j] == y.borders(f)[j]);
        for (size_t r = 0; r < x.nRows(); ++r) REQUIRE(binOf(x, r, f) == binOf(y, r, f));
    }
}

/// Rows the binner leaves out of its sample, by the sampler's own rule: row r's draw is the stream's r-th.
std::vector<size_t> rowsOutsideSample(uint64_t seed, size_t nRows, size_t sampleRows)
{
    const double rate               = double(sampleRows) / double(nRows);
    const std::vector<double> draws = streamDraws(seed, nRows);
    std::vector<size_t> out;
    for (size_t r = 0; r < nRows; ++r)
        if (!(draws[r] < rate)) out.push_back(r);
    return out;
}

/// Whether this RNG backend can draw uniform doubles from the engine, which sampled binning needs.
bool canSample(int brng = __DAAL_BRNG_PHILOX4X32X10)
{
    daal::internal::BaseRNGsInst<baseCpu> engine(1, brng);
    daal::internal::RNGsInst<double, baseCpu> rng;
    double v = 0;
    return engine.getState() && rng.uniform(1, &v, engine.getState(), 0.0, 1.0) == 0;
}

TEST("binRowStride keeps rows inside cache lines", "[unit]")
{
    REQUIRE(binRowStride<baseCpu>(1, 1) == 1);
    REQUIRE(binRowStride<baseCpu>(3, 1) == 4);
    REQUIRE(binRowStride<baseCpu>(28, 1) == 32);
    REQUIRE(binRowStride<baseCpu>(28, 2) == 64);
    REQUIRE(binRowStride<baseCpu>(33, 1) == 64);
    REQUIRE(binRowStride<baseCpu>(64, 1) == 64);
    REQUIRE(binRowStride<baseCpu>(65, 1) == 128);
    REQUIRE(binRowStride<baseCpu>(784, 1) == 832);
}

TEST("searchBins on every CpuType counts the borders below x", "[unit]")
{
    std::mt19937_64 gen(777);
    std::normal_distribution<double> normal;
    for (size_t nInner : { 0, 1, 15, 16, 17, 100, 255, 256, 1000 })
    {
        std::vector<double> inner(nInner);
        for (auto & v : inner) v = normal(gen);
        std::sort(inner.begin(), inner.end());
        inner.erase(std::unique(inner.begin(), inner.end()), inner.end());
        std::vector<double> x = { -inf, inf, -1e300, 1e300 };
        for (double v : inner) x.insert(x.end(), { v, std::nextafter(v, -inf), std::nextafter(v, inf) });
        for (size_t i = 0; i < 1000; ++i) x.push_back(normal(gen) * 2);
        std::vector<uint32_t> expected(x.size());
        for (size_t i = 0; i < x.size(); ++i) expected[i] = uint32_t(countBelow(inner.data(), inner.size(), x[i]));
        forEachCpu([&](auto cpu) {
            std::vector<uint32_t> bins(x.size());
            QuantileBinner<double, decltype(cpu)::value>::searchBins(inner.data(), inner.size(), x.data(), x.size(), 1, bins.data());
            REQUIRE(bins == expected);
        });
    }
}

TEST("computeBorders gives one bin per value up to maxDataBins distinct values", "[unit]")
{
    const std::vector<double> s = { 1, 1, 2, 5, 5, 5, 9 };
    BinningParams params;
    services::internal::TArray<double, baseCpu> borders;
    size_t nBins = 0;
    bool exact   = false;
    REQUIRE(QuantileBinner<double, baseCpu>::computeBorders(s.data(), s.size(), 4, s.size(), params, true, borders, nBins, exact).ok());
    REQUIRE(exact);
    REQUIRE(nBins == 4);
    REQUIRE(std::vector<double>(borders.get(), borders.get() + nBins) == std::vector<double> { 1, 2, 5, 9 });

    // One value fewer allowed: quantile borders, which are sample values.
    REQUIRE(QuantileBinner<double, baseCpu>::computeBorders(s.data(), s.size(), 3, s.size(), params, true, borders, nBins, exact).ok());
    REQUIRE(!exact);
}

TEST("computeBorders takes borders at sample ranks j / B and merges small bins", "[unit]")
{
    std::vector<double> s(1000);
    for (size_t i = 0; i < s.size(); ++i) s[i] = double(i);
    BinningParams params;
    params.minBinSize = 1;
    services::internal::TArray<double, baseCpu> borders;
    size_t nBins = 0;
    bool exact   = false;
    REQUIRE(QuantileBinner<double, baseCpu>::computeBorders(s.data(), s.size(), 10, s.size(), params, true, borders, nBins, exact).ok());
    REQUIRE(nBins == 10);
    for (size_t j = 1; j < 10; ++j) REQUIRE(borders[j - 1] == double(j * 100 - 1));
    REQUIRE(borders[9] == 999);

    // A value holding half the sample gets a single bin.
    std::vector<double> heavy(1000, 3.0);
    for (size_t i = 0; i < 500; ++i) heavy[i] = double(i) / 1000;
    std::sort(heavy.begin(), heavy.end());
    REQUIRE(QuantileBinner<double, baseCpu>::computeBorders(heavy.data(), heavy.size(), 10, heavy.size(), params, true, borders, nBins, exact).ok());
    REQUIRE(nBins == 6);
    REQUIRE(borders[nBins - 1] == 3.0);
    REQUIRE(borders[nBins - 2] < 3.0);

    // Bins estimated below minBinSize rows merge: 100 sampled rows per bin stand for 100 rows each here.
    params.minBinSize = 150;
    REQUIRE(QuantileBinner<double, baseCpu>::computeBorders(s.data(), s.size(), 10, s.size(), params, true, borders, nBins, exact).ok());
    REQUIRE(nBins == 5);
    REQUIRE(borders[0] == 199);

    // Averages: equal-width borders over the sample's range.
    params.minBinSize = 1;
    params.strategy   = decision_forest::training::averages;
    REQUIRE(QuantileBinner<double, baseCpu>::computeBorders(s.data(), s.size(), 10, s.size(), params, true, borders, nBins, exact).ok());
    REQUIRE(nBins == 10);
    for (size_t j = 1; j < 10; ++j) REQUIRE(std::abs(borders[j - 1] - 999.0 * double(j) / 10) < 1e-9);
}

struct MixedTable
{
    static constexpr size_t p = 6;
    std::vector<double> data;
    MixedTable(size_t nRows, uint64_t seed)
    {
        std::mt19937_64 gen(seed);
        std::normal_distribution<double> normal;
        std::uniform_int_distribution<int> digit(0, 9);
        const double codes[] = { 0, 1, 100 };
        data.resize(nRows * p);
        for (size_t r = 0; r < nRows; ++r)
        {
            double * row = data.data() + r * p;
            row[0]       = normal(gen);                     // continuous
            row[1]       = double(digit(gen));              // low cardinality
            row[2]       = r % 50 == 7 ? nan : normal(gen); // continuous with NaN
            row[3]       = 4.25;                            // constant
            row[4]       = codes[r % 3];                    // categorical, non-contiguous codes
            row[5]       = double(r % 256);                 // 256 values, as a pixel
        }
    }
};

TEST("an exact-mode build bins every row by its borders, the same on every CpuType", "[unit]")
{
    MixedTable t(20000, 1);
    NumericTablePtr x = makeTable(t.data, MixedTable::p, { 4 });
    BinningParams params;
    params.minBinSize = 1;
    Built<baseCpu> reference;
    build(*x, params, 777, reference);
    REQUIRE(reference.status.ok());
    REQUIRE(reference.binned);
    const auto & m = reference.bins;
    requireConsistent(m, t.data);

    REQUIRE(m.binBytes() == 1);
    REQUIRE(m.rowStride() == 8);
    REQUIRE(!m.exactBins(0));
    REQUIRE(m.nBins(0) == params.maxBins);
    REQUIRE(m.exactBins(1));
    REQUIRE(m.nBins(1) == 10);
    REQUIRE(m.hasMissing(2));
    REQUIRE(m.nBins(2) == params.maxBins - 1);
    REQUIRE(m.isConstant(3));
    REQUIRE(m.exactBins(4));
    REQUIRE(m.nBins(4) == 3);
    REQUIRE(m.borders(4)[2] == 100);
    REQUIRE(m.exactBins(5));
    REQUIRE(m.nBins(5) == 256);

    // Exact quantiles: below binSampleRows the sample is the whole column.
    std::vector<double> col(20000);
    for (size_t r = 0; r < col.size(); ++r) col[r] = t.data[r * MixedTable::p];
    std::sort(col.begin(), col.end());
    for (size_t j = 1; j < params.maxBins; ++j)
    {
        const size_t idx = (j * col.size() + params.maxBins - 1) / params.maxBins - 1;
        REQUIRE(m.borders(0)[j - 1] == col[idx]);
    }

    forEachCpu([&](auto cpu) {
        Built<decltype(cpu)::value> other;
        build(*x, params, 777, other);
        REQUIRE(other.status.ok());
        requireEqual(m, other.bins);
    });
}

TEST("the binning sample and the bins do not depend on the thread count, for every engine", "[unit]")
{
    MixedTable t(60000, 2);
    NumericTablePtr x = makeTable(t.data, MixedTable::p, { 4 });
    BinningParams params;
    params.sampleRows = 5000;
    for (int brng : engines)
    {
        if (!canSample(brng)) continue;
        const size_t saved = services::Environment::getInstance()->getNumberOfThreads();
        services::Environment::getInstance()->setNumberOfThreads(1);
        Built<baseCpu> one;
        build(*x, params, 99, one, brng);
        services::Environment::getInstance()->setNumberOfThreads(saved);
        Built<baseCpu> many;
        build(*x, params, 99, many, brng);
        REQUIRE(one.status.ok());
        REQUIRE(many.status.ok());
        requireConsistent(many.bins, t.data);
        requireEqual(one.bins, many.bins);
    }
}

TEST("binning groups hold at most one feature per thread and fit the scratch budget", "[unit]")
{
    // MNIST: 60k sample rows of doubles (20 bytes a value) on 224 threads; the budget alone would take 894.
    REQUIRE(binGroupFeatures(60000, 20, 224) == 224);
    // 4M sample rows: the budget holds 13 features.
    REQUIRE(binGroupFeatures(size_t(1) << 22, 20, 224) == binSampleScratchBytes / ((size_t(1) << 22) * 20));
    REQUIRE(binGroupFeatures(1000, 20, 1) == 1);
    REQUIRE(binGroupFeatures(size_t(1) << 30, 20, 8) == 1); // a feature above the budget is its own group
}

TEST("a wide table bins the same whatever the feature groups", "[unit]")
{
    // 42 features in groups of 1, 4, 16 or all, set by the thread count.
    constexpr size_t copies = 7, p = MixedTable::p * copies, nRows = 3000;
    MixedTable t(nRows, 3);
    std::vector<double> wide(nRows * p);
    for (size_t r = 0; r < nRows; ++r)
        for (size_t f = 0; f < p; ++f) wide[r * p + f] = t.data[r * MixedTable::p + f % MixedTable::p];
    std::vector<size_t> categorical;
    for (size_t c = 0; c < copies; ++c) categorical.push_back(c * MixedTable::p + 4);
    NumericTablePtr x = makeTable(wide, p, categorical);
    BinningParams params;
    Built<baseCpu> all;
    build(*x, params, 5, all);
    REQUIRE(all.status.ok());
    requireConsistent(all.bins, wide);
    const size_t saved = services::Environment::getInstance()->getNumberOfThreads();
    for (size_t nThreads : { 1, 4, 16 })
    {
        CAPTURE(nThreads);
        services::Environment::getInstance()->setNumberOfThreads(nThreads);
        Built<baseCpu> grouped;
        build(*x, params, 5, grouped);
        services::Environment::getInstance()->setNumberOfThreads(saved);
        REQUIRE(grouped.status.ok());
        requireEqual(all.bins, grouped.bins);
    }
}

TEST("sampling advances the stream by one draw per row, and exact mode does not read it", "[unit]")
{
    std::vector<double> data(20000);
    for (size_t r = 0; r < data.size(); ++r) data[r] = double(r % 977);
    NumericTablePtr x = makeTable(data, 1);
    for (int brng : engines)
    {
        if (!canSample(brng)) continue;
        const std::vector<double> reference = streamDraws(5, data.size() + 1, brng);
        for (size_t sampleRows : { size_t(2000), data.size() })
        {
            BinningParams params;
            params.sampleRows = sampleRows;
            daal::internal::BaseRNGsInst<baseCpu> engine(5, brng);
            BinMatrix<baseCpu> bins;
            bool binned = false;
            REQUIRE(QuantileBinner<double, baseCpu>::build(*x, params, engine.getState(), bins, binned).ok());
            REQUIRE(binned);
            daal::internal::RNGsInst<double, baseCpu> rng;
            double next = 0;
            REQUIRE(rng.uniform(1, &next, engine.getState(), 0.0, 1.0) == 0);
            REQUIRE(next == reference[sampleRows < data.size() ? data.size() : 0]);
        }
    }
}

TEST("a NaN outside the sample rebins a feature that uses all maxBins data bins", "[unit]")
{
    SKIP_IF(!canSample());
    constexpr size_t nRows = 40000, sampleRows = 4000;
    constexpr uint64_t seed = 5;
    std::mt19937_64 gen(3);
    std::normal_distribution<double> normal;
    std::vector<double> data(nRows);
    for (auto & v : data) v = normal(gen);
    const size_t hidden = rowsOutsideSample(seed, nRows, sampleRows)[10];
    data[hidden]        = nan;
    NumericTablePtr x   = makeTable(data, 1);
    BinningParams params;
    params.sampleRows = sampleRows;
    params.minBinSize = 1;
    Built<baseCpu> b;
    build(*x, params, seed, b);
    REQUIRE(b.status.ok());
    REQUIRE(b.bins.hasMissing(0));
    REQUIRE(b.bins.nBins(0) == params.maxBins - 1);
    requireConsistent(b.bins, data);
}

TEST("a value outside the sample's distinct set rebins a low-cardinality feature with quantile borders", "[unit]")
{
    SKIP_IF(!canSample());
    constexpr size_t nRows = 40000, sampleRows = 4000;
    constexpr uint64_t seed = 6;
    std::vector<double> data(nRows);
    for (size_t r = 0; r < nRows; ++r) data[r] = double(r % 5);
    data[rowsOutsideSample(seed, nRows, sampleRows)[3]] = 7.5;
    NumericTablePtr x                                   = makeTable(data, 1);
    BinningParams params;
    params.sampleRows = sampleRows;
    Built<baseCpu> b;
    build(*x, params, seed, b);
    REQUIRE(b.status.ok());
    REQUIRE(!b.bins.exactBins(0));
    REQUIRE(b.bins.max(0) == 7.5);
    requireConsistent(b.bins, data);
}

TEST("a category outside the sample is added from the full column", "[unit]")
{
    SKIP_IF(!canSample());
    constexpr size_t nRows = 40000, sampleRows = 4000;
    constexpr uint64_t seed = 7;
    std::vector<double> data(nRows);
    for (size_t r = 0; r < nRows; ++r) data[r] = double(r % 3);
    data[rowsOutsideSample(seed, nRows, sampleRows)[0]] = 50;
    NumericTablePtr x                                   = makeTable(data, 1, { 0 });
    BinningParams params;
    params.sampleRows = sampleRows;
    Built<baseCpu> b;
    build(*x, params, seed, b);
    REQUIRE(b.status.ok());
    REQUIRE(b.bins.exactBins(0));
    REQUIRE(b.bins.nBins(0) == 4);
    requireConsistent(b.bins, data);
}

TEST("sampled borders are within the DKW rank bound of the full column", "[unit]")
{
    SKIP_IF(!canSample());
    // eps = sqrt(ln(2 / delta) / (2 m)) bounds every quantile's rank error with probability 1 - delta.
    constexpr size_t nRows = size_t(1) << 18, sampleRows = size_t(1) << 14;
    constexpr double delta = 1e-3;
    const double eps       = std::sqrt(std::log(2 / delta) / (2 * double(sampleRows)));
    std::mt19937_64 gen(11);
    std::lognormal_distribution<double> skewed;
    std::vector<double> data(nRows);
    for (auto & v : data) v = skewed(gen);
    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    NumericTablePtr x = makeTable(data, 1);
    BinningParams params;
    params.sampleRows = sampleRows;
    params.minBinSize = 1;
    for (int brng : engines)
    {
        if (!canSample(brng)) continue;
        for (uint64_t seed = 1; seed <= 5; ++seed)
        {
            Built<baseCpu> b;
            build(*x, params, seed, b, brng);
            REQUIRE(b.status.ok());
            const size_t B = b.bins.nBins(0);
            REQUIRE(B == params.maxBins);
            for (size_t j = 1; j < B; ++j)
            {
                const double rank = double(std::upper_bound(sorted.begin(), sorted.end(), b.bins.borders(0)[j - 1]) - sorted.begin()) / double(nRows);
                REQUIRE(std::abs(rank - double(j) / double(B)) <= eps + 1.0 / double(sampleRows));
            }
        }
    }
}

TEST("u16 indices when a feature needs more than 256 bins", "[unit]")
{
    std::vector<double> data(5000);
    for (size_t r = 0; r < data.size(); ++r) data[r] = double(r);
    NumericTablePtr x = makeTable(data, 1);
    BinningParams params;
    params.maxBins    = 1000;
    params.minBinSize = 1;
    Built<baseCpu> b;
    build(*x, params, 1, b);
    REQUIRE(b.status.ok());
    REQUIRE(b.bins.binBytes() == 2);
    REQUIRE(b.bins.nBins(0) == 1000);
    requireConsistent(b.bins, data);
}

TEST("fits that cannot be binned are reported, not binned", "[unit]")
{
    std::vector<double> data(70000);
    for (size_t r = 0; r < data.size(); ++r) data[r] = double(r);
    BinningParams params;
    Built<baseCpu> b;

    params.maxBins = maxU16Bins + 1;
    build(*makeTable(data, 1), params, 1, b);
    REQUIRE(b.status.ok());
    REQUIRE(!b.binned);

    params.maxBins = 0;
    build(*makeTable(data, 1), params, 1, b);
    REQUIRE(b.status.ok());
    REQUIRE(!b.binned);

    params.maxBins = 256;
    build(*makeTable(data, 1, { 0 }), params, 1, b);
    REQUIRE(b.status.ok());
    REQUIRE(!b.binned);
}

TEST("invalid categorical codes and CSR tables are rejected", "[unit]")
{
    BinningParams params;
    for (double bad : { 1.5, -1.0, 16777216.0 })
    {
        std::vector<double> data = { 0, 1, 2, bad };
        Built<baseCpu> b;
        build(*makeTable(data, 1, { 0 }), params, 1, b);
        REQUIRE(!b.status.ok());
    }

    double values[]        = { 1, 2 };
    size_t colIndices[]    = { 1, 1 };
    size_t rowOffsets[]    = { 1, 2, 3 };
    CSRNumericTablePtr csr = CSRNumericTable::create(values, colIndices, rowOffsets, 1, 2);
    Built<baseCpu> b;
    build(*csr, params, 1, b);
    REQUIRE(!b.status.ok());
}

} // namespace daal::algorithms::dtrees::engine::internal::test
