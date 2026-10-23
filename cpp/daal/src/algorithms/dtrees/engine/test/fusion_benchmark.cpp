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

/// Random-forest training time with and without TreeParams::fusePartition and carryConstants, on a
/// float32 .npy classification table (Higgs by default). Bins once, then trains the configurations
/// interleaved and checks they grow the same trees. Run explicitly with the "[.benchmark]" tag.
/// Environment: FUSION_BENCHMARK_X / _Y (.npy paths), _ROWS (leading-row subset), _TREES, _REPEATS,
/// _DEPTH, _THREADS, _CONFIGS (letters from ABCD; D is the exact phase everywhere).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include "oneapi/dal/test/engine/common.hpp"
#include "data_management/data/homogen_numeric_table.h"
#include "services/env_detect.h"
#include "services/internal/daal_kernel_defines.h"
#include "src/algorithms/dtrees/engine/node_rng.h"
#include "src/algorithms/dtrees/engine/quantile_binner.h"
#include "src/algorithms/dtrees/engine/tree_builder.h"

namespace daal::algorithms::dtrees::engine::internal::test
{
using namespace daal::data_management;
using daal::internal::CpuType;
using dtrees::internal::DecisionTreeNode;

constexpr const char * defaultX = "/nfs/site/disks/dal_datasets/DAAL_datasets/higgs_10500K_x_train.npy";
constexpr const char * defaultY = "/nfs/site/disks/dal_datasets/DAAL_datasets/higgs_10500K_y_train.npy";

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
    // A 1-D array, as np.save writes a label vector, is one column.
    nCols = 1;
    if (open == std::string::npos || std::sscanf(header.c_str() + open, "(%zu, %zu)", &nRows, &nCols) < 1)
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

size_t envSize(const char * name, size_t fallback)
{
    const char * v = std::getenv(name);
    return v ? size_t(std::strtoull(v, nullptr, 10)) : fallback;
}

template <CpuType cpu>
class Inputs : public TreeInputs<ClsCountRecord, cpu>
{
public:
    Inputs(const ResponseSource<double, cpu> & src, const std::vector<std::vector<uint16_t> > & counts, uint64_t seed, size_t nTrees)
        : trees(nTrees), _src(src), _counts(counts), _seed(seed)
    {}

    services::Status prepareTree(size_t iTree, int numaNode, ResponseSet<ClsCountRecord, cpu> & records, uint64_t & treeSeed) override
    {
        double total = 0;
        treeSeed     = mix64<cpu>(_seed + iTree);
        return records.fill(_counts[iTree].data(), _src, numaNode, total);
    }

    services::Status emitTree(size_t iTree, EmittedTree & tree) override
    {
        std::lock_guard<std::mutex> lock(_mutex);
        trees[iTree] = std::move(tree);
        return services::Status();
    }

    std::vector<EmittedTree> trees;

private:
    const ResponseSource<double, cpu> & _src;
    const std::vector<std::vector<uint16_t> > & _counts;
    uint64_t _seed;
    std::mutex _mutex;
};

const DecisionTreeNode * rowsOf(const EmittedTree & t)
{
    return static_cast<const DecisionTreeNode *>(t.nodes->getArray());
}

bool sameTables(const EmittedTree & a, const EmittedTree & b)
{
    const size_t n = a.nodes->getNumberOfRows();
    if (n != b.nodes->getNumberOfRows()) return false;
    const DecisionTreeNode * ra = rowsOf(a);
    const DecisionTreeNode * rb = rowsOf(b);
    for (size_t i = 0; i < n; ++i)
    {
        if (ra[i].featureIndex != rb[i].featureIndex || ra[i].leftIndexOrClass != rb[i].leftIndexOrClass
            || ra[i].featureValueOrResponse != rb[i].featureValueOrResponse || a.impurity->getArray()[i] != b.impurity->getArray()[i]
            || a.nodeSamples->getArray()[i] != b.nodeSamples->getArray()[i])
            return false;
    }
    return true;
}

size_t depthOf(const EmittedTree & t, size_t r = 0)
{
    const DecisionTreeNode * rows = rowsOf(t);
    if (rows[r].featureIndex < 0) return 0;
    return 1 + std::max(depthOf(t, rows[r].leftIndexOrClass), depthOf(t, rows[r].leftIndexOrClass + 1));
}

template <CpuType cpu>
void run(const NumericTable & xt, const NumericTable & yt, size_t nRows, size_t nCols)
{
    const size_t nTrees       = envSize("FUSION_BENCHMARK_TREES", 32);
    const size_t repeats      = envSize("FUSION_BENCHMARK_REPEATS", 3);
    const char * cfgEnv       = std::getenv("FUSION_BENCHMARK_CONFIGS");
    const std::string configs = cfgEnv ? cfgEnv : "ABC";
    constexpr uint64_t seed   = 777;

    NumaReplicated<double, cpu> x;
    ResponseSource<double, cpu> src;
    BinMatrix<cpu> bins;
    bool binned = false;
    BinningParams binning;
    binning.placement  = NumaPlacement::host(true);
    const double tCopy = seconds([&]() {
        REQUIRE((copyTableRows<double, cpu>(xt, NumaPlacement::host(true), x)).ok());
        REQUIRE(src.build(yt, nullptr, true, NumaPlacement::host(true)).ok());
    });
    daal::internal::BaseRNGsInst<cpu> engine(unsigned(seed), __DAAL_BRNG_PHILOX4X32X10);
    const double tBin = seconds([&]() { REQUIRE((QuantileBinner<double, cpu>::build(xt, binning, engine.getState(), bins, binned)).ok()); });
    REQUIRE(binned);
    std::cout << "copy x/y: " << tCopy << " s, binning: " << tBin << " s (u8 bins: " << (bins.binBytes() == 1 ? "yes" : "no") << ")\n";

    std::vector<std::vector<uint16_t> > counts(nTrees);
    daal::threader_for(nTrees, 1, [&](size_t t) {
        std::mt19937_64 gen(seed * 1000003 + t);
        std::uniform_int_distribution<size_t> row(0, nRows - 1);
        counts[t].assign(nRows, 0);
        for (size_t i = 0; i < nRows; ++i) ++counts[t][row(gen)];
    });

    TreeData<double, cpu> data;
    data.x         = &x;
    data.responses = &src;
    data.bins      = &bins;
    data.nRows     = nRows;
    data.nFeatures = nCols;

    TreeParams base;
    base.nClasses        = 2;
    base.featuresPerNode = 5;
    base.maxDepth        = envSize("FUSION_BENCHMARK_DEPTH", 0);
    std::cout << "trees " << nTrees << ", depth " << base.maxDepth << ", threads " << services::Environment::getInstance()->getNumberOfThreads()
              << ", configs " << configs << " x " << repeats << "\n";

    {
        // Untimed: the first fit touches the allocator and the arenas.
        Inputs<cpu> inputs(src, counts, seed, nTrees);
        std::cout << "warm-up: " << seconds([&]() { REQUIRE((trainTrees<double, ClsCountRecord, cpu>(data, base, nTrees, inputs)).ok()); }) << " s\n";
    }
    std::vector<std::vector<double> > times(configs.size());
    std::vector<std::vector<EmittedTree> > first(configs.size());
    for (size_t rep = 0; rep < repeats; ++rep)
    {
        for (size_t c = 0; c < configs.size(); ++c)
        {
            TreeParams params     = base;
            params.fusePartition  = configs[c] != 'A';
            params.carryConstants = configs[c] == 'C' || configs[c] == 'D';
            if (configs[c] == 'D') params.denseThreshold = size_t(1) << 40;
            Inputs<cpu> inputs(src, counts, seed, nTrees);
            const double t = seconds([&]() { REQUIRE((trainTrees<double, ClsCountRecord, cpu>(data, params, nTrees, inputs)).ok()); });
            times[c].push_back(t);
            std::cout << "rep " << rep << " config " << configs[c] << ": " << t << " s" << std::endl;
            if (rep == 0) first[c] = std::move(inputs.trees);
        }
    }

    size_t nodes = 0, depth = 0;
    for (const auto & t : first[0])
    {
        nodes += t.nodes->getNumberOfRows();
        depth = std::max(depth, depthOf(t));
    }
    std::cout << "config " << configs[0] << " trees: " << nodes << " nodes total, max depth " << depth << "\n";
    for (size_t c = 0; c < configs.size(); ++c)
    {
        std::vector<double> s = times[c];
        std::sort(s.begin(), s.end());
        size_t same = 0;
        for (size_t t = 0; t < nTrees; ++t) same += sameTables(first[0][t], first[c][t]);
        std::cout << "config " << configs[c] << ": min " << s.front() << " s, median " << s[s.size() / 2] << " s; trees identical to " << configs[0]
                  << ": " << same << "/" << nTrees << "\n";
    }
}

TEST("fusion benchmark", "[.benchmark]")
{
    const char * xEnv = std::getenv("FUSION_BENCHMARK_X");
    const char * yEnv = std::getenv("FUSION_BENCHMARK_Y");
    std::vector<float> xf, yf;
    size_t nRows = 0, nCols = 0, yRows = 0, yCols = 0;
    SKIP_IF(!loadNpy(xEnv ? xEnv : defaultX, xf, nRows, nCols));
    SKIP_IF(!loadNpy(yEnv ? yEnv : defaultY, yf, yRows, yCols));
    REQUIRE(yRows == nRows);
    nRows                 = std::min(nRows, envSize("FUSION_BENCHMARK_ROWS", nRows));
    const size_t nThreads = envSize("FUSION_BENCHMARK_THREADS", 0);
    if (nThreads) services::Environment::getInstance()->setNumberOfThreads(nThreads);

    std::vector<double> xd(nRows * nCols), yd(nRows);
    daal::threader_for(nRows, 4096, [&](size_t i) {
        for (size_t f = 0; f < nCols; ++f) xd[i * nCols + f] = xf[i * nCols + f];
        yd[i] = yf[i * yCols];
    });
    std::vector<float>().swap(xf);
    NumericTablePtr xt = HomogenNumericTable<double>::create(xd.data(), nCols, nRows);
    NumericTablePtr yt = HomogenNumericTable<double>::create(yd.data(), 1, nRows);
    std::cout << nRows << " x " << nCols << "\n";

#if defined(TARGET_X86_64) && defined(DAAL_KERNEL_AVX512)
    if (services::Environment::getInstance()->getCpuId() >= int(daal::internal::avx512))
    {
        std::cout << "cpu: avx512\n";
        run<daal::internal::avx512>(*xt, *yt, nRows, nCols);
        return;
    }
#endif
    std::cout << "cpu: base\n";
    run<DAAL_BASE_CPU>(*xt, *yt, nRows, nCols);
}

} // namespace daal::algorithms::dtrees::engine::internal::test
