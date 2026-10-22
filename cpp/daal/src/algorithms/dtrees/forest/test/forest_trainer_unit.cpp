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

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "oneapi/dal/test/engine/common.hpp"
#include "algorithms/decision_forest/decision_forest_classification_training_types.h"
#include "algorithms/decision_forest/decision_forest_regression_training_types.h"
#include "algorithms/engines/mcg59/mcg59.h"
#include "algorithms/engines/mrg32k3a/mrg32k3a.h"
#include "algorithms/engines/mt19937/mt19937.h"
#include "algorithms/engines/mt2203/mt2203.h"
#include "algorithms/engines/philox4x32x10/philox4x32x10.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/env_detect.h"
#include "src/algorithms/dtrees/forest/classification/df_classification_model_impl.h"
#include "src/algorithms/dtrees/forest/classification/df_classification_train_kernel.h"
#include "src/algorithms/dtrees/forest/regression/df_regression_model_impl.h"
#include "src/algorithms/dtrees/forest/regression/df_regression_train_kernel.h"

namespace daal::algorithms::decision_forest::test
{
using namespace daal::data_management;
using daal::internal::CpuType;
using dtrees::internal::DecisionTreeNode;
namespace cls = decision_forest::classification;
namespace reg = decision_forest::regression;

constexpr CpuType cpu = DAAL_BASE_CPU;
using int64           = DAAL_INT64;

class ThreadCountGuard
{
public:
    explicit ThreadCountGuard(size_t n) : _saved(services::Environment::getInstance()->getNumberOfThreads())
    {
        services::Environment::getInstance()->setNumberOfThreads(n);
    }
    ~ThreadCountGuard() { services::Environment::getInstance()->setNumberOfThreads(_saved); }

private:
    size_t _saved;
};

/// Row-major x, responses (class or value); nClasses 0 for regression.
struct Dataset
{
    size_t n = 0, p = 0, nClasses = 0;
    std::vector<double> x, y;
};

Dataset makeData(size_t n, size_t p, size_t nClasses, uint64_t seed)
{
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> normal;
    Dataset d;
    d.n        = n;
    d.p        = p;
    d.nClasses = nClasses;
    d.x.resize(n * p);
    d.y.resize(n);
    for (size_t i = 0; i < n; ++i)
    {
        double s = 0;
        for (size_t f = 0; f < p; ++f)
        {
            d.x[i * p + f] = normal(gen);
            s += (f < 4 ? double(4 - f) : 0.) * d.x[i * p + f];
        }
        // Learnable targets: classes are bands of a noisy linear score; the response is smooth plus noise.
        const double z = s / 4 + 0.3 * normal(gen);
        if (nClasses)
        {
            const double band = (z + 1.5) / 3 * double(nClasses);
            d.y[i]            = band < 0 ? 0 : double(size_t(band) < nClasses ? size_t(band) : nClasses - 1);
        }
        else
        {
            d.y[i] = z + 0.5 * std::sin(2 * d.x[i * p]);
        }
    }
    return d;
}

/// Engine hyperparameters as the oneAPI interface sets them, with the legacy kernels or the tree engine.
template <typename Hyperparameter>
/// serial: the serialSubtreeRows id, or -1 to leave it unset (the trainer's L2-based cutoff).
void setHyperparameters(Hyperparameter & hp, bool legacy, int64 serial)
{
    if constexpr (std::is_same<Hyperparameter, cls::training::internal::Hyperparameter>::value)
    {
        REQUIRE(hp.set(cls::training::internal::smallNClassesThreshold, int64(8)).ok());
        REQUIRE(hp.set(cls::training::internal::minPartCoefficient, int64(4)).ok());
        REQUIRE(hp.set(cls::training::internal::minSizeCoefficient, int64(24000)).ok());
        REQUIRE(hp.set(cls::training::internal::legacyTraining, int64(legacy)).ok());
        if (serial >= 0) REQUIRE(hp.set(cls::training::internal::serialSubtreeRows, serial).ok());
    }
    else
    {
        REQUIRE(hp.set(reg::training::internal::minPartCoefficient, int64(4)).ok());
        REQUIRE(hp.set(reg::training::internal::minSizeCoefficient, int64(24000)).ok());
        REQUIRE(hp.set(reg::training::internal::legacyTraining, int64(legacy)).ok());
        if (serial >= 0) REQUIRE(hp.set(reg::training::internal::serialSubtreeRows, serial).ok());
    }
}

/// A fit's model and result tables.
template <typename ResultPtr>
struct Fit
{
    ResultPtr result;
    NumericTablePtr x, y;
    const dtrees::internal::ModelImpl * model = nullptr;
};

template <cls::training::Method method>
Fit<cls::training::ResultPtr> fitCls(const Dataset & d, cls::training::Parameter par, bool legacy = false, int64 serial = -1)
{
    Fit<cls::training::ResultPtr> f;
    // As the Batch API sets it; Parameter::check rejects the base default of 0.
    if (!par.minObservationsInLeafNode) par.minObservationsInLeafNode = 1;
    f.x = HomogenNumericTable<double>::create(d.p, d.n, NumericTable::doAllocate);
    f.y = HomogenNumericTable<double>::create(1, d.n, NumericTable::doAllocate);
    std::copy(d.x.begin(), d.x.end(), static_cast<HomogenNumericTable<double> *>(f.x.get())->getArray());
    std::copy(d.y.begin(), d.y.end(), static_cast<HomogenNumericTable<double> *>(f.y.get())->getArray());
    classifier::training::Input input;
    input.set(classifier::training::data, f.x);
    input.set(classifier::training::labels, f.y);
    f.result.reset(new cls::training::Result());
    REQUIRE(f.result->allocate<double>(&input, &par, int(method)).ok());
    auto * model = static_cast<cls::internal::ModelImpl *>(f.result->get(classifier::training::model).get());
    model->clear();
    model->setNFeatures(d.p);
    cls::training::internal::Hyperparameter hp;
    setHyperparameters(hp, legacy, serial);
    cls::training::internal::ClassificationTrainBatchKernel<double, method, cpu> kernel;
    REQUIRE(kernel.compute(f.x.get(), f.y.get(), nullptr, *model, *f.result, par, &hp).ok());
    f.model = model;
    return f;
}

template <reg::training::Method method>
Fit<reg::training::ResultPtr> fitReg(const Dataset & d, reg::training::Parameter par, bool legacy = false, int64 serial = -1)
{
    Fit<reg::training::ResultPtr> f;
    if (!par.minObservationsInLeafNode) par.minObservationsInLeafNode = 5;
    f.x = HomogenNumericTable<double>::create(d.p, d.n, NumericTable::doAllocate);
    f.y = HomogenNumericTable<double>::create(1, d.n, NumericTable::doAllocate);
    std::copy(d.x.begin(), d.x.end(), static_cast<HomogenNumericTable<double> *>(f.x.get())->getArray());
    std::copy(d.y.begin(), d.y.end(), static_cast<HomogenNumericTable<double> *>(f.y.get())->getArray());
    reg::training::Input input;
    input.set(reg::training::data, f.x);
    input.set(reg::training::dependentVariable, f.y);
    f.result.reset(new reg::training::Result());
    REQUIRE(f.result->allocate<double>(&input, &par, int(method)).ok());
    auto * model = static_cast<reg::internal::ModelImpl *>(f.result->get(reg::training::model).get());
    model->clear();
    reg::training::internal::Hyperparameter hp;
    setHyperparameters(hp, legacy, serial);
    reg::training::internal::RegressionTrainBatchKernel<double, method, cpu> kernel;
    REQUIRE(kernel.compute(f.x.get(), f.y.get(), nullptr, *model, *f.result, par, &hp).ok());
    f.model = model;
    return f;
}

const DecisionTreeNode * nodesOf(const dtrees::internal::ModelImpl & m, size_t t)
{
    return static_cast<const DecisionTreeNode *>(m.at(t)->getArray());
}

void requireSameModels(const dtrees::internal::ModelImpl & a, const dtrees::internal::ModelImpl & b)
{
    REQUIRE(a.size() == b.size());
    for (size_t t = 0; t < a.size(); ++t)
    {
        const size_t n = a.at(t)->getNumberOfRows();
        REQUIRE(n == b.at(t)->getNumberOfRows());
        const DecisionTreeNode * na = nodesOf(a, t);
        const DecisionTreeNode * nb = nodesOf(b, t);
        for (size_t i = 0; i < n; ++i)
        {
            REQUIRE(na[i].featureIndex == nb[i].featureIndex);
            REQUIRE(na[i].leftIndexOrClass == nb[i].leftIndexOrClass);
            REQUIRE(na[i].featureValueOrResponse == nb[i].featureValueOrResponse);
            REQUIRE(a.getImpVals(t)[i] == b.getImpVals(t)[i]);
            REQUIRE(a.getNodeSampleCount(t)[i] == b.getNodeSampleCount(t)[i]);
        }
    }
}

/// The leaf of tree t that row x reaches, routing as the prediction kernels do (ordered features).
size_t leafOf(const DecisionTreeNode * nodes, const double * x)
{
    size_t r = 0;
    while (nodes[r].featureIndex >= 0) r = nodes[r].leftIndexOrClass + (x[nodes[r].featureIndex] > nodes[r].featureValueOrResponse ? 1 : 0);
    return r;
}

/// Majority vote accuracy (classification) or R^2 (regression) of the forest on d.
double score(const dtrees::internal::ModelImpl & m, const Dataset & d)
{
    double good = 0, sse = 0, sst = 0, mean = 0;
    for (size_t i = 0; i < d.n; ++i) mean += d.y[i] / double(d.n);
    for (size_t i = 0; i < d.n; ++i)
    {
        const double * x = d.x.data() + i * d.p;
        if (d.nClasses)
        {
            std::vector<size_t> votes(d.nClasses, 0);
            for (size_t t = 0; t < m.size(); ++t) ++votes[nodesOf(m, t)[leafOf(nodesOf(m, t), x)].leftIndexOrClass];
            size_t best = 0;
            for (size_t c = 1; c < d.nClasses; ++c) best = votes[c] > votes[best] ? c : best;
            good += best == size_t(d.y[i]);
        }
        else
        {
            double sum = 0;
            for (size_t t = 0; t < m.size(); ++t) sum += nodesOf(m, t)[leafOf(nodesOf(m, t), x)].featureValueOrResponse;
            const double e = sum / double(m.size()) - d.y[i];
            sse += e * e;
            sst += (d.y[i] - mean) * (d.y[i] - mean);
        }
    }
    return d.nClasses ? good / double(d.n) : 1 - sse / sst;
}

TEST("philox forests are identical across thread counts and memory saving modes", "[unit]")
{
    const Dataset c = makeData(4000, 8, 3, 1);
    const Dataset r = makeData(4000, 8, 0, 2);
    for (bool memSave : { false, true })
    {
        cls::training::Parameter pc(3);
        pc.nTrees           = 6;
        pc.memorySavingMode = memSave;
        reg::training::Parameter pr;
        pr.nTrees           = 6;
        pr.memorySavingMode = memSave;
        std::unique_ptr<Fit<cls::training::ResultPtr> > c1;
        std::unique_ptr<Fit<reg::training::ResultPtr> > r1;
        {
            ThreadCountGuard one(1);
            c1.reset(new Fit<cls::training::ResultPtr>(fitCls<cls::training::hist>(c, pc)));
            r1.reset(new Fit<reg::training::ResultPtr>(fitReg<reg::training::hist>(r, pr)));
        }
        const auto cn = fitCls<cls::training::hist>(c, pc);
        const auto rn = fitReg<reg::training::hist>(r, pr);
        requireSameModels(*c1->model, *cn.model);
        requireSameModels(*r1->model, *rn.model);
    }
}

/// MDI of the documentation from the model's own tables: per tree, the sum over splits on f of
/// (n_t imp_t - n_l imp_l - n_r imp_r) / n_root, averaged over trees.
std::vector<double> mdiFromTables(const dtrees::internal::ModelImpl & m, size_t p)
{
    std::vector<double> mdi(p, 0);
    for (size_t t = 0; t < m.size(); ++t)
    {
        const DecisionTreeNode * nodes = nodesOf(m, t);
        const double * imp             = m.getImpVals(t);
        const int * cnt                = m.getNodeSampleCount(t);
        for (size_t i = 0; i < m.at(t)->getNumberOfRows(); ++i)
        {
            if (nodes[i].featureIndex < 0) continue;
            const size_t l = nodes[i].leftIndexOrClass;
            mdi[size_t(nodes[i].featureIndex)] += (cnt[i] * imp[i] - cnt[l] * imp[l] - cnt[l + 1] * imp[l + 1]) / double(cnt[0]);
        }
    }
    for (auto & v : mdi) v /= double(m.size());
    return mdi;
}

TEST("MDI is the documented weighted impurity decrease, averaged over trees", "[unit]")
{
    const Dataset c = makeData(2000, 6, 3, 3);
    const Dataset r = makeData(2000, 6, 0, 4);
    cls::training::Parameter pc(3);
    pc.nTrees        = 4;
    pc.bootstrap     = false; // node sample counts are then the node's weight
    pc.varImportance = training::MDI;
    reg::training::Parameter pr;
    pr.nTrees        = 4;
    pr.bootstrap     = false;
    pr.varImportance = training::MDI;
    const auto fc    = fitCls<cls::training::defaultDense>(c, pc);
    const auto fr    = fitReg<reg::training::defaultDense>(r, pr);
    const struct
    {
        const dtrees::internal::ModelImpl * model;
        NumericTablePtr imp;
    } fits[] = { { fc.model, fc.result->get(cls::training::variableImportance) }, { fr.model, fr.result->get(reg::training::variableImportance) } };
    for (const auto & f : fits)
    {
        const std::vector<double> expected = mdiFromTables(*f.model, 6);
        const double * got                 = static_cast<HomogenNumericTable<double> *>(f.imp.get())->getArray();
        for (size_t j = 0; j < 6; ++j) REQUIRE(std::abs(got[j] - expected[j]) <= 1e-9 * (1 + std::abs(expected[j])));
    }
}

TEST("OOB error is the mean of the per-observation errors over the rows some tree left out", "[unit]")
{
    const Dataset c = makeData(3000, 6, 3, 5);
    const Dataset r = makeData(3000, 6, 0, 6);
    cls::training::Parameter pc(3);
    pc.nTrees           = 10;
    pc.resultsToCompute = training::computeOutOfBagError | training::computeOutOfBagErrorPerObservation;
    reg::training::Parameter pr;
    pr.nTrees                         = 10;
    pr.resultsToCompute               = training::computeOutOfBagError | training::computeOutOfBagErrorPerObservation;
    const auto fc                     = fitCls<cls::training::hist>(c, pc);
    const auto fr                     = fitReg<reg::training::hist>(r, pr);
    const NumericTablePtr tables[][2] = {
        { fc.result->get(cls::training::outOfBagError), fc.result->get(cls::training::outOfBagErrorPerObservation) },
        { fr.result->get(reg::training::outOfBagError), fr.result->get(reg::training::outOfBagErrorPerObservation) }
    };
    for (const auto & t : tables)
    {
        const double err      = static_cast<HomogenNumericTable<double> *>(t[0].get())->getArray()[0];
        const double * perObs = static_cast<HomogenNumericTable<double> *>(t[1].get())->getArray();
        double sum            = 0;
        size_t n              = 0;
        for (size_t i = 0; i < 3000; ++i)
            if (perObs[i] >= 0)
            {
                sum += perObs[i];
                ++n;
            }
        REQUIRE(n > 2900); // with 10 trees a row is in no OOB set with probability 0.63^10
        REQUIRE(std::abs(err - sum / double(n)) <= 1e-9);
    }
}

TEST("every engine trains, and a fit on the returned engine draws other trees", "[unit]")
{
    const Dataset c                    = makeData(1500, 6, 2, 7);
    const engines::EnginePtr engines[] = { engines::philox4x32x10::Batch<>::create(5), engines::mt2203::Batch<>::create(5),
                                           engines::mt19937::Batch<>::create(5), engines::mcg59::Batch<>::create(5),
                                           engines::mrg32k3a::Batch<>::create(5) };
    for (const auto & e : engines)
    {
        cls::training::Parameter p(2);
        p.nTrees         = 3;
        p.engine         = e;
        const auto first = fitCls<cls::training::hist>(c, p);
        p.engine         = first.result->get(cls::training::updatedEngine);
        REQUIRE(p.engine);
        const auto second = fitCls<cls::training::hist>(c, p);
        // The second fit reads other stream positions: its first tree has another bootstrap and draws.
        bool same = first.model->at(0)->getNumberOfRows() == second.model->at(0)->getNumberOfRows();
        if (same)
            for (size_t i = 0; i < first.model->at(0)->getNumberOfRows() && same; ++i)
                same = nodesOf(*first.model, 0)[i].featureValueOrResponse == nodesOf(*second.model, 0)[i].featureValueOrResponse;
        REQUIRE(!same);
    }
}

TEST("the tree engine scores as the legacy kernels do", "[unit]")
{
    const Dataset c  = makeData(6000, 10, 3, 8);
    const Dataset ct = makeData(3000, 10, 3, 9);
    const Dataset r  = makeData(6000, 10, 0, 10);
    const Dataset rt = makeData(3000, 10, 0, 11);
    for (bool hist : { true, false })
    {
        // Legacy as it shipped, with mt2203; the tree engine with philox4x32x10, its default.
        cls::training::Parameter pc(3);
        pc.nTrees = 30;
        reg::training::Parameter pr;
        pr.nTrees       = 30;
        pc.engine       = engines::philox4x32x10::Batch<>::create(1);
        pr.engine       = engines::philox4x32x10::Batch<>::create(1);
        const double cn = hist ? score(*fitCls<cls::training::hist>(c, pc).model, ct) : score(*fitCls<cls::training::defaultDense>(c, pc).model, ct);
        const double rn = hist ? score(*fitReg<reg::training::hist>(r, pr).model, rt) : score(*fitReg<reg::training::defaultDense>(r, pr).model, rt);
        pc.engine       = engines::mt2203::Batch<>::create(1);
        pr.engine       = engines::mt2203::Batch<>::create(1);
        const double cl =
            hist ? score(*fitCls<cls::training::hist>(c, pc, true).model, ct) : score(*fitCls<cls::training::defaultDense>(c, pc, true).model, ct);
        const double rl =
            hist ? score(*fitReg<reg::training::hist>(r, pr, true).model, rt) : score(*fitReg<reg::training::defaultDense>(r, pr, true).model, rt);
        CAPTURE(hist, cn, cl, rn, rl);
        REQUIRE(cn > cl - 0.02);
        REQUIRE(rn > rl - 0.02);
    }
}

TEST("serial subtrees do not change the forests", "[unit]")
{
    const Dataset c = makeData(6000, 8, 3, 12);
    const Dataset r = makeData(6000, 8, 0, 13);
    cls::training::Parameter pc(3);
    pc.nTrees = 6;
    reg::training::Parameter pr;
    pr.nTrees     = 6;
    const auto c0 = fitCls<cls::training::hist>(c, pc, false, 0);
    const auto r0 = fitReg<reg::training::hist>(r, pr, false, 0);
    for (int64 serial : { int64(-1), int64(64), int64(100000) }) // the L2-based cutoff, a small one, the whole tree
    {
        CAPTURE(serial);
        requireSameModels(*c0.model, *fitCls<cls::training::hist>(c, pc, false, serial).model);
        requireSameModels(*r0.model, *fitReg<reg::training::hist>(r, pr, false, serial).model);
    }
}

} // namespace daal::algorithms::decision_forest::test
