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
#include "src/algorithms/dtrees/forest/df_forest_trainer.h"
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
void setHyperparameters(Hyperparameter & hp, bool legacy)
{
    if constexpr (std::is_same<Hyperparameter, cls::training::internal::Hyperparameter>::value)
    {
        REQUIRE(hp.set(cls::training::internal::smallNClassesThreshold, int64(8)).ok());
        REQUIRE(hp.set(cls::training::internal::minPartCoefficient, int64(4)).ok());
        REQUIRE(hp.set(cls::training::internal::minSizeCoefficient, int64(24000)).ok());
        REQUIRE(hp.set(cls::training::internal::legacyTraining, int64(legacy)).ok());
    }
    else
    {
        REQUIRE(hp.set(reg::training::internal::minPartCoefficient, int64(4)).ok());
        REQUIRE(hp.set(reg::training::internal::minSizeCoefficient, int64(24000)).ok());
        REQUIRE(hp.set(reg::training::internal::legacyTraining, int64(legacy)).ok());
    }
}

/// A fit's model and result tables.
template <typename ResultPtr>
struct Fit
{
    ResultPtr result;
    NumericTablePtr x, y, w;
    const dtrees::internal::ModelImpl * model = nullptr;
};

/// Copies d (and weights w, if any) into tables of FPType.
template <typename FPType, typename ResultPtr>
void makeTables(const Dataset & d, const std::vector<double> & w, Fit<ResultPtr> & f)
{
    f.x = HomogenNumericTable<FPType>::create(d.p, d.n, NumericTable::doAllocate);
    f.y = HomogenNumericTable<FPType>::create(1, d.n, NumericTable::doAllocate);
    std::copy(d.x.begin(), d.x.end(), static_cast<HomogenNumericTable<FPType> *>(f.x.get())->getArray());
    std::copy(d.y.begin(), d.y.end(), static_cast<HomogenNumericTable<FPType> *>(f.y.get())->getArray());
    if (w.empty()) return;
    f.w = HomogenNumericTable<FPType>::create(1, d.n, NumericTable::doAllocate);
    std::copy(w.begin(), w.end(), static_cast<HomogenNumericTable<FPType> *>(f.w.get())->getArray());
}

template <cls::training::Method method, typename FPType = double>
Fit<cls::training::ResultPtr> fitCls(const Dataset & d, cls::training::Parameter par, bool legacy = false, const std::vector<double> & w = {})
{
    Fit<cls::training::ResultPtr> f;
    // As the Batch API sets it; Parameter::check rejects the base default of 0.
    if (!par.minObservationsInLeafNode) par.minObservationsInLeafNode = 1;
    makeTables<FPType>(d, w, f);
    classifier::training::Input input;
    input.set(classifier::training::data, f.x);
    input.set(classifier::training::labels, f.y);
    if (f.w) input.set(classifier::training::weights, f.w);
    f.result.reset(new cls::training::Result());
    REQUIRE(f.result->allocate<FPType>(&input, &par, int(method)).ok());
    auto * model = static_cast<cls::internal::ModelImpl *>(f.result->get(classifier::training::model).get());
    model->clear();
    model->setNFeatures(d.p);
    cls::training::internal::Hyperparameter hp;
    setHyperparameters(hp, legacy);
    cls::training::internal::ClassificationTrainBatchKernel<FPType, method, cpu> kernel;
    REQUIRE(kernel.compute(f.x.get(), f.y.get(), f.w.get(), *model, *f.result, par, &hp).ok());
    f.model = model;
    return f;
}

template <reg::training::Method method, typename FPType = double>
Fit<reg::training::ResultPtr> fitReg(const Dataset & d, reg::training::Parameter par, bool legacy = false, const std::vector<double> & w = {})
{
    Fit<reg::training::ResultPtr> f;
    if (!par.minObservationsInLeafNode) par.minObservationsInLeafNode = 5;
    makeTables<FPType>(d, w, f);
    reg::training::Input input;
    input.set(reg::training::data, f.x);
    input.set(reg::training::dependentVariable, f.y);
    if (f.w) input.set(reg::training::weights, f.w);
    f.result.reset(new reg::training::Result());
    REQUIRE(f.result->allocate<FPType>(&input, &par, int(method)).ok());
    auto * model = static_cast<reg::internal::ModelImpl *>(f.result->get(reg::training::model).get());
    model->clear();
    reg::training::internal::Hyperparameter hp;
    setHyperparameters(hp, legacy);
    reg::training::internal::RegressionTrainBatchKernel<FPType, method, cpu> kernel;
    REQUIRE(kernel.compute(f.x.get(), f.y.get(), f.w.get(), *model, *f.result, par, &hp).ok());
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

/// a and b hold the same values, bit for bit (NaN equal to NaN).
void requireSameTable(const NumericTablePtr & a, const NumericTablePtr & b)
{
    REQUIRE(a);
    REQUIRE(b);
    REQUIRE(a->getNumberOfRows() == b->getNumberOfRows());
    REQUIRE(a->getNumberOfColumns() == b->getNumberOfColumns());
    const double * va = static_cast<HomogenNumericTable<double> *>(a.get())->getArray();
    const double * vb = static_cast<HomogenNumericTable<double> *>(b.get())->getArray();
    for (size_t i = 0; i < a->getNumberOfRows() * a->getNumberOfColumns(); ++i)
    {
        CAPTURE(i, va[i], vb[i]);
        REQUIRE((va[i] == vb[i] || (std::isnan(va[i]) && std::isnan(vb[i]))));
    }
}

TEST("philox forests, OOB results and MDA are identical across thread counts and memory saving modes", "[unit]")
{
    const Dataset c = makeData(4000, 8, 3, 1);
    const Dataset r = makeData(4000, 8, 0, 2);
    for (bool memSave : { false, true })
    {
        // OOB and MDA sums once depended on which thread predicted which tree (#3911).
        cls::training::Parameter pc(3);
        pc.nTrees           = 6;
        pc.memorySavingMode = memSave;
        pc.varImportance    = training::MDA_Raw;
        pc.resultsToCompute = training::computeOutOfBagError | training::computeOutOfBagErrorPerObservation | training::computeOutOfBagErrorAccuracy
                              | training::computeOutOfBagErrorDecisionFunction;
        reg::training::Parameter pr;
        pr.nTrees           = 6;
        pr.memorySavingMode = memSave;
        pr.varImportance    = training::MDA_Raw;
        pr.resultsToCompute = training::computeOutOfBagError | training::computeOutOfBagErrorPerObservation | training::computeOutOfBagErrorR2
                              | training::computeOutOfBagErrorPrediction;
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
        for (auto id : { cls::training::outOfBagError, cls::training::outOfBagErrorPerObservation, cls::training::outOfBagErrorAccuracy,
                         cls::training::outOfBagErrorDecisionFunction, cls::training::variableImportance })
            requireSameTable(c1->result->get(id), cn.result->get(id));
        for (auto id : { reg::training::outOfBagError, reg::training::outOfBagErrorPerObservation, reg::training::outOfBagErrorR2,
                         reg::training::outOfBagErrorPrediction, reg::training::variableImportance })
            requireSameTable(r1->result->get(id), rn.result->get(id));
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

/// Feature 0 holds adjacent floats from 1, so the double midpoint of two neighbouring values is a rounding tie
/// in float; feature 1 is noise. Classes and responses change every few rows.
Dataset adjacentFloats(size_t n, size_t nClasses, uint64_t seed)
{
    std::mt19937_64 gen(seed);
    std::normal_distribution<float> normal;
    Dataset d;
    d.n        = n;
    d.p        = 2;
    d.nClasses = nClasses;
    d.x.resize(2 * n);
    d.y.resize(n);
    float v = 1.f;
    for (size_t i = 0; i < n; ++i)
    {
        d.x[2 * i]        = v;
        d.x[2 * i + 1]    = normal(gen);
        v                 = std::nextafter(v, 2.f);
        const size_t band = (i / 5 + (gen() % 8 == 0)) % 2;
        d.y[i]            = nClasses ? double(band) : double(band) + 0.01 * normal(gen);
    }
    return d;
}

/// Every split of the model routes every row of d as float prediction does, which casts the threshold to float.
void requireFloatRouting(const dtrees::internal::ModelImpl & m, const Dataset & d)
{
    size_t nSplits = 0;
    for (size_t t = 0; t < m.size(); ++t)
    {
        const DecisionTreeNode * nodes = nodesOf(m, t);
        for (size_t i = 0; i < m.at(t)->getNumberOfRows(); ++i)
        {
            if (nodes[i].featureIndex < 0) continue;
            ++nSplits;
            const double t64 = nodes[i].featureValueOrResponse;
            for (size_t r = 0; r < d.n; ++r)
            {
                const float x = float(d.x[r * d.p + size_t(nodes[i].featureIndex)]);
                CAPTURE(t, i, r, x, t64);
                REQUIRE((double(x) <= t64) == !(x > float(t64)));
            }
        }
    }
    REQUIRE(nSplits > 0);
}

TEST("float thresholds route the training rows as float prediction does", "[unit]")
{
    const Dataset c = adjacentFloats(2000, 2, 17);
    const Dataset r = adjacentFloats(2000, 0, 18);
    for (const auto splitter : { training::best, training::random })
        for (const auto strategy : { training::quantiles, training::averages })
        {
            CAPTURE(int(splitter), int(strategy));
            cls::training::Parameter pc(2);
            pc.nTrees          = 4;
            pc.splitter        = splitter;
            pc.binningStrategy = strategy;
            reg::training::Parameter pr;
            pr.nTrees          = 4;
            pr.splitter        = splitter;
            pr.binningStrategy = strategy;
            requireFloatRouting(*fitCls<cls::training::hist, float>(c, pc).model, c);
            requireFloatRouting(*fitReg<reg::training::hist, float>(r, pr).model, r);
            if (strategy != training::quantiles) continue;
            requireFloatRouting(*fitCls<cls::training::defaultDense, float>(c, pc).model, c);
            requireFloatRouting(*fitReg<reg::training::defaultDense, float>(r, pr).model, r);
        }
}

TEST("regression fits responses with a large offset as it fits them without", "[unit]")
{
    const Dataset r = makeData(4000, 6, 0, 19);
    Dataset shifted = r;
    for (auto & v : shifted.y) v += 1e8;
    reg::training::Parameter pr;
    pr.nTrees = 5;
    for (bool hist : { true, false })
    {
        const double plain = hist ? score(*fitReg<reg::training::hist>(r, pr).model, r) : score(*fitReg<reg::training::defaultDense>(r, pr).model, r);
        const double offset = hist ? score(*fitReg<reg::training::hist>(shifted, pr).model, shifted) :
                                     score(*fitReg<reg::training::defaultDense>(shifted, pr).model, shifted);
        CAPTURE(hist, plain, offset);
        REQUIRE(plain > 0.8);
        REQUIRE(offset > plain - 0.02);
    }
}

TEST("trees whose in-bag rows all have zero weight are single leaves", "[unit]")
{
    const Dataset c = makeData(100, 4, 3, 20);
    const Dataset r = makeData(100, 4, 0, 21);
    std::vector<double> w(100, 0);
    w[0] = 1;
    cls::training::Parameter pc(3);
    pc.nTrees = 20;
    reg::training::Parameter pr;
    pr.nTrees     = 20;
    const auto fc = fitCls<cls::training::defaultDense>(c, pc, false, w);
    const auto fr = fitReg<reg::training::defaultDense>(r, pr, false, w);
    size_t empty  = 0;
    for (size_t t = 0; t < pc.nTrees; ++t)
    {
        // Only row 0 has weight: a tree is one leaf, of row 0 if it drew row 0 and uninformed otherwise.
        REQUIRE(fc.model->at(t)->getNumberOfRows() == 1);
        REQUIRE(fr.model->at(t)->getNumberOfRows() == 1);
        const double * prob = fc.model->getProbas(t);
        const bool drewRow0 = prob[size_t(c.y[0])] == 1;
        empty += !drewRow0;
        for (size_t k = 0; k < 3; ++k) REQUIRE(prob[k] == (drewRow0 ? (k == size_t(c.y[0]) ? 1. : 0.) : 1. / 3));
        REQUIRE(std::abs(nodesOf(*fr.model, t)[0].featureValueOrResponse - r.y[0]) <= 1e-12 * (1 + std::abs(r.y[0])));
    }
    REQUIRE(empty > 0);
}

/// The models have the same tree structures and splits, and leaf values equal up to rounding (node sample counts
/// are not compared).
bool sameModels(const dtrees::internal::ModelImpl & a, const dtrees::internal::ModelImpl & b)
{
    if (a.size() != b.size()) return false;
    for (size_t t = 0; t < a.size(); ++t)
    {
        const size_t n = a.at(t)->getNumberOfRows();
        if (n != b.at(t)->getNumberOfRows()) return false;
        for (size_t i = 0; i < n; ++i)
        {
            const DecisionTreeNode & na = nodesOf(a, t)[i];
            const DecisionTreeNode & nb = nodesOf(b, t)[i];
            if (na.featureIndex != nb.featureIndex || na.leftIndexOrClass != nb.leftIndexOrClass) return false;
            const double va = na.featureValueOrResponse, vb = nb.featureValueOrResponse;
            if (na.featureIndex >= 0 ? va != vb : std::abs(va - vb) > 1e-12 * (1 + std::abs(va))) return false;
        }
    }
    return true;
}

TEST("a constant user weight changes neither the trees nor the minimum impurity decrease", "[unit]")
{
    // Weights are normalized to a maximum of 1 and the minimum decrease scales by the normalized total weight,
    // so weights of 7 fit as no weights. Scaling it by the raw weights (#2077) would act as a 7 times larger
    // minimum, which the last checks show changes the forests.
    const Dataset c = makeData(3000, 6, 3, 22);
    const Dataset r = makeData(3000, 6, 0, 23);
    const std::vector<double> w(3000, 7.);
    cls::training::Parameter pc(3);
    pc.nTrees                         = 4;
    pc.minImpurityDecreaseInSplitNode = 1e-3;
    reg::training::Parameter pr;
    pr.nTrees                         = 4;
    pr.minImpurityDecreaseInSplitNode = 1e-3;
    // Leaf sizes count bootstrap draws without weights and rows with them (TreeParams::sizesInDraws), so only
    // a minimum of 1 fits both alike.
    pr.minObservationsInLeafNode = 1;
    for (bool hist : { true, false })
    {
        CAPTURE(hist);
        auto fitC = [&](const cls::training::Parameter & p, const std::vector<double> & weights) {
            return hist ? fitCls<cls::training::hist>(c, p, false, weights) : fitCls<cls::training::defaultDense>(c, p, false, weights);
        };
        auto fitR = [&](const reg::training::Parameter & p, const std::vector<double> & weights) {
            return hist ? fitReg<reg::training::hist>(r, p, false, weights) : fitReg<reg::training::defaultDense>(r, p, false, weights);
        };
        const auto c1 = fitC(pc, {});
        const auto r1 = fitR(pr, {});
        // Node sample counts differ: weighted records count distinct rows, unweighted ones bootstrap draws.
        REQUIRE(sameModels(*c1.model, *fitC(pc, w).model));
        REQUIRE(sameModels(*r1.model, *fitR(pr, w).model));
        cls::training::Parameter pc7       = pc;
        pc7.minImpurityDecreaseInSplitNode = 7e-3;
        reg::training::Parameter pr7       = pr;
        pr7.minImpurityDecreaseInSplitNode = 7e-3;
        REQUIRE(!sameModels(*c1.model, *fitC(pc7, {}).model));
        REQUIRE(!sameModels(*r1.model, *fitR(pr7, {}).model));
    }
}

TEST("with every feature drawn, each of two identical features wins some splits", "[unit]")
{
    // The feature order is drawn even when every feature is taken (legacy once used the identity order there,
    // #2077), and a tie keeps the feature drawn first, so either copy can win.
    Dataset c = makeData(2000, 2, 3, 24);
    for (size_t i = 0; i < c.n; ++i) c.x[2 * i + 1] = c.x[2 * i];
    cls::training::Parameter pc(3);
    pc.nTrees          = 5;
    pc.featuresPerNode = 2;
    for (bool hist : { true, false })
    {
        CAPTURE(hist);
        const auto f       = hist ? fitCls<cls::training::hist>(c, pc) : fitCls<cls::training::defaultDense>(c, pc);
        size_t splitsOn[2] = { 0, 0 };
        for (size_t t = 0; t < f.model->size(); ++t)
            for (size_t i = 0; i < f.model->at(t)->getNumberOfRows(); ++i)
                if (nodesOf(*f.model, t)[i].featureIndex >= 0) ++splitsOn[nodesOf(*f.model, t)[i].featureIndex];
        CAPTURE(splitsOn[0], splitsOn[1]);
        REQUIRE(splitsOn[0] > 0);
        REQUIRE(splitsOn[1] > 0);
    }
}

TEST("float regression splits responses whose variance is below float epsilon", "[unit]")
{
    // The impurity threshold defaults to double epsilon whatever the data type. With float's (1.2e-7, #3653)
    // the root, of variance 2.5e-9, would be a leaf.
    Dataset d;
    d.n = 200;
    d.p = 1;
    d.x.resize(d.n);
    d.y.resize(d.n);
    for (size_t i = 0; i < d.n; ++i)
    {
        d.x[i] = double(i);
        d.y[i] = double(float((i / 10) % 2 ? 1e-4 : 0.));
    }
    reg::training::Parameter pr;
    pr.nTrees                    = 1;
    pr.bootstrap                 = false;
    pr.minObservationsInLeafNode = 1;
    for (bool hist : { true, false })
    {
        CAPTURE(hist);
        const auto f = hist ? fitReg<reg::training::hist, float>(d, pr) : fitReg<reg::training::defaultDense, float>(d, pr);
        REQUIRE(f.model->at(0)->getNumberOfRows() > 1);
        double sse = 0;
        for (size_t i = 0; i < d.n; ++i)
        {
            const double e = nodesOf(*f.model, 0)[leafOf(nodesOf(*f.model, 0), &d.x[i])].featureValueOrResponse - d.y[i];
            sse += e * e;
        }
        REQUIRE(sse / double(d.n) < 1e-16);
    }
}

TEST("node sample counts are bootstrap draws without weights and distinct rows with them", "[unit]")
{
    // Pins the current semantics (design 8, step 6 notes): weighted records keep no bootstrap count. The
    // distinct-rows follow-up makes unweighted fits count rows too, and changes the first check.
    const size_t n  = 3000;
    const Dataset c = makeData(n, 6, 3, 25);
    const Dataset r = makeData(n, 6, 0, 26);
    const std::vector<double> w(n, 7.);
    cls::training::Parameter pc(3);
    pc.nTrees = 4;
    reg::training::Parameter pr;
    pr.nTrees                                        = 4;
    const auto c1                                    = fitCls<cls::training::hist>(c, pc);
    const auto cw                                    = fitCls<cls::training::hist>(c, pc, false, w);
    const auto r1                                    = fitReg<reg::training::hist>(r, pr);
    const auto rw                                    = fitReg<reg::training::hist>(r, pr, false, w);
    const dtrees::internal::ModelImpl * unweighted[] = { c1.model, r1.model };
    const dtrees::internal::ModelImpl * weighted[]   = { cw.model, rw.model };
    for (size_t m = 0; m < 2; ++m)
        for (size_t t = 0; t < 4; ++t)
        {
            CAPTURE(m, t);
            // The root holds n draws, of about 0.63 n distinct rows.
            REQUIRE(size_t(unweighted[m]->getNodeSampleCount(t)[0]) == n);
            const size_t rows = size_t(weighted[m]->getNodeSampleCount(t)[0]);
            REQUIRE(rows > n / 2);
            REQUIRE(rows < 3 * n / 4);
        }
}

TEST("tables below numaMinTableBytes use one NUMA node and do not query the topology", "[unit]")
{
    // Querying the topology (threader_numa_count) loads hwloc, the cost the threshold avoids, so the small-table
    // placement must not come from it.
    const auto small = training::internal::numaPlacement(training::internal::numaMinTableBytes - 1, true);
    REQUIRE(small.nodes == 1);
    REQUIRE(!small.replicate);
    const auto large = training::internal::numaPlacement(training::internal::numaMinTableBytes, true);
    REQUIRE(large.nodes == size_t(threader_numa_count()));
    REQUIRE(large.replicate);
    REQUIRE(!training::internal::numaPlacement(training::internal::numaMinTableBytes, false).replicate);
}

} // namespace daal::algorithms::decision_forest::test
