/* file: df_regression_train_dense_default_impl.i */
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
//  Implementation of auxiliary functions for decision forest regression
//  (defaultDense) method.
//--
*/

#ifndef __DF_REGRESSION_TRAIN_DENSE_DEFAULT_IMPL_I__
#define __DF_REGRESSION_TRAIN_DENSE_DEFAULT_IMPL_I__

#include "src/algorithms/dtrees/forest/df_train_dense_default_impl.i"
#include "src/algorithms/dtrees/forest/regression/df_regression_train_kernel.h"
#include "src/algorithms/dtrees/forest/regression/df_regression_model_impl.h"
#include "src/algorithms/dtrees/dtrees_predict_dense_default_impl.i"
#include "src/algorithms/dtrees/forest/regression/df_regression_training_types_result.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace regression
{
namespace training
{
namespace internal
{
using namespace decision_forest::training::internal;
using namespace dtrees::internal;
using namespace dtrees::training::internal;

//computes mean2 and var2 as the mean and mse for the set of elements s2, s2 = s - s1
//where mean, var are mean and mse for s,
//where mean1, var1 are mean and mse for s1
template <typename algorithmFPType, CpuType cpu>
void subtractImpurity(algorithmFPType var, algorithmFPType mean, algorithmFPType var1, algorithmFPType mean1, algorithmFPType leftWeights,
                      algorithmFPType & var2, algorithmFPType & mean2, algorithmFPType rightWeights)
{
    //TODO: investigate reusing decision_tree::regression::training::internal::MSEDataStatistics here
    mean2                   = mean + (leftWeights * (mean - mean1)) / rightWeights;
    const algorithmFPType b = leftWeights / rightWeights;
    var2                    = var + (mean - mean2) * (mean + mean2) + (var - var1 + (mean - mean1) * (mean + mean1)) * b;
    if (var2 < 0) var2 = 0;
}

//computes meanPrev as the mean of n-1 elements after removing of element x (based on mean of n elements passed as 'mean' argument)
//instead of impurity, computes the sum of (xi - meanPrev)(xi - meanPrev) for n-1 elements
//(based on the sum of (xi - mean)*(xi - mean) of n elements passed as 'var' argument)
template <typename algorithmFPType, CpuType cpu>
void calcPrevImpurity(algorithmFPType var, algorithmFPType mean, algorithmFPType & varPrev, algorithmFPType & meanPrev, algorithmFPType x,
                      algorithmFPType totalWeights, algorithmFPType weights)
{
    algorithmFPType residual = (isPositive<algorithmFPType, cpu>(totalWeights - weights) ? (totalWeights - weights) : 1.);
    algorithmFPType delta    = (x - mean) / residual;
    varPrev                  = var - delta * totalWeights * (x - mean) * weights;
    meanPrev                 = mean - delta * weights;
    if (varPrev < 0) varPrev = 0;
}

//////////////////////////////////////////////////////////////////////////////////////////
// Service structure, contains regression error data for OOB calculation
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
struct RegErr
{
    algorithmFPType value = 0;
    size_t count          = 0;
    void add(const RegErr & o)
    {
        count += o.count;
        value += o.value;
    }
};

//////////////////////////////////////////////////////////////////////////////////////////
// OrderedRespHelperBest
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class OrderedRespHelperBest : public DataHelper<algorithmFPType, algorithmFPType, cpu>
{
public:
    typedef algorithmFPType TResponse;
    typedef DataHelper<algorithmFPType, algorithmFPType, cpu> super;
    typedef double intermSummFPType;

    struct ImpurityData
    {
        double var; //impurity is a variance
        double mean;
        double value() const { return var; }
    };

    typedef SplitData<algorithmFPType, ImpurityData> TSplitData;

public:
    OrderedRespHelperBest(const dtrees::internal::IndexedFeatures * indexedFeatures, size_t dummy,
                          const regression::training::internal::Hyperparameter * hp = nullptr)
        : super(indexedFeatures)
    {}

    template <bool noWeights>
    void calcImpurity(const IndexType * aIdx, size_t n, ImpurityData & imp, double & totalweights) const;

    template <bool noWeights, bool featureUnordered>
    int findBestSplitByHist(size_t nDiffFeatMax, intermSummFPType sumTotal, algorithmFPType * buf, size_t n, size_t nMinSplitPart,
                            const ImpurityData & curImpurity, TSplitData & split, const algorithmFPType minWeightLeaf,
                            const algorithmFPType totalWeights, const IndexType iFeature) const;

    template <bool noWeights>
    bool findBestSplitOrderedFeature(const algorithmFPType * featureVal, const IndexType * aIdx, size_t n, size_t nMinSplitPart,
                                     const algorithmFPType accuracy, const ImpurityData & curImpurity, TSplitData & split,
                                     const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights) const;
    template <bool noWeights>
    bool findBestSplitCategoricalFeature(const algorithmFPType * featureVal, const IndexType * aIdx, size_t n, size_t nMinSplitPart,
                                         const algorithmFPType accuracy, const ImpurityData & curImpurity, TSplitData & split,
                                         const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights) const;

#ifdef DEBUG_CHECK_IMPURITY
    void checkImpurity(const IndexType * ptrIdx, size_t n, const ImpurityData & expected) const
    {
        checkImpurityInternal(ptrIdx, n, expected, false);
    }
    void checkImpurityInternal(const IndexType * ptrIdx, size_t n, const ImpurityData & expected, bool bInternal = true) const;
#endif

protected:
    //buffer for the computation using indexed features
    mutable TVector<IndexType, cpu, DefaultAllocator<cpu> > _idxFeatureBuf;
    mutable TVector<algorithmFPType, cpu, DefaultAllocator<cpu> > _weightsFeatureBuf;
};

template <typename algorithmFPType, CpuType cpu>
template <bool noWeights>
void OrderedRespHelperBest<algorithmFPType, cpu>::calcImpurity(const IndexType * aIdx, size_t n, ImpurityData & imp, double & totalWeights) const
{
    imp.var  = 0;
    imp.mean = this->_aResponse[aIdx[0]].val;
    if (noWeights)
    {
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 1; i < n; ++i)
        {
            const double delta = this->_aResponse[aIdx[i]].val - imp.mean; //x[i] - mean
            imp.mean += delta / double(i + 1);
            imp.var += delta * (this->_aResponse[aIdx[i]].val - imp.mean);
        }
        totalWeights = double(n);
        imp.var /= double(n); //impurity is MSE
    }
    else
    {
        totalWeights = this->_aWeights[aIdx[0]].val;
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 1; i < n; ++i)
        {
            const double weights = this->_aWeights[aIdx[i]].val;
            const double delta   = this->_aResponse[aIdx[i]].val - imp.mean; //x[i] - mean
            totalWeights += weights;
            DAAL_ASSERT(!(isZero<double, cpu>(totalWeights)));
            imp.mean += weights * delta / totalWeights;
            imp.var += weights * delta * (this->_aResponse[aIdx[i]].val - imp.mean);
        }
        imp.var /= totalWeights; //impurity is MSE
    }

#ifdef DEBUG_CHECK_IMPURITY
    if (!this->_weights)
    {
        TResponse mean1 = this->_aResponse[aIdx[0]].val / algorithmFPType(n);
        for (size_t i = 1; i < n; ++i) mean1 += this->_aResponse[aIdx[i]].val / algorithmFPType(n);
        algorithmFPType var1 = 0;
        for (size_t i = 0; i < n; ++i) var1 += (this->_aResponse[aIdx[i]].val - mean1) * (this->_aResponse[aIdx[i]].val - mean1);
        var1 /= algorithmFPType(n); //impurity is MSE
        DAAL_ASSERT(fabs(mean1 - imp.mean) < 0.001);
        DAAL_ASSERT(fabs(var1 - imp.var) < 0.001);
    }
#endif
}

template <typename algorithmFPType, CpuType cpu>
template <bool noWeights, bool featureUnordered>
int OrderedRespHelperBest<algorithmFPType, cpu>::findBestSplitByHist(size_t nDiffFeatMax, intermSummFPType sumTotal, algorithmFPType * buf, size_t n,
                                                                     size_t nMinSplitPart, const ImpurityData & curImpurity, TSplitData & split,
                                                                     const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights,
                                                                     const IndexType iFeature) const
{
    auto featWeights = _weightsFeatureBuf.get();
    auto nFeatIdx    = _idxFeatureBuf.get(); //number of indexed feature values, array

    intermSummFPType bestImpDecreasePart =
        split.impurityDecrease < 0 ? -1 : (split.impurityDecrease + curImpurity.mean * curImpurity.mean) * totalWeights;
    size_t nLeft                = 0;
    algorithmFPType leftWeights = 0.;
    intermSummFPType sumLeft    = 0;
    int idxFeatureBestSplit     = -1; //index of best feature value in the array of sorted feature values
    for (size_t i = 0; i < nDiffFeatMax; ++i)
    {
        if (!nFeatIdx[i]) continue;

        algorithmFPType thisFeatWeights = noWeights ? nFeatIdx[i] : featWeights[i];

        nLeft       = (featureUnordered ? nFeatIdx[i] : nLeft + nFeatIdx[i]);
        leftWeights = (featureUnordered ? thisFeatWeights : leftWeights + thisFeatWeights);
        if ((nLeft == n) //last split
            || ((n - nLeft) < nMinSplitPart) || ((totalWeights - leftWeights) < minWeightLeaf))
            break;
        sumLeft = (featureUnordered ? buf[i] : sumLeft + buf[i]);
        if ((nLeft < nMinSplitPart) || (leftWeights < minWeightLeaf)) continue;
        intermSummFPType sumRight = sumTotal - sumLeft;
        //the part of the impurity decrease dependent on split itself
        const intermSummFPType impDecreasePart = sumLeft * sumLeft / leftWeights + sumRight * sumRight / (totalWeights - leftWeights);
        if (impDecreasePart > bestImpDecreasePart)
        {
            split.left.mean     = algorithmFPType(sumLeft);
            split.nLeft         = nLeft;
            split.leftWeights   = leftWeights;
            idxFeatureBestSplit = i;
            bestImpDecreasePart = impDecreasePart;
        }
    }
    if (idxFeatureBestSplit >= 0)
    {
        split.totalWeights     = totalWeights;
        split.impurityDecrease = (bestImpDecreasePart / totalWeights - curImpurity.mean * curImpurity.mean);
        //note, left.mean and right.mean are not actually the means but the sums
    }
    return idxFeatureBestSplit;
}

template <typename algorithmFPType, CpuType cpu>
template <bool noWeights>
bool OrderedRespHelperBest<algorithmFPType, cpu>::findBestSplitOrderedFeature(const algorithmFPType * featureVal, const IndexType * aIdx, size_t n,
                                                                              size_t nMinSplitPart, const algorithmFPType accuracy,
                                                                              const ImpurityData & curImpurity, TSplitData & split,
                                                                              const algorithmFPType minWeightLeaf,
                                                                              const algorithmFPType totalWeights) const
{
    ImpurityData left;
    ImpurityData right;
    algorithmFPType xi = this->_aResponse[aIdx[0]].val;
    left.var           = 0;
    left.mean          = xi;
    IndexType iBest    = -1;
    algorithmFPType vBest;
    auto aResponse = this->_aResponse.get();
    auto aWeights  = this->_aWeights.get();
    auto weights   = aWeights[aIdx[0]].val;
    calcPrevImpurity<double, cpu>(curImpurity.var * totalWeights, curImpurity.mean, right.var, right.mean, xi, totalWeights, weights);
#ifdef DEBUG_CHECK_IMPURITY
    checkImpurityInternal(aIdx + 1, n - 1, right);
#endif

    vBest = split.impurityDecrease < 0 ? daal::services::internal::MaxVal<algorithmFPType>::get() :
                                         (curImpurity.var - split.impurityDecrease) * totalWeights;
    if (noWeights)
    {
        for (size_t i = 1; i < (n - nMinSplitPart + 1); ++i)
        {
            const bool bSameFeaturePrev(featureVal[i] <= featureVal[i - 1] + accuracy);

            if (!(bSameFeaturePrev || (i < nMinSplitPart) || (i < minWeightLeaf) || ((n - i) < minWeightLeaf)))
            {
                //can make a split
                //nLeft == i, nRight == n - i
                const algorithmFPType v = left.var + right.var;
                if (v < vBest)
                {
                    vBest             = v;
                    split.left.var    = left.var;
                    split.left.mean   = left.mean;
                    split.leftWeights = i;
                    iBest             = i;
                }
            }

            //update impurity and continue
            xi                    = aResponse[aIdx[i]].val;
            algorithmFPType delta = xi - left.mean;
            left.mean += delta / algorithmFPType(i + 1);
            left.var += delta * (xi - left.mean);
            if (left.var < 0) left.var = 0;
            calcPrevImpurity<double, cpu>(right.var, right.mean, right.var, right.mean, xi, double(n - i), 1.);
#ifdef DEBUG_CHECK_IMPURITY
            checkImpurityInternal(aIdx, i + 1, left);
            checkImpurityInternal(aIdx + i + 1, n - i - 1, right);
#endif
        }
    }
    else
    {
        algorithmFPType leftWeights = weights;
        for (size_t i = 1; i < (n - nMinSplitPart + 1); ++i)
        {
            weights = aWeights[aIdx[i]].val;
            const bool bSameFeaturePrev(featureVal[i] <= featureVal[i - 1] + accuracy);

            if (!(bSameFeaturePrev || (i < nMinSplitPart) || (leftWeights < minWeightLeaf) || ((totalWeights - leftWeights) < minWeightLeaf)))
            {
                //can make a split
                //nLeft == i, nRight == n - i
                const algorithmFPType v = left.var + right.var;
                if (v < vBest)
                {
                    vBest             = v;
                    split.left.var    = left.var;
                    split.left.mean   = left.mean;
                    split.leftWeights = leftWeights;
                    iBest             = i;
                }
            }

            //update impurity and continue
            xi                    = aResponse[aIdx[i]].val;
            algorithmFPType delta = xi - left.mean;
            left.mean += weights * delta / (isPositive<algorithmFPType, cpu>(leftWeights + weights) ? leftWeights + weights : 1.);
            left.var += weights * delta * (xi - left.mean);
            if (left.var < 0) left.var = 0;
            calcPrevImpurity<double, cpu>(right.var, right.mean, right.var, right.mean, xi, totalWeights - leftWeights, weights);
            leftWeights += weights;
#ifdef DEBUG_CHECK_IMPURITY
            checkImpurityInternal(aIdx, i + 1, left);
            checkImpurityInternal(aIdx + i + 1, n - i - 1, right);
#endif
        }
    }

    if (iBest < 0) return false;

    split.impurityDecrease = curImpurity.var - vBest / totalWeights;
    split.nLeft            = iBest;
    split.totalWeights     = totalWeights;
    split.left.var /= (isPositive<algorithmFPType, cpu>(split.leftWeights) ? split.leftWeights : 1.);
    split.iStart       = 0;
    split.featureValue = featureVal[iBest - 1];
    return true;
}

template <typename algorithmFPType, CpuType cpu>
template <bool noWeights>
bool OrderedRespHelperBest<algorithmFPType, cpu>::findBestSplitCategoricalFeature(const algorithmFPType * featureVal, const IndexType * aIdx,
                                                                                  size_t n, size_t nMinSplitPart, const algorithmFPType accuracy,
                                                                                  const ImpurityData & curImpurity, TSplitData & split,
                                                                                  const algorithmFPType minWeightLeaf,
                                                                                  const algorithmFPType totalWeights) const
{
    DAAL_ASSERT(n >= 2 * nMinSplitPart);
    ImpurityData left;
    ImpurityData right;
    algorithmFPType vBest;
    bool bFound               = false;
    size_t nDiffFeatureValues = 0;
    auto aResponse            = this->_aResponse.get();
    auto aWeights             = this->_aWeights.get();

    for (size_t i = 0; i < n - nMinSplitPart;)
    {
        ++nDiffFeatureValues;
        size_t count                   = 1;
        const algorithmFPType firstVal = featureVal[i];
        const size_t iStart            = i;
        algorithmFPType leftWeights    = aWeights[aIdx[i]].val;
        for (++i; (i < n) && (featureVal[i] == firstVal); ++count, ++i)
        {
            leftWeights += aWeights[aIdx[i]].val;
        }
        if ((count < nMinSplitPart) || ((n - count) < nMinSplitPart) || (leftWeights < minWeightLeaf)
            || ((totalWeights - leftWeights) < minWeightLeaf))
            continue;

        if ((i == n) && (nDiffFeatureValues == 2) && bFound) break; //only 2 feature values, one possible split, already found

        double weights = double(0);
        calcImpurity<noWeights>(aIdx + iStart, count, left, weights);
        DAAL_ASSERT(fabs(weights - leftWeights) < 0.001);
        subtractImpurity<double, cpu>(curImpurity.var, curImpurity.mean, left.var, left.mean, leftWeights, right.var, right.mean,
                                      totalWeights - leftWeights);
        const algorithmFPType v = leftWeights * left.var + (totalWeights - leftWeights) * right.var;
        if (!bFound || v < vBest)
        {
            vBest              = v;
            split.left.var     = left.var;
            split.left.mean    = left.mean;
            split.nLeft        = count;
            split.leftWeights  = leftWeights;
            split.iStart       = iStart;
            split.featureValue = firstVal;
            bFound             = true;
        }
    }
    if (bFound)
    {
        const algorithmFPType impurityDecrease = curImpurity.var - vBest / (isPositive<algorithmFPType, cpu>(totalWeights) ? totalWeights : 1.);
        if (split.impurityDecrease < 0 || split.impurityDecrease < impurityDecrease)
        {
            split.impurityDecrease = impurityDecrease;
            split.totalWeights     = totalWeights;
            return true;
        }
    }
    return false;
}

#ifdef DEBUG_CHECK_IMPURITY
template <typename algorithmFPType, CpuType cpu>
void OrderedRespHelperBest<algorithmFPType, cpu>::checkImpurityInternal(const IndexType * ptrIdx, size_t n, const ImpurityData & expected,
                                                                        bool bInternal) const
{
    if (!this->_weights)
    {
        algorithmFPType div = 1. / algorithmFPType(n);
        TResponse cMean     = this->_aResponse[ptrIdx[0]].val * div;
        for (size_t i = 1; i < n; ++i) cMean += this->_aResponse[ptrIdx[i]].val * div;
        algorithmFPType cVar = 0;
        for (size_t i = 0; i < n; ++i) cVar += (this->_aResponse[ptrIdx[i]].val - cMean) * (this->_aResponse[ptrIdx[i]].val - cMean);
        if (!bInternal) cVar *= div;
        DAAL_ASSERT(fabs(cMean - expected.mean) < 0.001);
        DAAL_ASSERT(fabs(cVar - expected.var) < 0.001);
    }
}
#endif

//////////////////////////////////////////////////////////////////////////////////////////
// RespHelperBase contains common elements needed to select and split data
// Using CRTP, the base allows for certain templated functions to be polymorphic.
// Specifically, those functions which generates splits for features.
// It understands them through inheritance from OrderedRespHelperBest, and is
// then has different versions in OrderedRespHelperRandom.
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu, typename crtp>
class RespHelperBase : public OrderedRespHelperBest<algorithmFPType, cpu>
{
public:
    typedef algorithmFPType TResponse;
    typedef dtrees::internal::TreeImpRegression<> TreeType;
    typedef typename TreeType::NodeType NodeType;
    using ImpurityData = typename OrderedRespHelperBest<algorithmFPType, cpu>::ImpurityData;
    using TSplitData   = typename OrderedRespHelperBest<algorithmFPType, cpu>::TSplitData;
    using super        = typename OrderedRespHelperBest<algorithmFPType, cpu>::super;

    engines::internal::BatchBaseImpl * engineImpl;

public:
    RespHelperBase(const dtrees::internal::IndexedFeatures * indexedFeatures, size_t dummy,
                   const regression::training::internal::Hyperparameter * hp = nullptr)
        : OrderedRespHelperBest<algorithmFPType, cpu>(indexedFeatures, dummy, hp)
    {}
    virtual bool init(const NumericTable * data, const NumericTable * resp, const IndexType * aSample,
                      const NumericTable * weights) DAAL_C11_OVERRIDE;
    void convertLeftImpToRight(size_t n, const ImpurityData & total, TSplitData & split)
    {
        subtractImpurity<double, cpu>(total.var, total.mean, split.left.var, split.left.mean, split.leftWeights, split.left.var, split.left.mean,
                                      split.totalWeights - split.leftWeights);
        split.nLeft       = n - split.nLeft;
        split.leftWeights = split.totalWeights - split.leftWeights;
    }

    bool findSplitForFeature(const algorithmFPType * featureVal, const IndexType * aIdx, size_t n, size_t nMinSplitPart,
                             const algorithmFPType accuracy, const ImpurityData & curImpurity, TSplitData & split,
                             const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights) const;

    template <typename BinIndexType>
    int findSplitForFeatureSorted(algorithmFPType * featureBuf, IndexType iFeature, const IndexType * aIdx, size_t n, size_t nMinSplitPart,
                                  const ImpurityData & curImpurity, TSplitData & split, const algorithmFPType minWeightLeaf,
                                  const algorithmFPType totalWeights, const BinIndexType * binIndex) const;

    typedef double intermSummFPType;
    template <typename BinIndexType>
    void computeHistWithWeights(algorithmFPType * buf, IndexType iFeature, const IndexType * aIdx, const BinIndexType * binIndex, size_t n,
                                intermSummFPType & sumTotal) const;
    template <typename BinIndexType>
    void computeHistWithoutWeights(algorithmFPType * buf, IndexType iFeature, const IndexType * aIdx, const BinIndexType * binIndex, size_t n,
                                   intermSummFPType & sumTotal) const;

    template <bool noWeights, typename BinIndexType>
    void finalizeBestSplit(const IndexType * aIdx, const BinIndexType * binIndex, size_t n, IndexType iFeature, size_t idxFeatureValueBestSplit,
                           TSplitData & bestSplit, IndexType * bestSplitIdx) const;
    void simpleSplit(const algorithmFPType * featureVal, const IndexType * aIdx, TSplitData & split) const;
    void singleSwap(const IndexType aIdx, TSplitData & split) const;
    bool terminateCriteria(ImpurityData & imp, algorithmFPType impurityThreshold, size_t nSamples) const { return imp.value() < impurityThreshold; }

    TResponse predict(const dtrees::internal::Tree & t, const algorithmFPType * x) const
    {
        const typename TreeType::NodeType::Base * pNode = dtrees::prediction::internal::findNode<algorithmFPType, TreeType, cpu>(t, x);
        DAAL_ASSERT(pNode);
        return pNode ? TreeType::NodeType::castLeaf(pNode)->response : 0.;
    }

    algorithmFPType predictionError(TResponse prediction, TResponse response) const { return (prediction - response) * (prediction - response); }

    algorithmFPType predictionError(const dtrees::internal::Tree & t, const algorithmFPType * x, const NumericTable * resp, size_t iRow,
                                    byte * oobBuf) const
    {
        ReadRows<algorithmFPType, cpu> y(const_cast<NumericTable *>(resp), iRow, 1);
        const TResponse response(this->predict(t, x));
        algorithmFPType val = this->predictionError(response, *y.get());
        if (oobBuf)
        {
            ((RegErr<algorithmFPType, cpu> *)oobBuf)[iRow].value += response;
            ((RegErr<algorithmFPType, cpu> *)oobBuf)[iRow].count++;
        }
        return val;
    }

    void setLeafData(typename TreeType::NodeType::Leaf & node, const IndexType * idx, size_t n, ImpurityData & imp) const
    {
        DAAL_ASSERT(n > 0);
        node.response = imp.mean;
#ifdef DEBUG_CHECK_IMPURITY
        if (!this->_weights)
        {
            algorithmFPType response;
            algorithmFPType val = calcResponse(response, idx, n);
            node.response       = response;
            DAAL_ASSERT(fabs(val - imp.mean) < 0.001);
        }
#endif
        node.count    = n;
        node.impurity = imp.var;
    }

private:
#ifdef DEBUG_CHECK_IMPURITY
    algorithmFPType calcResponse(algorithmFPType & res, const IndexType * idx, size_t n) const;
#endif
};

#ifdef DEBUG_CHECK_IMPURITY
template <typename algorithmFPType, CpuType cpu, typename crtp>
void RespHelperBase<algorithmFPType, cpu, crtp>::checkImpurityInternal(const IndexType * ptrIdx, size_t n, const ImpurityData & expected,
                                                                       bool bInternal) const
{
    if (!this->_weights)
    {
        algorithmFPType div = 1. / algorithmFPType(n);
        TResponse cMean     = this->_aResponse[ptrIdx[0]].val * div;
        for (size_t i = 1; i < n; ++i) cMean += this->_aResponse[ptrIdx[i]].val * div;
        algorithmFPType cVar = 0;
        for (size_t i = 0; i < n; ++i) cVar += (this->_aResponse[ptrIdx[i]].val - cMean) * (this->_aResponse[ptrIdx[i]].val - cMean);
        if (!bInternal) cVar *= div;
        DAAL_ASSERT(fabs(cMean - expected.mean) < 0.001);
        DAAL_ASSERT(fabs(cVar - expected.var) < 0.001);
    }
}
#endif

template <typename algorithmFPType, CpuType cpu, typename crtp>
bool RespHelperBase<algorithmFPType, cpu, crtp>::init(const NumericTable * data, const NumericTable * resp, const IndexType * aSample,
                                                      const NumericTable * weights)
{
    DAAL_CHECK_STATUS_VAR(super::init(data, resp, aSample, weights));
    if (this->_indexedFeatures)
    {
        //init work buffer for the computation using indexed features
        const auto nDiffFeatMax = this->indexedFeatures().maxNumIndices();
        this->_idxFeatureBuf.reset(nDiffFeatMax);
        this->_weightsFeatureBuf.reset(nDiffFeatMax);
        return this->_idxFeatureBuf.get() && this->_weightsFeatureBuf.get();
    }
    return true;
}

#ifdef DEBUG_CHECK_IMPURITY
template <typename algorithmFPType, CpuType cpu, typename crtp>
algorithmFPType RespHelperBase<algorithmFPType, cpu, crtp>::calcResponse(algorithmFPType & res, const IndexType * idx, size_t n) const
{
    const algorithmFPType cDiv = 1. / algorithmFPType(n);
    res                        = this->_aResponse[idx[0]].val * cDiv;
    for (size_t i = 1; i < n; ++i) res += this->_aResponse[idx[i]].val * cDiv;
    return res;
}
#endif

template <typename algorithmFPType, CpuType cpu, typename crtp>
void RespHelperBase<algorithmFPType, cpu, crtp>::simpleSplit(const algorithmFPType * featureVal, const IndexType * aIdx, TSplitData & split) const
{
    split.featureValue = featureVal[0];
    split.left.var     = 0;
    split.left.mean    = this->_aResponse[aIdx[0]].val;
    split.nLeft        = 1;
    split.iStart       = 0;
    split.totalWeights = this->_aWeights[aIdx[0]].val + this->_aWeights[aIdx[1]].val;
    split.leftWeights  = this->_aWeights[aIdx[0]].val;
}

template <typename algorithmFPType, CpuType cpu, typename crtp>
void RespHelperBase<algorithmFPType, cpu, crtp>::singleSwap(const IndexType aIdx, TSplitData & split) const
{
    split.left.var    = 0;
    split.left.mean   = this->_aResponse[aIdx].val;
    split.nLeft       = 1;
    split.leftWeights = this->_aWeights[aIdx].val;
}

template <typename algorithmFPType, CpuType cpu, typename crtp>
bool RespHelperBase<algorithmFPType, cpu, crtp>::findSplitForFeature(const algorithmFPType * featureVal, const IndexType * aIdx, size_t n,
                                                                     size_t nMinSplitPart, const algorithmFPType accuracy,
                                                                     const ImpurityData & curImpurity, TSplitData & split,
                                                                     const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights) const
{
    const bool noWeights = !this->_weights;
    if (noWeights)
    {
        return split.featureUnordered ? static_cast<const crtp *>(this)->template findBestSplitCategoricalFeature<true>(
                   featureVal, aIdx, n, nMinSplitPart, accuracy, curImpurity, split, minWeightLeaf, totalWeights) :
                                        static_cast<const crtp *>(this)->template findBestSplitOrderedFeature<true>(
                                            featureVal, aIdx, n, nMinSplitPart, accuracy, curImpurity, split, minWeightLeaf, totalWeights);
    }
    else
    {
        return split.featureUnordered ? static_cast<const crtp *>(this)->template findBestSplitCategoricalFeature<false>(
                   featureVal, aIdx, n, nMinSplitPart, accuracy, curImpurity, split, minWeightLeaf, totalWeights) :
                                        static_cast<const crtp *>(this)->template findBestSplitOrderedFeature<false>(
                                            featureVal, aIdx, n, nMinSplitPart, accuracy, curImpurity, split, minWeightLeaf, totalWeights);
    }
}

template <typename algorithmFPType, CpuType cpu, typename crtp>
template <bool noWeights, typename BinIndexType>
void RespHelperBase<algorithmFPType, cpu, crtp>::finalizeBestSplit(const IndexType * aIdx, const BinIndexType * binIndex, size_t n,
                                                                   IndexType iFeature, size_t idxFeatureValueBestSplit, TSplitData & bestSplit,
                                                                   IndexType * bestSplitIdx) const
{
    DAAL_ASSERT(bestSplit.nLeft > 0);
    DAAL_ASSERT(bestSplit.leftWeights > 0.);
    algorithmFPType divL = 1.;
    int iRowSplitVal     = -1;
    int iNext            = -1;
    int idxNext          = this->_aResponse.size() - 1;
    size_t iLeft         = 0;
    size_t iRight        = 0;
    if (noWeights)
    {
        divL = algorithmFPType(1.) / algorithmFPType(bestSplit.nLeft);

        bestSplit.left.mean *= divL;
        bestSplit.left.var            = 0;
        IndexType * bestSplitIdxRight = bestSplitIdx + bestSplit.nLeft;
        const auto aResponse          = this->_aResponse.get();
        for (size_t i = 0; i < n; ++i)
        {
            const auto iSample = aIdx[i];
            const auto idx     = binIndex[aResponse[iSample].idx];
            if ((bestSplit.featureUnordered && (idx != idxFeatureValueBestSplit))
                || ((!bestSplit.featureUnordered) && (idx > idxFeatureValueBestSplit)))
            {
                DAAL_ASSERT(iRight < n - bestSplit.nLeft);
                bestSplitIdxRight[iRight++] = iSample;
            }
            else
            {
                if (idx == idxFeatureValueBestSplit) iRowSplitVal = aResponse[iSample].idx;
                DAAL_ASSERT(iLeft < bestSplit.nLeft);
                bestSplitIdx[iLeft++]   = iSample;
                const algorithmFPType y = aResponse[iSample].val;
                bestSplit.left.var += (y - bestSplit.left.mean) * (y - bestSplit.left.mean);
            }
            if ((idx > idxFeatureValueBestSplit) && (idxNext > idx))
            {
                idxNext = idx;
                iNext   = aResponse[iSample].idx;
            }
        }
    }
    else
    {
        divL = isZero<algorithmFPType, cpu>(bestSplit.leftWeights) ? algorithmFPType(1.) : (algorithmFPType(1.) / bestSplit.leftWeights);

        bestSplit.left.mean *= divL;
        bestSplit.left.var            = 0;
        IndexType * bestSplitIdxRight = bestSplitIdx + bestSplit.nLeft;
        const auto aResponse          = this->_aResponse.get();
        const auto aWeights           = this->_aWeights.get();
        for (size_t i = 0; i < n; ++i)
        {
            const auto iSample = aIdx[i];
            const auto idx     = binIndex[aResponse[iSample].idx];
            if ((bestSplit.featureUnordered && (idx != idxFeatureValueBestSplit))
                || ((!bestSplit.featureUnordered) && (idx > idxFeatureValueBestSplit)))
            {
                DAAL_ASSERT(iRight < n - bestSplit.nLeft);
                bestSplitIdxRight[iRight++] = iSample;
            }
            else
            {
                if (idx == idxFeatureValueBestSplit) iRowSplitVal = aResponse[iSample].idx;
                DAAL_ASSERT(iLeft < bestSplit.nLeft);
                bestSplitIdx[iLeft++]   = iSample;
                const algorithmFPType y = aResponse[iSample].val;
                const algorithmFPType w = aWeights[iSample].val;
                bestSplit.left.var += w * (y - bestSplit.left.mean) * (y - bestSplit.left.mean);
            }
            if ((idx > idxFeatureValueBestSplit) && (idxNext > idx))
            {
                idxNext = idx;
                iNext   = aResponse[iSample].idx;
            }
        }
    }

    DAAL_ASSERT(iRight == n - bestSplit.nLeft);
    DAAL_ASSERT(iLeft == bestSplit.nLeft);
    bestSplit.left.var *= divL;
    bestSplit.iStart = 0;
    DAAL_ASSERT(iRowSplitVal >= 0);
    if (idxNext == this->_aResponse.size() - 1) iNext = iRowSplitVal;
    if (iNext >= 0 && iRowSplitVal >= 0)
    {
        bestSplit.featureValue = (this->getValue(iFeature, iRowSplitVal) + this->getValue(iFeature, iNext)) / (algorithmFPType)2.;
        if (bestSplit.featureValue == this->getValue(iFeature, iNext)) bestSplit.featureValue = this->getValue(iFeature, iRowSplitVal);
    }
}

template <typename algorithmFPType, CpuType cpu, typename crtp>
template <typename BinIndexType>
void RespHelperBase<algorithmFPType, cpu, crtp>::computeHistWithoutWeights(algorithmFPType * buf, IndexType iFeature, const IndexType * aIdx,
                                                                           const BinIndexType * binIndex, size_t n, intermSummFPType & sumTotal) const
{
    auto nFeatIdx  = this->_idxFeatureBuf.get(); //number of indexed feature values, array
    auto aResponse = this->_aResponse.get();
    sumTotal       = 0; //total sum of responses in the set being split
    {
        for (size_t i = 0; i < n; ++i)
        {
            const IndexType iSample            = aIdx[i];
            const typename super::Response & r = aResponse[aIdx[i]];
            const BinIndexType idx             = binIndex[r.idx];
            ++nFeatIdx[idx];
            buf[idx] += aResponse[iSample].val;
            sumTotal += aResponse[iSample].val;
        }
    }
}

template <typename algorithmFPType, CpuType cpu, typename crtp>
template <typename BinIndexType>
void RespHelperBase<algorithmFPType, cpu, crtp>::computeHistWithWeights(algorithmFPType * buf, IndexType iFeature, const IndexType * aIdx,
                                                                        const BinIndexType * binIndex, size_t n, intermSummFPType & sumTotal) const
{
    auto nFeatIdx    = this->_idxFeatureBuf.get(); //number of indexed feature values, array
    auto featWeights = this->_weightsFeatureBuf.get();
    auto aResponse   = this->_aResponse.get();
    auto aWeights    = this->_aWeights.get();
    sumTotal         = 0; //total sum of responses in the set being split
    {
        for (size_t i = 0; i < n; ++i)
        {
            const IndexType iSample            = aIdx[i];
            const typename super::Response & r = aResponse[aIdx[i]];
            const BinIndexType idx             = binIndex[r.idx];
            const auto weights                 = aWeights[iSample].val;
            ++nFeatIdx[idx];
            featWeights[idx] += weights;
            buf[idx] += aResponse[iSample].val * weights;
            sumTotal += aResponse[iSample].val * weights;
        }
    }
}

template <typename algorithmFPType, CpuType cpu, typename crtp>
template <typename BinIndexType>
int RespHelperBase<algorithmFPType, cpu, crtp>::findSplitForFeatureSorted(algorithmFPType * buf, IndexType iFeature, const IndexType * aIdx, size_t n,
                                                                          size_t nMinSplitPart, const ImpurityData & curImpurity, TSplitData & split,
                                                                          const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights,
                                                                          const BinIndexType * binIndex) const
{
    const auto nDiffFeatMax = this->indexedFeatures().numIndices(iFeature);
    this->_idxFeatureBuf.setValues(nDiffFeatMax, 0);

    //the buffer keeps sums of responses for each of unique feature values
    for (size_t i = 0; i < nDiffFeatMax; ++i) buf[i] = algorithmFPType(0);

    const bool noWeights      = !this->_weights;
    intermSummFPType sumTotal = 0; //total sum of responses in the set being split

    if (noWeights)
    {
        computeHistWithoutWeights(buf, iFeature, aIdx, binIndex, n, sumTotal);

        if (split.featureUnordered)
        {
            return static_cast<const crtp *>(this)->template findBestSplitByHist<true, true>(
                nDiffFeatMax, sumTotal, buf, n, nMinSplitPart, curImpurity, split, minWeightLeaf, totalWeights, iFeature);
        }
        else
        {
            return static_cast<const crtp *>(this)->template findBestSplitByHist<true, false>(
                nDiffFeatMax, sumTotal, buf, n, nMinSplitPart, curImpurity, split, minWeightLeaf, totalWeights, iFeature);
        }
    }
    else
    {
        this->_weightsFeatureBuf.setValues(nDiffFeatMax, algorithmFPType(0));
        computeHistWithWeights(buf, iFeature, aIdx, binIndex, n, sumTotal);

        if (split.featureUnordered)
        {
            return static_cast<const crtp *>(this)->template findBestSplitByHist<false, true>(
                nDiffFeatMax, sumTotal, buf, n, nMinSplitPart, curImpurity, split, minWeightLeaf, totalWeights, iFeature);
        }
        else
        {
            return static_cast<const crtp *>(this)->template findBestSplitByHist<false, false>(
                nDiffFeatMax, sumTotal, buf, n, nMinSplitPart, curImpurity, split, minWeightLeaf, totalWeights, iFeature);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
// OrderedRespHelperRandom class for random splitting regression
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class OrderedRespHelperRandom : public RespHelperBase<algorithmFPType, cpu, OrderedRespHelperRandom<algorithmFPType, cpu> >
{
public:
    typedef double intermSummFPType;
    using ImpurityData = typename RespHelperBase<algorithmFPType, cpu, OrderedRespHelperRandom<algorithmFPType, cpu> >::ImpurityData;
    using TSplitData   = typename RespHelperBase<algorithmFPType, cpu, OrderedRespHelperRandom<algorithmFPType, cpu> >::TSplitData;

public:
    OrderedRespHelperRandom(const dtrees::internal::IndexedFeatures * indexedFeatures, size_t dummy,
                            const regression::training::internal::Hyperparameter * hp = nullptr)
        : RespHelperBase<algorithmFPType, cpu, OrderedRespHelperRandom<algorithmFPType, cpu> >(indexedFeatures, dummy, hp)
    {}

    size_t genRandomBinIdx(const IndexType iFeature, const size_t minidx, const size_t maxidx) const;

    template <bool noWeights, bool featureUnordered>
    int findBestSplitByHist(size_t nDiffFeatMax, intermSummFPType sumTotal, algorithmFPType * buf, size_t n, size_t nMinSplitPart,
                            const ImpurityData & curImpurity, TSplitData & split, const algorithmFPType minWeightLeaf,
                            const algorithmFPType totalWeights, const IndexType iFeature) const;

    template <bool noWeights>
    bool findBestSplitOrderedFeature(const algorithmFPType * featureVal, const IndexType * aIdx, size_t n, size_t nMinSplitPart,
                                     const algorithmFPType accuracy, const ImpurityData & curImpurity, TSplitData & split,
                                     const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights) const;
    template <bool noWeights>
    bool findBestSplitCategoricalFeature(const algorithmFPType * featureVal, const IndexType * aIdx, size_t n, size_t nMinSplitPart,
                                         const algorithmFPType accuracy, const ImpurityData & curImpurity, TSplitData & split,
                                         const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights) const;
};

template <typename algorithmFPType, CpuType cpu>
size_t OrderedRespHelperRandom<algorithmFPType, cpu>::genRandomBinIdx(const IndexType iFeature, const size_t minidx, const size_t maxidx) const
{
    //randomly select a histogram split index
    algorithmFPType fidx   = 0;
    algorithmFPType minval = minidx ? this->indexedFeatures().binRightBorder(iFeature, minidx - 1) : this->indexedFeatures().min(iFeature);
    algorithmFPType maxval = this->indexedFeatures().binRightBorder(iFeature, maxidx);
    size_t mid;
    size_t l   = minidx;
    size_t idx = maxidx;
    RNGsInst<algorithmFPType, cpu> rng;
    rng.uniform(1, &fidx, this->engineImpl->getState(), minval, maxval); //find random index between minidx and maxidx

    while (l < idx)
    {
        mid = l + (idx - l) / 2;
        if (this->indexedFeatures().binRightBorder(iFeature, idx) > fidx)
        {
            idx = mid;
        }
        else
        {
            l = mid + 1;
        }
    }
    return idx;
}

template <typename algorithmFPType, CpuType cpu>
template <bool noWeights, bool featureUnordered>
int OrderedRespHelperRandom<algorithmFPType, cpu>::findBestSplitByHist(size_t nDiffFeatMax, intermSummFPType sumTotal, algorithmFPType * buf,
                                                                       size_t n, size_t nMinSplitPart, const ImpurityData & curImpurity,
                                                                       TSplitData & split, const algorithmFPType minWeightLeaf,
                                                                       const algorithmFPType totalWeights, const IndexType iFeature) const
{
    auto featWeights = this->_weightsFeatureBuf.get();
    auto nFeatIdx    = this->_idxFeatureBuf.get(); //number of indexed feature values, array

    intermSummFPType bestImpDecreasePart =
        split.impurityDecrease < 0 ? -1 : (split.impurityDecrease + curImpurity.mean * curImpurity.mean) * totalWeights;
    size_t nLeft                = 0;
    algorithmFPType leftWeights = 0.;
    intermSummFPType sumLeft    = 0;
    int idxFeatureBestSplit     = -1; //index of best feature value in the array of sorted feature values

    size_t minidx = 0;
    size_t maxidx = nDiffFeatMax - 1;
    size_t idx;

    for (; (minidx < maxidx) && isZero<IndexType, cpu>(nFeatIdx[minidx]); minidx++)
        ;

    for (; (minidx < maxidx) && isZero<IndexType, cpu>(nFeatIdx[maxidx]); maxidx--)
        ;

    DAAL_ASSERT(minidx < maxidx); //if the if statement after minidx search doesn't activate, we have an issue.
    if (minidx == maxidx)
    {
        return idxFeatureBestSplit;
    }

    //randomly select a histogram split index
    if (featureUnordered)
    {
        RNGsInst<size_t, cpu> rng;
        rng.uniform(1, &idx, this->engineImpl->getState(), minidx, maxidx); //find random index between minidx and maxidx
    }
    else
    {
        idx = this->genRandomBinIdx(iFeature, minidx, maxidx);
    }

    for (; isZero<IndexType, cpu>(nFeatIdx[idx]); idx--)
        ;

    if (noWeights)
    {
        if (featureUnordered)
        {
            nLeft       = nFeatIdx[idx];
            sumLeft     = buf[idx];
            leftWeights = nFeatIdx[idx];
        }
        else
        {
            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t i = minidx; i <= idx; ++i)
            {
                nLeft += nFeatIdx[i];
                sumLeft += buf[i];
            }
            leftWeights = nLeft;
        }
    }
    else
    {
        if (featureUnordered)
        {
            nLeft       = nFeatIdx[idx];
            sumLeft     = buf[idx];
            leftWeights = featWeights[idx];
        }
        else
        {
            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t i = minidx; i <= idx; ++i)
            {
                nLeft += nFeatIdx[i];
                sumLeft += buf[i];
                leftWeights += featWeights[i];
            }
        }
    }

    if (!(((n - nLeft) < nMinSplitPart) || ((totalWeights - leftWeights) < minWeightLeaf) || (nLeft < nMinSplitPart)
          || (leftWeights < minWeightLeaf)))
    {
        intermSummFPType sumRight = sumTotal - sumLeft;
        //the part of the impurity decrease dependent on split itself
        const intermSummFPType impDecreasePart = sumLeft * sumLeft / leftWeights + sumRight * sumRight / (totalWeights - leftWeights);

        if (impDecreasePart > bestImpDecreasePart)
        {
            split.left.mean     = algorithmFPType(sumLeft);
            split.nLeft         = nLeft;
            split.leftWeights   = leftWeights;
            idxFeatureBestSplit = idx;
            bestImpDecreasePart = impDecreasePart;
        }
    }

    if (idxFeatureBestSplit >= 0)
    {
        split.totalWeights     = totalWeights;
        split.impurityDecrease = (bestImpDecreasePart / totalWeights - curImpurity.mean * curImpurity.mean);
        //note, left.mean and right.mean are not actually the means but the sums
    }
    return idxFeatureBestSplit;
}

template <typename algorithmFPType, CpuType cpu>
template <bool noWeights>
bool OrderedRespHelperRandom<algorithmFPType, cpu>::findBestSplitOrderedFeature(const algorithmFPType * featureVal, const IndexType * aIdx, size_t n,
                                                                                size_t nMinSplitPart, const algorithmFPType accuracy,
                                                                                const ImpurityData & curImpurity, TSplitData & split,
                                                                                const algorithmFPType minWeightLeaf,
                                                                                const algorithmFPType totalWeights) const
{
    ImpurityData left;
    ImpurityData right;
    IndexType iBest = -1;
    algorithmFPType vBest;
    algorithmFPType leftWeights = 0.;
    auto aResponse              = this->_aResponse.get();
    auto aWeights               = this->_aWeights.get();
    algorithmFPType idx;
    vBest = split.impurityDecrease < 0 ? daal::services::internal::MaxVal<algorithmFPType>::get() :
                                         (curImpurity.var - split.impurityDecrease) * totalWeights;
    size_t i;

    //select random split index
    RNGsInst<algorithmFPType, cpu> rng;
    rng.uniform(1, &idx, this->engineImpl->getState(), featureVal[0],
                featureVal[n - 1]); //this strategy follows sklearn's implementation

    if (idx >= featureVal[n - nMinSplitPart]
        || idx < featureVal[nMinSplitPart - 1]) //check if sufficient samples will exist, and not a constant feature
    {
        return false;
    }

    //binary search to reduce computation, rather than O(n) vector lookups, we do a O(log(n)) index find
    size_t mid;
    size_t l = 0;
    size_t r = n - 1;

    while (l < r)
    {
        mid = l + (r - l) / 2;
        if (featureVal[mid] > idx)
        {
            r = mid;
        }
        else
        {
            l = mid + 1;
        }
    }

    left.var   = 0;
    left.mean  = this->_aResponse[aIdx[0]].val;
    right.var  = 0;
    right.mean = this->_aResponse[aIdx[r]].val;
    if (noWeights)
    {
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 1; i < r; ++i)
        {
            const double delta = this->_aResponse[aIdx[i]].val - left.mean; //x[i] - mean
            left.mean += delta / double(i + 1);
            left.var += delta * (this->_aResponse[aIdx[i]].val - left.mean);
            DAAL_ASSERT(left.var >= 0);
        }

        PRAGMA_VECTOR_ALWAYS
        for (size_t i = r + 1; i < n; ++i)
        {
            const double delta = this->_aResponse[aIdx[i]].val - right.mean; //x[i] - mean
            right.mean += delta / double(i + 1 - r);
            right.var += delta * (this->_aResponse[aIdx[i]].val - right.mean);
        }

        leftWeights = r;
    }
    else
    {
        leftWeights = this->_aWeights[aIdx[0]].val;
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 1; i < r; ++i)
        {
            const double weights = this->_aWeights[aIdx[i]].val;
            const double delta   = this->_aResponse[aIdx[i]].val - left.mean; //x[i] - mean
            leftWeights += weights;
            DAAL_ASSERT(!(isZero<double, cpu>(leftWeights)));
            left.mean += weights * delta / leftWeights;
            left.var += weights * delta * (this->_aResponse[aIdx[i]].val - left.mean);
        }

        algorithmFPType rightWeights = this->_aWeights[aIdx[r]].val;
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = r + 1; i < n; ++i)
        {
            const double weights = this->_aWeights[aIdx[i]].val;
            const double delta   = this->_aResponse[aIdx[i]].val - right.mean; //x[i] - mean
            rightWeights += weights;
            DAAL_ASSERT(!(isZero<double, cpu>(rightWeights)));
            right.mean += weights * delta / rightWeights;
            right.var += weights * delta * (this->_aResponse[aIdx[i]].val - right.mean);
        }
    }

    if (!((leftWeights < minWeightLeaf) || ((totalWeights - leftWeights) < minWeightLeaf)))
    {
        const algorithmFPType v = left.var + right.var;
        if (v < vBest)
        {
            vBest             = v;
            split.left.var    = left.var;
            split.left.mean   = left.mean;
            split.leftWeights = leftWeights;
            iBest             = r;
        }
    }

    if (iBest < 0) return false;

    split.impurityDecrease = curImpurity.var - vBest / totalWeights;
    split.nLeft            = iBest;
    split.totalWeights     = totalWeights;
    split.left.var /= (isPositive<algorithmFPType, cpu>(split.leftWeights) ? split.leftWeights : 1.);
    split.iStart       = 0;
    split.featureValue = idx;
    return true;
}

template <typename algorithmFPType, CpuType cpu>
template <bool noWeights>
bool OrderedRespHelperRandom<algorithmFPType, cpu>::findBestSplitCategoricalFeature(const algorithmFPType * featureVal, const IndexType * aIdx,
                                                                                    size_t n, size_t nMinSplitPart, const algorithmFPType accuracy,
                                                                                    const ImpurityData & curImpurity, TSplitData & split,
                                                                                    const algorithmFPType minWeightLeaf,
                                                                                    const algorithmFPType totalWeights) const
{
    DAAL_ASSERT(n >= 2 * nMinSplitPart);
    ImpurityData left;
    ImpurityData right;
    algorithmFPType vBest;
    bool bFound = false;
    //size_t nDiffFeatureValues = 0;
    auto aResponse      = this->_aResponse.get();
    auto aWeights       = this->_aWeights.get();
    algorithmFPType min = featureVal[0];
    algorithmFPType max = featureVal[0];
    algorithmFPType idx;
    algorithmFPType firstVal;

    for (size_t i = 1; i < n; ++i)
    {
        max = featureVal[i] > max ? featureVal[i] : max;
        min = featureVal[i] < min ? featureVal[i] : min;
    }

    firstVal = min;

    RNGsInst<algorithmFPType, cpu> rng;
    rng.uniform(1, &idx, this->engineImpl->getState(), min, max); //this strategy follows sklearn's implementation

    for (size_t i = 1; i < n; ++i)
    {
        firstVal = featureVal[i] <= idx && featureVal[i] > firstVal ? featureVal[i] : firstVal;
    }
    //first is the closest categorical feature less than the idx O(n) computation as ordering of featureVal is unknown.

    for (size_t i = 0; i < n - nMinSplitPart;)
    {
        //++nDiffFeatureValues;
        if (featureVal[i] != firstVal)
        {
            i++;
            continue;
        }
        size_t count                = 1;
        firstVal                    = featureVal[i];
        const size_t iStart         = i;
        algorithmFPType leftWeights = aWeights[aIdx[i]].val;
        for (++i; (i < n) && (featureVal[i] == firstVal); ++count, ++i)
        {
            leftWeights += aWeights[aIdx[i]].val;
        }
        if ((count < nMinSplitPart) || ((n - count) < nMinSplitPart) || (leftWeights < minWeightLeaf)
            || ((totalWeights - leftWeights) < minWeightLeaf))
            continue;

        //if ((i == n) && (nDiffFeatureValues == 2) && bFound) break; //only 2 feature values, one possible split, already found

        double weights = double(0);
        this->template calcImpurity<noWeights>(aIdx + iStart, count, left, weights);
        DAAL_ASSERT(fabs(weights - leftWeights) < 0.001);
        subtractImpurity<double, cpu>(curImpurity.var, curImpurity.mean, left.var, left.mean, leftWeights, right.var, right.mean,
                                      totalWeights - leftWeights);
        const algorithmFPType v = leftWeights * left.var + (totalWeights - leftWeights) * right.var;
        if (!bFound || v < vBest)
        {
            vBest              = v;
            split.left.var     = left.var;
            split.left.mean    = left.mean;
            split.nLeft        = count;
            split.leftWeights  = leftWeights;
            split.iStart       = iStart;
            split.featureValue = firstVal;
            bFound             = true;
        }
    }
    if (bFound)
    {
        const algorithmFPType impurityDecrease = curImpurity.var - vBest / (isPositive<algorithmFPType, cpu>(totalWeights) ? totalWeights : 1.);
        if (split.impurityDecrease < 0 || split.impurityDecrease < impurityDecrease)
        {
            split.impurityDecrease = impurityDecrease;
            split.totalWeights     = totalWeights;
            return true;
        }
    }
    return false;
}

//////////////////////////////////////////////////////////////////////////////////////////
// TreeThreadCtx class for regression
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class TreeThreadCtx : public TreeThreadCtxBase<algorithmFPType, cpu>
{
public:
    typedef TreeThreadCtxBase<algorithmFPType, cpu> super;
    TreeThreadCtx(algorithmFPType * _varImp = nullptr) : super(_varImp) {}
    bool init(const decision_forest::training::Parameter & par, const NumericTable * x, size_t /*dummy*/)
    {
        DAAL_CHECK_STATUS_VAR(super::init(par, x));
        using namespace decision_forest::training;
        if (par.resultsToCompute
            & (computeOutOfBagError | computeOutOfBagErrorPerObservation | computeOutOfBagErrorR2 | computeOutOfBagErrorPrediction))
        {
            size_t sz    = sizeof(RegErr<algorithmFPType, cpu>) * x->getNumberOfRows();
            this->oobBuf = service_calloc<byte, cpu>(sz);
            DAAL_CHECK_STATUS_VAR(this->oobBuf);
        }
        return true;
    }

    void reduceTo(decision_forest::training::VariableImportanceMode mode, TreeThreadCtx & other, size_t nVars, size_t nSamples) const
    {
        super::reduceTo(mode, other, nVars, nSamples);
        if (this->oobBuf)
        {
            RegErr<algorithmFPType, cpu> * dst       = (RegErr<algorithmFPType, cpu> *)other.oobBuf;
            const RegErr<algorithmFPType, cpu> * src = (const RegErr<algorithmFPType, cpu> *)this->oobBuf;
            for (size_t i = 0; i < nSamples; ++i) dst[i].add(src[i]);
        }
    }

    Status finalizeOOBError(const NumericTable * resp, algorithmFPType * res, algorithmFPType * resPerObs, algorithmFPType * resAccuracy,
                            algorithmFPType * resR2, algorithmFPType * resDecisionFunction, algorithmFPType * resPrediction) const
    {
        DAAL_ASSERT(this->oobBuf);
        const size_t nSamples = resp->getNumberOfRows();
        ReadRows<algorithmFPType, cpu> y(const_cast<NumericTable *>(resp), 0, nSamples);
        DAAL_CHECK_BLOCK_STATUS(y);
        const algorithmFPType * py         = y.get();
        size_t nPredicted                  = 0;
        algorithmFPType _res               = 0;
        algorithmFPType yMean              = 0;
        algorithmFPType sumMeanDiff        = 0;
        RegErr<algorithmFPType, cpu> * ptr = (RegErr<algorithmFPType, cpu> *)this->oobBuf;

        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < nSamples; ++i)
        {
            yMean += py[i];
        }
        yMean /= nSamples;

        for (size_t i = 0; i < nSamples; ++i)
        {
            if (ptr[i].count)
            {
                ptr[i].value /= algorithmFPType(ptr[i].count);
                const algorithmFPType oobForObs = (ptr[i].value - py[i]) * (ptr[i].value - py[i]);

                if (resPerObs) resPerObs[i] = oobForObs;
                _res += oobForObs;
                ++nPredicted;

                if (resPrediction) resPrediction[i] = ptr[i].value;
                sumMeanDiff += (py[i] - yMean) * (py[i] - yMean);
            }
            else
            {
                if (resPerObs) resPerObs[i] = algorithmFPType(-1); //was not in OOB set of any tree and hence not predicted
                if (resPrediction) resPrediction[i] = algorithmFPType(0);
            }
        }
        if (res && nPredicted > 0) *res = _res / algorithmFPType(nPredicted);
        if (resR2) *resR2 = 1 - _res / sumMeanDiff;
        return Status();
    }
};

//////////////////////////////////////////////////////////////////////////////////////////
// TrainBatchTask for regression
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, typename BinIndexType, decision_forest::regression::training::Method method, typename Helper,
          typename HyperparameterType, CpuType cpu>
class TrainBatchTask : public TrainBatchTaskBase<algorithmFPType, BinIndexType, Helper, HyperparameterType, cpu>
{
    typedef TrainBatchTaskBase<algorithmFPType, BinIndexType, Helper, HyperparameterType, cpu> super;

public:
    typedef TreeThreadCtx<algorithmFPType, cpu> ThreadCtxType;
    TrainBatchTask(HostAppIface * hostApp, const NumericTable * x, const NumericTable * y, const NumericTable * w,
                   const decision_forest::training::Parameter & par, const dtrees::internal::FeatureTypes & featTypes,
                   const dtrees::internal::IndexedFeatures * indexedFeatures, const BinIndexType * binIndex, typename super::ThreadCtxType & ctx,
                   size_t dummy, const HyperparameterType * hyperparameter = nullptr)
        : super(hostApp, x, y, w, par, featTypes, indexedFeatures, binIndex, ctx, dummy, hyperparameter)
    {
        if (!this->_nFeaturesPerNode)
        {
            size_t nF                                     = x->getNumberOfColumns() / 3;
            const_cast<size_t &>(this->_nFeaturesPerNode) = (nF < 1 ? 1 : nF);
        }
    }
};

//////////////////////////////////////////////////////////////////////////////////////////
// RegressionTrainBatchKernel
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, Method method, CpuType cpu, typename helper>
services::Status computeForSpecificHelper(HostAppIface * pHostApp, const NumericTable * x, const NumericTable * y, const NumericTable * w,
                                          decision_forest::regression::Model & m, Result & res, const Parameter & par, bool memSave,
                                          const regression::training::internal::Hyperparameter * hyperparameter)
{
    typedef regression::training::internal::Hyperparameter HyperparameterType;
    ResultData rd(par, res.get(variableImportance).get(), res.get(outOfBagError).get(), res.get(outOfBagErrorPerObservation).get(), nullptr,
                  res.get(outOfBagErrorR2).get(), nullptr, res.get(outOfBagErrorPrediction).get());
    services::Status s;
    dtrees::internal::FeatureTypes featTypes;
    DAAL_CHECK(featTypes.init(*x), ErrorMemoryAllocationFailed);
    dtrees::internal::IndexedFeatures indexedFeatures;

    if (method == hist)
    {
        if (!memSave)
        {
            BinParams prm(par.maxBins, par.minBinSize, par.binningStrategy);
            s = indexedFeatures.init<algorithmFPType, cpu>(*x, &featTypes, &prm);
            DAAL_CHECK_STATUS_VAR(s);

            if (indexedFeatures.maxNumIndices() <= 256)
                s = computeImpl<algorithmFPType, uint8_t, cpu, daal::algorithms::decision_forest::regression::internal::ModelImpl,
                                TrainBatchTask<algorithmFPType, uint8_t, hist, helper, HyperparameterType, cpu> >(
                    pHostApp, x, y, w, *static_cast<daal::algorithms::decision_forest::regression::internal::ModelImpl *>(&m), rd, par, 0, featTypes,
                    &indexedFeatures, hyperparameter);
            else if (indexedFeatures.maxNumIndices() <= 65536)
                s = computeImpl<algorithmFPType, uint16_t, cpu, daal::algorithms::decision_forest::regression::internal::ModelImpl,
                                TrainBatchTask<algorithmFPType, uint16_t, hist, helper, HyperparameterType, cpu> >(
                    pHostApp, x, y, w, *static_cast<daal::algorithms::decision_forest::regression::internal::ModelImpl *>(&m), rd, par, 0, featTypes,
                    &indexedFeatures, hyperparameter);
            else
                s = computeImpl<
                    algorithmFPType, dtrees::internal::IndexedFeatures::IndexType, cpu,
                    daal::algorithms::decision_forest::regression::internal::ModelImpl,
                    TrainBatchTask<algorithmFPType, dtrees::internal::IndexedFeatures::IndexType, hist, helper, HyperparameterType, cpu> >(
                    pHostApp, x, y, w, *static_cast<daal::algorithms::decision_forest::regression::internal::ModelImpl *>(&m), rd, par, 0, featTypes,
                    &indexedFeatures, hyperparameter);
        }
        else
            s = computeImpl<algorithmFPType, dtrees::internal::IndexedFeatures::IndexType, cpu,
                            daal::algorithms::decision_forest::regression::internal::ModelImpl,
                            TrainBatchTask<algorithmFPType, dtrees::internal::IndexedFeatures::IndexType, hist, helper, HyperparameterType, cpu> >(
                pHostApp, x, y, w, *static_cast<daal::algorithms::decision_forest::regression::internal::ModelImpl *>(&m), rd, par, 0, featTypes,
                nullptr, hyperparameter);
    }
    else
    {
        if (!memSave)
        {
            s = indexedFeatures.init<algorithmFPType, cpu>(*x, &featTypes);
            DAAL_CHECK_STATUS_VAR(s);
            s = computeImpl<
                algorithmFPType, dtrees::internal::IndexedFeatures::IndexType, cpu,
                daal::algorithms::decision_forest::regression::internal::ModelImpl,
                TrainBatchTask<algorithmFPType, dtrees::internal::IndexedFeatures::IndexType, defaultDense, helper, HyperparameterType, cpu> >(
                pHostApp, x, y, w, *static_cast<daal::algorithms::decision_forest::regression::internal::ModelImpl *>(&m), rd, par, 0, featTypes,
                &indexedFeatures, hyperparameter);
        }
        else
            s = computeImpl<
                algorithmFPType, dtrees::internal::IndexedFeatures::IndexType, cpu,
                daal::algorithms::decision_forest::regression::internal::ModelImpl,
                TrainBatchTask<algorithmFPType, dtrees::internal::IndexedFeatures::IndexType, defaultDense, helper, HyperparameterType, cpu> >(
                pHostApp, x, y, w, *static_cast<daal::algorithms::decision_forest::regression::internal::ModelImpl *>(&m), rd, par, 0, featTypes,
                nullptr, hyperparameter);
    }

    if (s.ok()) res.impl()->setEngine(rd.updatedEngine);
    return s;
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status RegressionTrainBatchKernel<algorithmFPType, method, cpu>::compute(HostAppIface * pHostApp, const NumericTable * x,
                                                                                   const NumericTable * y, const NumericTable * w,
                                                                                   decision_forest::regression::Model & m, Result & res,
                                                                                   const Parameter & par, const HyperparameterType * hp)
{
    services::Status s;
    if (hp)
    {
        hp->check(s);
        if (!s) return s;
    }

    if (par.splitter == decision_forest::training::SplitterMode::best)
    {
        s = computeForSpecificHelper<algorithmFPType, method, cpu,
                                     RespHelperBase<algorithmFPType, cpu, OrderedRespHelperBest<algorithmFPType, cpu> > >(
            pHostApp, x, y, w, m, res, par, par.memorySavingMode, hp);
    }
    else if (par.splitter == decision_forest::training::SplitterMode::random)
    {
        s = computeForSpecificHelper<algorithmFPType, method, cpu,
                                     RespHelperBase<algorithmFPType, cpu, OrderedRespHelperRandom<algorithmFPType, cpu> > >(
            pHostApp, x, y, w, m, res, par, par.memorySavingMode || method == defaultDense, hp);
    }
    return s;
}

} /* namespace internal */
} /* namespace training */
} /* namespace regression */
} /* namespace decision_forest */
} /* namespace algorithms */
} /* namespace daal */

#endif
