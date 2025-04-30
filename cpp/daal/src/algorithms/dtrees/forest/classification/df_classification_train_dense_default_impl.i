/* file: df_classification_train_dense_default_impl.i */
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
//  Implementation of auxiliary functions for decision forest classification
//  (defaultDense) method.
//--
*/

#ifndef __DF_CLASSIFICATION_TRAIN_DENSE_DEFAULT_IMPL_I__
#define __DF_CLASSIFICATION_TRAIN_DENSE_DEFAULT_IMPL_I__

#include "src/algorithms/dtrees/forest/df_train_dense_default_impl.i"
#include "src/algorithms/dtrees/forest/classification/df_classification_train_kernel.h"
#include "src/algorithms/dtrees/forest/classification/df_classification_model_impl.h"
#include "src/algorithms/dtrees/dtrees_predict_dense_default_impl.i"
#include "src/algorithms/dtrees/forest/classification/df_classification_training_types_result.h"

#define OOBClassificationData size_t

using namespace daal::algorithms::decision_forest::training::internal;
using namespace daal::algorithms::dtrees::internal;
using namespace daal::algorithms::dtrees::training::internal;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace classification
{
namespace training
{
namespace internal
{
//////////////////////////////////////////////////////////////////////////////////////////
// UnorderedRespHelperBest class for best splitting classification
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class UnorderedRespHelperBest : public DataHelper<algorithmFPType, ClassIndexType, cpu>
{
public:
    typedef DataHelper<algorithmFPType, ClassIndexType, cpu> super;
    typedef typename dtrees::internal::TVector<float, cpu, dtrees::internal::ScalableAllocator<cpu> >
        Histogramm; //not sure why this is hard-coded to float and not algorithmFPType

    struct ImpurityData
    {
        double var; //impurity is a variance
        Histogramm hist;

        ImpurityData() {}
        ImpurityData(size_t nClasses) : hist(nClasses), var(0) {}
        algorithmFPType value() const { return var; }
        void init(size_t nClasses)
        {
            var = 0;
            hist.resize(nClasses, 0);
        }
    };

    typedef SplitData<algorithmFPType, ImpurityData> TSplitData;

public:
    UnorderedRespHelperBest(const dtrees::internal::IndexedFeatures * indexedFeatures, size_t nClasses,
                            const classification::training::internal::Hyperparameter * hp = nullptr)
        : super(indexedFeatures), _nClasses(nClasses), _histLeft(nClasses), _impLeft(nClasses), _impRight(nClasses)
    {
        DAAL_INT64 nClassesThreshold = classification::training::internal::MAX_SMALL_N_CLASSES;
        if (hp != nullptr)
        {
            hp->find(classification::training::internal::smallNClassesThreshold, nClassesThreshold);
        }
        _nClassesThreshold = nClassesThreshold;
    }

    int findSplitByHistDefault(int nDiffFeatMax, size_t n, size_t nMinSplitPart, const ImpurityData & curImpurity, TSplitData & split,
                               const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights, const IndexType iFeature) const;

    template <int K, bool noWeights>
    int findSplitFewClasses(int nDiffFeatMax, size_t n, size_t nMinSplitPart, const ImpurityData & curImpurity, TSplitData & split,
                            const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights, const IndexType iFeature) const;

    template <bool noWeights>
    bool findSplitOrderedFeature(const algorithmFPType * featureVal, const IndexType * aIdx, size_t n, size_t nMinSplitPart,
                                 const algorithmFPType accuracy, const ImpurityData & curImpurity, TSplitData & split,
                                 const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights) const;

    bool findSplitCategoricalFeature(const algorithmFPType * featureVal, const IndexType * aIdx, size_t n, size_t nMinSplitPart,
                                     const algorithmFPType accuracy, const ImpurityData & curImpurity, TSplitData & split,
                                     const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights) const;

protected: //enables specific functions for UnorderedRespHelperBest
    // Calculate impurity for right child
    static void updateRightImpurity(ImpurityData & imp, ClassIndexType iClass, double totalWeights, double moveWeights)
    {
        double delta = (2. * totalWeights - moveWeights) * imp.var + 2. * (imp.hist[iClass] - totalWeights);
        imp.var      = isZero<double, cpu>((totalWeights - moveWeights) * (totalWeights - moveWeights)) ?
                           1. :
                           (imp.var + moveWeights * delta / ((totalWeights - moveWeights) * (totalWeights - moveWeights)));
        imp.hist[iClass] -= moveWeights;
    }

    // Calculate impurity for left and right childs
    static void updateImpurity(ImpurityData & left, ImpurityData & right, ClassIndexType iClass, double totalWeights, double startWeights,
                               double & moveWeights)
    {
        double tmp = startWeights * (2. * moveWeights + left.var * startWeights) - 2. * moveWeights * left.hist[iClass];
        // Update impurity for left child
        left.hist[iClass] += moveWeights;
        left.var = isZero<algorithmFPType, cpu>((startWeights + moveWeights) * (startWeights + moveWeights)) ?
                       1. :
                       (tmp / ((startWeights + moveWeights) * (startWeights + moveWeights)));
        // Update impurity for right child
        updateRightImpurity(right, iClass, totalWeights - startWeights, moveWeights);
        moveWeights = 0.;
    }

    void calcGini(double totalWeights, ImpurityData & imp) const
    {
        const double sqWeights = totalWeights * totalWeights;
        const double one       = double(1);
        const double cDiv      = isZero<double, cpu>(sqWeights) ? one : (one / sqWeights);
        double var             = one;
        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < _nClasses; ++i) var -= cDiv * double(imp.hist[i]) * double(imp.hist[i]);
        imp.var = var;
        if (!isPositive<double, cpu>(imp.var)) imp.var = 0; //roundoff error
    }

protected:
    const size_t _nClasses;
    //set of buffers for indexed features processing, used in findBestSplitForFeatureIndexed only
    size_t _nClassesThreshold;
    mutable TVector<IndexType, cpu> _idxFeatureBuf;
    mutable TVector<algorithmFPType, cpu> _weightsFeatureBuf;
    mutable TVector<float, cpu> _samplesPerClassBuf;
    mutable Histogramm _histLeft;
    //work variables used in memory saving mode only
    mutable ImpurityData _impLeft;
    mutable ImpurityData _impRight;

protected:
    size_t nClasses() const { return this->_nClasses; }
};

template <typename algorithmFPType, CpuType cpu>
int UnorderedRespHelperBest<algorithmFPType, cpu>::findSplitByHistDefault(int nDiffFeatMax, size_t n, size_t nMinSplitPart,
                                                                          const ImpurityData & curImpurity, TSplitData & split,
                                                                          const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights,
                                                                          const IndexType iFeature) const
{
    auto nFeatIdx         = _idxFeatureBuf.get();
    auto featWeights      = _weightsFeatureBuf.get();
    auto nSamplesPerClass = _samplesPerClassBuf.get();

    algorithmFPType bestImpDecrease =
        split.impurityDecrease < 0 ? split.impurityDecrease : totalWeights * (split.impurityDecrease + algorithmFPType(1.) - curImpurity.var);

    //init histogram for the left part
    _histLeft.setAll(0);
    auto histLeft               = _histLeft.get();
    size_t nLeft                = 0;
    algorithmFPType leftWeights = 0.;
    int idxFeatureBestSplit     = -1; //index of best feature value in the array of sorted feature values
    for (size_t i = 0; i < nDiffFeatMax; ++i)
    {
        if (!nFeatIdx[i]) continue;
        algorithmFPType thisFeatWeights = featWeights[i];

        nLeft       = (split.featureUnordered ? nFeatIdx[i] : nLeft + nFeatIdx[i]);
        leftWeights = (split.featureUnordered ? thisFeatWeights : leftWeights + thisFeatWeights);
        if ((nLeft == n) //last split
            || ((n - nLeft) < nMinSplitPart) || ((totalWeights - leftWeights) < minWeightLeaf))
            break;

        if (!split.featureUnordered)
        {
            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t iClass = 0; iClass < _nClasses; ++iClass) histLeft[iClass] += nSamplesPerClass[i * _nClasses + iClass];
        }
        if ((nLeft < nMinSplitPart) || leftWeights < minWeightLeaf) continue;

        if (split.featureUnordered)
        {
            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            //one against others
            for (size_t iClass = 0; iClass < _nClasses; ++iClass) histLeft[iClass] = nSamplesPerClass[i * _nClasses + iClass];
        }

        auto histTotal           = curImpurity.hist.get();
        algorithmFPType sumLeft  = 0;
        algorithmFPType sumRight = 0;
        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        //proximal impurity improvement
        for (size_t iClass = 0; iClass < _nClasses; ++iClass)
        {
            sumLeft += histLeft[iClass] * histLeft[iClass];
            sumRight += (histTotal[iClass] - histLeft[iClass]) * (histTotal[iClass] - histLeft[iClass]);
        }

        const algorithmFPType decrease = sumLeft / leftWeights + sumRight / (totalWeights - leftWeights);
        if (decrease > bestImpDecrease)
        {
            split.left.hist     = _histLeft;
            split.left.var      = sumLeft;
            split.nLeft         = nLeft;
            split.leftWeights   = leftWeights;
            idxFeatureBestSplit = i;
            bestImpDecrease     = decrease;
        }
    }
    if (idxFeatureBestSplit >= 0)
    {
        split.impurityDecrease = curImpurity.var + bestImpDecrease / totalWeights - algorithmFPType(1);
        split.totalWeights     = totalWeights;
    }

    return idxFeatureBestSplit;
}

template <typename algorithmFPType, CpuType cpu>
template <int K, bool noWeights>
int UnorderedRespHelperBest<algorithmFPType, cpu>::findSplitFewClasses(int nDiffFeatMax, size_t n, size_t nMinSplitPart,
                                                                       const ImpurityData & curImpurity, TSplitData & split,
                                                                       const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights,
                                                                       const IndexType iFeature) const
{
    auto nSamplesPerClass = _samplesPerClassBuf.get();
    auto nFeatIdx         = _idxFeatureBuf.get();

    algorithmFPType bestImpDecrease =
        split.impurityDecrease < 0 ? split.impurityDecrease : totalWeights * (split.impurityDecrease + algorithmFPType(1.) - curImpurity.var);

    //init histogram for the left part
    _histLeft.setAll(0);
    auto histLeft               = _histLeft.get();
    size_t nLeft                = 0;
    algorithmFPType leftWeights = 0.;
    int idxFeatureBestSplit     = -1; //index of best feature value in the array of sorted feature values
    for (size_t i = 0; i < nDiffFeatMax; ++i)
    {
        algorithmFPType thisNFeatIdx(0);
        if (noWeights)
        {
            for (size_t iClass = 0; iClass < K; ++iClass)
            {
                thisNFeatIdx += nSamplesPerClass[i * K + iClass];
            }
        }

        else
        {
            thisNFeatIdx = nFeatIdx[i];
        }

        if (!thisNFeatIdx) continue;

        algorithmFPType thisFeatWeights(0);
        if (noWeights)
        {
            thisFeatWeights = thisNFeatIdx;
        }
        else
        {
            for (size_t iClass = 0; iClass < K; ++iClass)
            {
                thisFeatWeights += nSamplesPerClass[i * K + iClass];
            }
        }

        nLeft       = (split.featureUnordered ? thisNFeatIdx : nLeft + thisNFeatIdx);
        leftWeights = (split.featureUnordered ? thisFeatWeights : leftWeights + thisFeatWeights);
        if ((nLeft == n) //last split
            || ((n - nLeft) < nMinSplitPart) || ((totalWeights - leftWeights) < minWeightLeaf))
            break;

        if (!split.featureUnordered)
        {
            for (size_t iClass = 0; iClass < K; ++iClass) histLeft[iClass] += nSamplesPerClass[i * K + iClass];
        }
        if ((nLeft < nMinSplitPart) || leftWeights < minWeightLeaf) continue;

        if (split.featureUnordered)
        {
            for (size_t iClass = 0; iClass < K; ++iClass) histLeft[iClass] = nSamplesPerClass[i * K + iClass];
        }

        auto histTotal           = curImpurity.hist.get();
        algorithmFPType sumLeft  = 0;
        algorithmFPType sumRight = 0;

        //proximal impurity improvement
        for (size_t iClass = 0; iClass < K; ++iClass)
        {
            sumLeft += histLeft[iClass] * histLeft[iClass];
            sumRight += (histTotal[iClass] - histLeft[iClass]) * (histTotal[iClass] - histLeft[iClass]);
        }

        const algorithmFPType decrease = sumLeft / leftWeights + sumRight / (totalWeights - leftWeights);
        if (decrease > bestImpDecrease)
        {
            split.left.hist     = _histLeft;
            split.left.var      = sumLeft;
            split.nLeft         = nLeft;
            split.leftWeights   = leftWeights;
            idxFeatureBestSplit = i;
            bestImpDecrease     = decrease;
        }
    }
    if (idxFeatureBestSplit >= 0)
    {
        split.impurityDecrease = curImpurity.var + bestImpDecrease / totalWeights - algorithmFPType(1);
        split.totalWeights     = totalWeights;
    }

    return idxFeatureBestSplit;
}

template <typename algorithmFPType, CpuType cpu>
template <bool noWeights>
bool UnorderedRespHelperBest<algorithmFPType, cpu>::findSplitOrderedFeature(const algorithmFPType * featureVal, const IndexType * aIdx, size_t n,
                                                                            size_t nMinSplitPart, const algorithmFPType accuracy,
                                                                            const ImpurityData & curImpurity, TSplitData & split,
                                                                            const algorithmFPType minWeightLeaf, algorithmFPType totalWeights) const
{
    ClassIndexType iClass = this->_aResponse[aIdx[0]].val;
    _impLeft.init(_nClasses);
    _impRight = curImpurity;

    const bool bBestFromOtherFeatures      = isPositive<algorithmFPType, cpu>(split.impurityDecrease);
    algorithmFPType vBestFromOtherFeatures = algorithmFPType(-1);
    if (noWeights)
    {
        vBestFromOtherFeatures = bBestFromOtherFeatures ? algorithmFPType(n) * (curImpurity.var - split.impurityDecrease) : algorithmFPType(-1);
    }
    else
    {
        vBestFromOtherFeatures = bBestFromOtherFeatures ? totalWeights * (curImpurity.var - split.impurityDecrease) : algorithmFPType(-1);
    }

    bool bFound           = false;
    algorithmFPType vBest = algorithmFPType(-1);
    IndexType iBest       = -1;

    double nEqualRespValues      = this->_aWeights[aIdx[0]].val;
    double iStartEqualRespValues = double(0);
    algorithmFPType leftWeights  = algorithmFPType(0);
    const algorithmFPType last   = featureVal[n - nMinSplitPart];
    for (size_t i = 1; i < (n - nMinSplitPart + 1); ++i)
    {
        const algorithmFPType weights = this->_aWeights[aIdx[i]].val;
        const bool bSameFeaturePrev(featureVal[i] <= featureVal[i - 1] + accuracy);
        leftWeights += this->_aWeights[aIdx[i - 1]].val;
        if (bSameFeaturePrev || (i < nMinSplitPart) || (leftWeights < minWeightLeaf) || (totalWeights - leftWeights < minWeightLeaf))
        {
            //can't make a split
            //update impurity and continue
            if (iClass == this->_aResponse[aIdx[i]].val)
            {
                //prev response was the same
                nEqualRespValues += weights;
            }
            else
            {
                updateImpurity(_impLeft, _impRight, iClass, totalWeights, iStartEqualRespValues, nEqualRespValues);
#ifdef DEBUG_CHECK_IMPURITY
                checkImpurity(aIdx, leftWeights, _impLeft);
                checkImpurity(aIdx + i, totalWeights - leftWeights, _impRight);
#endif
                iClass                = this->_aResponse[aIdx[i]].val;
                nEqualRespValues      = weights;
                iStartEqualRespValues = leftWeights;
            }
            continue;
        }

        updateImpurity(_impLeft, _impRight, iClass, totalWeights, iStartEqualRespValues, nEqualRespValues);
#ifdef DEBUG_CHECK_IMPURITY
        checkImpurity(aIdx, leftWeights, _impLeft);
        checkImpurity(aIdx + i, totalWeights - leftWeights, _impRight);
#endif
        iClass                = this->_aResponse[aIdx[i]].val;
        nEqualRespValues      = weights;
        iStartEqualRespValues = leftWeights;
        if (!isPositive<algorithmFPType, cpu>(_impLeft.var)) _impLeft.var = 0;
        if (!isPositive<algorithmFPType, cpu>(_impRight.var)) _impRight.var = 0;

        const algorithmFPType v = leftWeights * _impLeft.var + (totalWeights - leftWeights) * _impRight.var;
        if (iBest < 0)
        {
            if (bBestFromOtherFeatures && isGreater<algorithmFPType, cpu>(v, vBestFromOtherFeatures))
            {
                if (featureVal[i] < last) continue;
                break;
            }
        }
        else if (isGreater<algorithmFPType, cpu>(v, vBest))
        {
            if (featureVal[i] < last) continue;
            break;
        }
        bFound             = true;
        vBest              = v;
        split.left.var     = _impLeft.var;
        split.left.hist    = _impLeft.hist;
        iBest              = i;
        split.nLeft        = i;
        split.leftWeights  = leftWeights;
        split.totalWeights = totalWeights;
        if (featureVal[i] < last) continue;
    }

    if (bFound)
    {
        DAAL_ASSERT(iBest > 0);
        const algorithmFPType impurityDecrease = curImpurity.var - vBest / totalWeights;
        DAAL_CHECK_STATUS_VAR(!(isZero<algorithmFPType, cpu>(impurityDecrease)));
        split.impurityDecrease = impurityDecrease;
#ifdef DEBUG_CHECK_IMPURITY
        checkImpurity(aIdx, split.nLeft, split.left);
#endif
        split.featureValue = featureVal[iBest - 1];
        split.iStart       = 0;
        DAAL_ASSERT(split.nLeft >= nMinSplitPart);
        DAAL_ASSERT((n - split.nLeft) >= nMinSplitPart);
        DAAL_ASSERT(split.leftWeights >= minWeightLeaf);
        DAAL_ASSERT((split.totalWeights - split.leftWeights) >= minWeightLeaf);
    }
    return bFound;
}

template <typename algorithmFPType, CpuType cpu>
bool UnorderedRespHelperBest<algorithmFPType, cpu>::findSplitCategoricalFeature(const algorithmFPType * featureVal, const IndexType * aIdx, size_t n,
                                                                                size_t nMinSplitPart, const algorithmFPType accuracy,
                                                                                const ImpurityData & curImpurity, TSplitData & split,
                                                                                const algorithmFPType minWeightLeaf,
                                                                                const algorithmFPType totalWeights) const
{
    DAAL_ASSERT(n >= 2 * nMinSplitPart);
    _impRight.init(_nClasses);
    bool bFound                       = false;
    const bool bBestFromOtherFeatures = !(split.impurityDecrease < 0);
    algorithmFPType vBest             = -1;
    IndexType iBest                   = -1;

    const algorithmFPType vBestFromOtherFeatures = bBestFromOtherFeatures ? totalWeights * (curImpurity.var - split.impurityDecrease) : -1;
    for (size_t i = 0; i < n - nMinSplitPart;)
    {
        _impLeft.init(_nClasses);
        auto weights                = this->_aWeights[aIdx[i]].val;
        size_t count                = 1;
        algorithmFPType leftWeights = weights;
        const algorithmFPType first = featureVal[i];
        ClassIndexType xi           = this->_aResponse[aIdx[i]].val;
        _impLeft.hist[xi]           = weights;
        const size_t iStart         = i;
        for (++i; (i < n) && (featureVal[i] == first); ++count, ++i)
        {
            weights = this->_aWeights[aIdx[i]].val;
            xi      = this->_aResponse[aIdx[i]].val;
            leftWeights += weights;
            _impLeft.hist[xi] += weights;
        }
        if ((count < nMinSplitPart) || ((n - count) < nMinSplitPart) || (leftWeights < minWeightLeaf)
            || ((totalWeights - leftWeights) < minWeightLeaf))
            continue;
        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < _nClasses; ++j) _impRight.hist[j] = curImpurity.hist[j] - _impLeft.hist[j];
        calcGini(leftWeights, _impLeft);
        calcGini(totalWeights - leftWeights, _impRight);
        const algorithmFPType v = leftWeights * _impLeft.var + (totalWeights - leftWeights) * _impRight.var;
        if (iBest < 0)
        {
            if (bBestFromOtherFeatures && isGreater<algorithmFPType, cpu>(v, vBestFromOtherFeatures)) continue;
        }
        else if (isGreater<algorithmFPType, cpu>(v, vBest))
            continue;
        iBest              = i;
        vBest              = v;
        split.left.var     = _impLeft.var;
        split.left.hist    = _impLeft.hist;
        split.nLeft        = count;
        split.leftWeights  = leftWeights;
        split.totalWeights = totalWeights;
        split.iStart       = iStart;
        split.featureValue = first;
        bFound             = true;
    }
    if (bFound)
    {
        const algorithmFPType impurityDecrease = curImpurity.var - vBest / totalWeights;
        DAAL_CHECK_STATUS_VAR(!(isZero<algorithmFPType, cpu>(impurityDecrease)));
        split.impurityDecrease = impurityDecrease;
        DAAL_ASSERT(split.nLeft >= nMinSplitPart);
        DAAL_ASSERT((n - split.nLeft) >= nMinSplitPart);
        DAAL_ASSERT(split.leftWeights >= minWeightLeaf);
        DAAL_ASSERT((split.totalWeights - split.leftWeights) >= minWeightLeaf);
    }
    return bFound;
}

//////////////////////////////////////////////////////////////////////////////////////////
// RespHelperBase contains common elements needed to select and split data
// Using CRTP, the base allows for certain templated functions to be polymorphic.
// Specifically, those functions which generates splits for features.
// It understands them through inheritance from UnorderedRespHelperBest, and is
// then has different versions in UnorderedRespHelperRandom.
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu, typename crtp>
class RespHelperBase : public UnorderedRespHelperBest<algorithmFPType, cpu>
{
public:
    typedef ClassIndexType TResponse;
    typedef typename dtrees::internal::TreeImpClassification<> TreeType;
    typedef typename TreeType::NodeType NodeType;
    typedef double intermSummFPType;
    using super        = UnorderedRespHelperBest<algorithmFPType, cpu>;
    using Histogramm   = typename super::Histogramm;
    using ImpurityData = typename super::ImpurityData;
    using TSplitData   = typename super::TSplitData;

    engines::internal::BatchBaseImpl * engineImpl;

public:
    RespHelperBase(const dtrees::internal::IndexedFeatures * indexedFeatures, size_t nClasses,
                   const classification::training::internal::Hyperparameter * hp = nullptr)
        : UnorderedRespHelperBest<algorithmFPType, cpu>(indexedFeatures, nClasses, hp)
    {}

    virtual bool init(const NumericTable * data, const NumericTable * resp, const IndexType * aSample,
                      const NumericTable * weights) DAAL_C11_OVERRIDE;
    void convertLeftImpToRight(size_t n, const ImpurityData & total, TSplitData & split)
    {
        computeRightHistogramm(total.hist, split.left.hist, split.left.hist);
        split.nLeft       = n - split.nLeft;
        split.leftWeights = split.totalWeights - split.leftWeights;
        this->calcGini(split.leftWeights, split.left);
    }

    template <bool noWeights>
    void calcImpurity(const IndexType * aIdx, size_t n, ImpurityData & imp, double & totalweights) const;

    bool findSplitForFeature(const algorithmFPType * featureVal, const IndexType * aIdx, size_t n, size_t nMinSplitPart,
                             const algorithmFPType accuracy, const ImpurityData & curImpurity, TSplitData & split,
                             const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights) const
    {
        const bool noWeights = !this->_weights;
        if (noWeights)
        {
            return split.featureUnordered ? static_cast<const crtp *>(this)->findSplitCategoricalFeature(
                       featureVal, aIdx, n, nMinSplitPart, accuracy, curImpurity, split, minWeightLeaf, totalWeights) :
                                            static_cast<const crtp *>(this)->template findSplitOrderedFeature<true>(
                                                featureVal, aIdx, n, nMinSplitPart, accuracy, curImpurity, split, minWeightLeaf, totalWeights);
        }
        else
        {
            return split.featureUnordered ? static_cast<const crtp *>(this)->findSplitCategoricalFeature(
                       featureVal, aIdx, n, nMinSplitPart, accuracy, curImpurity, split, minWeightLeaf, totalWeights) :
                                            static_cast<const crtp *>(this)->template findSplitOrderedFeature<false>(
                                                featureVal, aIdx, n, nMinSplitPart, accuracy, curImpurity, split, minWeightLeaf, totalWeights);
        }
    }
    bool terminateCriteria(ImpurityData & imp, algorithmFPType impurityThreshold, size_t nSamples) const { return imp.value() < impurityThreshold; }

    template <typename BinIndexType>
    int findSplitForFeatureSorted(algorithmFPType * featureBuf, IndexType iFeature, const IndexType * aIdx, size_t n, size_t nMinSplitPart,
                                  const ImpurityData & curImpurity, TSplitData & split, const algorithmFPType minWeightLeaf,
                                  const algorithmFPType totalWeights, const BinIndexType * binIndex) const;
    template <typename BinIndexType>
    void computeHistFewClassesWithoutWeights(IndexType iFeature, const IndexType * aIdx, const BinIndexType * binIndex, size_t n) const;
    template <typename BinIndexType>
    void computeHistFewClassesWithWeights(IndexType iFeature, const IndexType * aIdx, const BinIndexType * binIndex, size_t n) const;
    template <typename BinIndexType>
    void computeHistManyClasses(IndexType iFeature, const IndexType * aIdx, const BinIndexType * binIndex, size_t n) const;

    template <bool noWeights>
    int findSplitFewClassesDispatch(int nDiffFeatMax, size_t n, size_t nMinSplitPart, const ImpurityData & curImpurity, TSplitData & split,
                                    const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights, const IndexType iFeature) const;

    template <bool noWeights, typename BinIndexType>
    void finalizeBestSplit(const IndexType * aIdx, const BinIndexType * binIndex, size_t n, IndexType iFeature, size_t idxFeatureValueBestSplit,
                           TSplitData & bestSplit, IndexType * bestSplitIdx) const;
    void simpleSplit(const algorithmFPType * featureVal, const IndexType * aIdx, TSplitData & split) const;
    void singleSwap(const IndexType aIdx, TSplitData & split) const;

    TResponse predict(const dtrees::internal::Tree & t, const algorithmFPType * x) const
    {
        const typename TreeType::NodeType::Base * pNode = dtrees::prediction::internal::findNode<algorithmFPType, TreeType, cpu>(t, x);
        DAAL_ASSERT(pNode);
        return TreeType::NodeType::castLeaf(pNode)->response.value;
    }

    algorithmFPType predictionError(TResponse prediction, TResponse response) const { return algorithmFPType(prediction != response); }

    algorithmFPType predictionError(const dtrees::internal::Tree & t, const algorithmFPType * x, const NumericTable * resp, size_t iRow,
                                    byte * oobBuf) const
    {
        ReadRows<algorithmFPType, cpu> y(const_cast<NumericTable *>(resp), iRow, 1);
        const TResponse response(this->predict(t, x));
        if (oobBuf)
        {
            OOBClassificationData * ptr = ((OOBClassificationData *)oobBuf) + this->_nClasses * iRow;
            ptr[response]++;
        }
        return this->predictionError(response, *y.get());
    }

    void setLeafData(typename TreeType::NodeType::Leaf & node, const IndexType * idx, size_t n, ImpurityData & imp) const
    {
        DAAL_ASSERT(n > 0);
        node.count    = n;
        node.impurity = imp.var;
        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < this->_nClasses; ++i)
        {
            node.hist[i] = imp.hist[i];
        }
#ifdef DEBUG_CHECK_IMPURITY
        {
            Histogramm res(_nClasses, 0);
            for (size_t i = 0; i < n; ++i)
            {
                const ClassIndexType iClass = this->_aResponse[idx[i]].val;
                res[iClass] += 1;
            }
            for (size_t i = 0; i < _nClasses; ++i) DAAL_ASSERT(res[i] == imp.hist[i]);
        }
#endif
        auto maxVal             = imp.hist[0];
        ClassIndexType maxClass = 0;
        for (size_t i = 1; i < this->_nClasses; ++i)
        {
            if (maxVal < imp.hist[i])
            {
                maxVal   = imp.hist[i];
                maxClass = i;
            }
        }
        node.response.value = maxClass;
#ifdef KEEP_CLASSES_PROBABILITIIES
        node.response.size = imp.hist.size();
        node.response.hist = imp.hist.detach();
#endif
    }

#ifdef DEBUG_CHECK_IMPURITY
    void checkImpurity(const IndexType * ptrIdx, size_t n, const ImpurityData & expected) const;
#endif

protected:
    size_t nClasses() const
    {
        return this->_nClasses;
    }

    void computeRightHistogramm(const Histogramm & total, const Histogramm & left, Histogramm & right) const
    {
        auto histTotal = total.get();
        auto histRight = right.get();
        auto histLeft  = left.get();
        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t iClass = 0; iClass < this->_nClasses; ++iClass) histRight[iClass] = histTotal[iClass] - histLeft[iClass];
    }
};

#ifdef DEBUG_CHECK_IMPURITY
template <typename algorithmFPType, CpuType cpu, typename crtp>
void RespHelperBase<algorithmFPType, cpu, crtp>::checkImpurity(const IndexType * ptrIdx, algorithmFPType totalWeights,
                                                               const ImpurityData & expected) const
{
    Histogramm hist;
    hist.resize(_nClasses, 0);
    const algorithmFPType cDiv = isZero<algorithmFPType, cpu>(totalWeights * totalWeights) ? 1. : (1. / (totalWeights * totalWeights));
    algorithmFPType var(1.);
    for (size_t i = 0; i < _nClasses; ++i) var -= cDiv * algorithmFPType(hist[i]) * algorithmFPType(hist[i]);
    for (size_t i = 0; i < _nClasses; ++i) DAAL_ASSERT(hist[i] == expected.hist[i]);
    DAAL_ASSERT(!(fabs(var - expected.var) > 0.001));
}
#endif

template <typename algorithmFPType, CpuType cpu, typename crtp>
bool RespHelperBase<algorithmFPType, cpu, crtp>::init(const NumericTable * data, const NumericTable * resp, const IndexType * aSample,
                                                      const NumericTable * weights)
{
    DAAL_CHECK_STATUS_VAR(super::super::init(data, resp, aSample, weights)); //super::super is DataHelper
    if (this->_indexedFeatures)
    {
        //init work buffers for the computation using indexed features
        const auto nDiffFeatMax = this->indexedFeatures().maxNumIndices();
        this->_idxFeatureBuf.reset(nDiffFeatMax);
        this->_weightsFeatureBuf.reset(nDiffFeatMax);
        this->_samplesPerClassBuf.reset(nClasses() * nDiffFeatMax);
        return this->_idxFeatureBuf.get() && this->_weightsFeatureBuf.get() && this->_samplesPerClassBuf.get();
    }
    return true;
}

template <typename algorithmFPType, CpuType cpu, typename crtp>
template <bool noWeights>
void RespHelperBase<algorithmFPType, cpu, crtp>::calcImpurity(const IndexType * aIdx, size_t n, ImpurityData & imp, double & totalWeights) const
{
    imp.init(this->_nClasses);
    if (noWeights)
    {
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < n; ++i)
        {
            const ClassIndexType iClass = this->_aResponse[aIdx[i]].val;
            imp.hist[iClass] += algorithmFPType(1);
        }
        totalWeights = double(n);
    }
    else
    {
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < n; ++i)
        {
            const ClassIndexType iClass = this->_aResponse[aIdx[i]].val;
            imp.hist[iClass] += this->_aWeights[aIdx[i]].val;
            totalWeights += this->_aWeights[aIdx[i]].val;
        }
    }
    this->calcGini(totalWeights, imp);
}

template <typename algorithmFPType, CpuType cpu, typename crtp>
void RespHelperBase<algorithmFPType, cpu, crtp>::simpleSplit(const algorithmFPType * featureVal, const IndexType * aIdx, TSplitData & split) const
{
    split.left.init(this->_nClasses);
    const ClassIndexType iClass = this->_aResponse[aIdx[0]].val;
    split.featureValue          = featureVal[0];
    split.iStart                = 0;
    split.left.hist[iClass]     = this->_aWeights[aIdx[0]].val;
    split.nLeft                 = 1;
    split.leftWeights           = this->_aWeights[aIdx[0]].val;
    split.totalWeights          = this->_aWeights[aIdx[0]].val + this->_aWeights[aIdx[1]].val;
}

template <typename algorithmFPType, CpuType cpu, typename crtp>
void RespHelperBase<algorithmFPType, cpu, crtp>::singleSwap(const IndexType aIdx, TSplitData & split) const
{
    split.left.init(this->_nClasses);
    const ClassIndexType iClass = this->_aResponse[aIdx].val;
    split.left.hist[iClass]     = this->_aWeights[aIdx].val;
    split.nLeft                 = 1;
    split.leftWeights           = this->_aWeights[aIdx].val;
}

template <typename algorithmFPType, CpuType cpu, typename crtp>
template <typename BinIndexType>
void RespHelperBase<algorithmFPType, cpu, crtp>::computeHistFewClassesWithoutWeights(IndexType iFeature, const IndexType * aIdx,
                                                                                     const BinIndexType * binIndex, size_t n) const
{
    const algorithmFPType one(1.0);
    const auto aResponse  = this->_aResponse.get();
    auto nSamplesPerClass = this->_samplesPerClassBuf.get();
    {
        for (size_t i = 0; i < n; ++i)
        {
            const IndexType iSample = aIdx[i];
            const auto & r          = aResponse[aIdx[i]];

            const BinIndexType idx      = binIndex[r.idx];
            const ClassIndexType iClass = r.val;
            nSamplesPerClass[idx * this->_nClasses + iClass] += one;
        }
    }
}

template <typename algorithmFPType, CpuType cpu, typename crtp>
template <typename BinIndexType>
void RespHelperBase<algorithmFPType, cpu, crtp>::computeHistFewClassesWithWeights(IndexType iFeature, const IndexType * aIdx,
                                                                                  const BinIndexType * binIndex, size_t n) const
{
    const auto aResponse = this->_aResponse.get();
    const auto aWeights  = this->_aWeights.get();

    auto nFeatIdx         = this->_idxFeatureBuf.get();
    auto nSamplesPerClass = this->_samplesPerClassBuf.get();

    {
        for (size_t i = 0; i < n; ++i)
        {
            const IndexType iSample = aIdx[i];
            const auto & r          = aResponse[aIdx[i]];
            const BinIndexType idx  = binIndex[r.idx];
            ++nFeatIdx[idx];
            const auto weights          = aWeights[iSample].val;
            const ClassIndexType iClass = r.val;
            nSamplesPerClass[idx * this->_nClasses + iClass] += weights;
        }
    }
}

template <typename algorithmFPType, CpuType cpu, typename crtp>
template <typename BinIndexType>
void RespHelperBase<algorithmFPType, cpu, crtp>::computeHistManyClasses(IndexType iFeature, const IndexType * aIdx, const BinIndexType * binIndex,
                                                                        size_t n) const
{
    const auto aResponse = this->_aResponse.get();
    const auto aWeights  = this->_aWeights.get();

    auto nFeatIdx         = this->_idxFeatureBuf.get();
    auto featWeights      = this->_weightsFeatureBuf.get();
    auto nSamplesPerClass = this->_samplesPerClassBuf.get();

    {
        for (size_t i = 0; i < n; ++i)
        {
            const IndexType iSample = aIdx[i];
            const auto & r          = aResponse[aIdx[i]];
            const BinIndexType idx  = binIndex[r.idx];
            ++nFeatIdx[idx];
            const auto weights          = aWeights[iSample].val;
            const ClassIndexType iClass = r.val;
            featWeights[idx] += weights; //use for calculate leftWeights
            nSamplesPerClass[idx * this->_nClasses + iClass] += weights;
        }
    }
}

template <typename algorithmFPType, CpuType cpu, typename crtp>
template <bool noWeights>
int RespHelperBase<algorithmFPType, cpu, crtp>::findSplitFewClassesDispatch(int nDiffFeatMax, size_t n, size_t nMinSplitPart,
                                                                            const ImpurityData & curImpurity, TSplitData & split,
                                                                            const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights,
                                                                            const IndexType iFeature) const
{
    DAAL_ASSERT(this->_nClasses <= this->_nClassesThreshold);
    switch (this->_nClasses)
    {
    case 2:
        return static_cast<const crtp *>(this)->template findSplitFewClasses<2, noWeights>(nDiffFeatMax, n, nMinSplitPart, curImpurity, split,
                                                                                           minWeightLeaf, totalWeights, iFeature);
    case 3:
        return static_cast<const crtp *>(this)->template findSplitFewClasses<3, noWeights>(nDiffFeatMax, n, nMinSplitPart, curImpurity, split,
                                                                                           minWeightLeaf, totalWeights, iFeature);
    case 4:
        return static_cast<const crtp *>(this)->template findSplitFewClasses<4, noWeights>(nDiffFeatMax, n, nMinSplitPart, curImpurity, split,
                                                                                           minWeightLeaf, totalWeights, iFeature);
    case 5:
        return static_cast<const crtp *>(this)->template findSplitFewClasses<5, noWeights>(nDiffFeatMax, n, nMinSplitPart, curImpurity, split,
                                                                                           minWeightLeaf, totalWeights, iFeature);
    case 6:
        return static_cast<const crtp *>(this)->template findSplitFewClasses<6, noWeights>(nDiffFeatMax, n, nMinSplitPart, curImpurity, split,
                                                                                           minWeightLeaf, totalWeights, iFeature);
    case 7:
        return static_cast<const crtp *>(this)->template findSplitFewClasses<7, noWeights>(nDiffFeatMax, n, nMinSplitPart, curImpurity, split,
                                                                                           minWeightLeaf, totalWeights, iFeature);
    case 8:
        return static_cast<const crtp *>(this)->template findSplitFewClasses<8, noWeights>(nDiffFeatMax, n, nMinSplitPart, curImpurity, split,
                                                                                           minWeightLeaf, totalWeights, iFeature);
    }
    return -1;
}

template <typename algorithmFPType, CpuType cpu, typename crtp>
template <typename BinIndexType>
int RespHelperBase<algorithmFPType, cpu, crtp>::findSplitForFeatureSorted(algorithmFPType * featureBuf, IndexType iFeature, const IndexType * aIdx,
                                                                          size_t n, size_t nMinSplitPart, const ImpurityData & curImpurity,
                                                                          TSplitData & split, const algorithmFPType minWeightLeaf,
                                                                          const algorithmFPType totalWeights, const BinIndexType * binIndex) const
{
    const auto nDiffFeatMax = this->indexedFeatures().numIndices(iFeature);
    this->_samplesPerClassBuf.setValues(nClasses() * nDiffFeatMax, 0);

    int idxFeatureBestSplit = -1; //index of best feature value in the array of sorted feature values

    if (this->_nClasses <= this->_nClassesThreshold)
    {
        if (!this->_weights)
        {
            // nSamplesPerClass - computed. nFeatIdx and featWeights - no
            computeHistFewClassesWithoutWeights(iFeature, aIdx, binIndex, n);
            idxFeatureBestSplit =
                findSplitFewClassesDispatch<true>(nDiffFeatMax, n, nMinSplitPart, curImpurity, split, minWeightLeaf, totalWeights, iFeature);
        }
        else
        {
            // nSamplesPerClass and nFeatIdx - computed, featWeights - no
            this->_idxFeatureBuf.setValues(nDiffFeatMax, algorithmFPType(0));
            computeHistFewClassesWithWeights(iFeature, aIdx, binIndex, n);
            idxFeatureBestSplit =
                findSplitFewClassesDispatch<false>(nDiffFeatMax, n, nMinSplitPart, curImpurity, split, minWeightLeaf, totalWeights, iFeature);
        }
    }
    else
    {
        // nSamplesPerClass, nFeatIdx and featWeights - computed
        this->_weightsFeatureBuf.setValues(nDiffFeatMax, algorithmFPType(0));
        this->_idxFeatureBuf.setValues(nDiffFeatMax, algorithmFPType(0));
        computeHistManyClasses(iFeature, aIdx, binIndex, n);
        idxFeatureBestSplit = static_cast<const crtp *>(this)->findSplitByHistDefault(nDiffFeatMax, n, nMinSplitPart, curImpurity, split,
                                                                                      minWeightLeaf, totalWeights, iFeature);
    }

    return idxFeatureBestSplit;
}

template <typename algorithmFPType, CpuType cpu, typename crtp>
template <bool noWeights, typename BinIndexType>
void RespHelperBase<algorithmFPType, cpu, crtp>::finalizeBestSplit(const IndexType * aIdx, const BinIndexType * binIndex, size_t n,
                                                                   IndexType iFeature, size_t idxFeatureValueBestSplit, TSplitData & bestSplit,
                                                                   IndexType * bestSplitIdx) const
{
    DAAL_ASSERT(bestSplit.nLeft > 0);
    DAAL_ASSERT(bestSplit.leftWeights > 0.);
    algorithmFPType divL = algorithmFPType(1);
    if (noWeights)
    {
        divL = algorithmFPType(1) / algorithmFPType(bestSplit.nLeft);
    }
    else
    {
        divL = isZero<algorithmFPType, cpu>(bestSplit.leftWeights) ? algorithmFPType(1.) : (algorithmFPType(1.) / bestSplit.leftWeights);
    }
    bestSplit.left.var            = 1. - bestSplit.left.var * divL * divL; // Gini node impurity
    IndexType * bestSplitIdxRight = bestSplitIdx + bestSplit.nLeft;
    size_t iLeft                  = 0;
    size_t iRight                 = 0;
    int iRowSplitVal              = -1;
    int iNext                     = -1;
    int idxNext                   = this->_aResponse.size() - 1;
    const auto aResponse          = this->_aResponse.get();
    for (size_t i = 0; i < n; ++i)
    {
        const IndexType iSample = aIdx[i];
        const BinIndexType idx  = binIndex[aResponse[iSample].idx];
        if ((bestSplit.featureUnordered && (idx != idxFeatureValueBestSplit)) || ((!bestSplit.featureUnordered) && (idx > idxFeatureValueBestSplit)))
        {
            DAAL_ASSERT(iRight < n - bestSplit.nLeft);
            bestSplitIdxRight[iRight++] = iSample;
        }
        else
        {
            if (idx == idxFeatureValueBestSplit) iRowSplitVal = aResponse[iSample].idx;
            DAAL_ASSERT(iLeft < bestSplit.nLeft);
            bestSplitIdx[iLeft++] = iSample;
        }
        if ((idx > idxFeatureValueBestSplit) && (idxNext > idx))
        {
            idxNext = idx;
            iNext   = aResponse[iSample].idx;
        }
    }
    DAAL_ASSERT(iRight == n - bestSplit.nLeft);
    DAAL_ASSERT(iLeft == bestSplit.nLeft);
    bestSplit.iStart = 0;
    DAAL_ASSERT(iRowSplitVal >= 0);
    if (idxNext == this->_aResponse.size() - 1) iNext = iRowSplitVal;
    if (iNext >= 0 && iRowSplitVal >= 0)
    {
        bestSplit.featureValue = (this->getValue(iFeature, iRowSplitVal) + this->getValue(iFeature, iNext)) / (algorithmFPType)2.;
        if (bestSplit.featureValue == this->getValue(iFeature, iNext)) bestSplit.featureValue = this->getValue(iFeature, iRowSplitVal);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
// UnorderedRespHelperRandom class for random splitting classification
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class UnorderedRespHelperRandom : public RespHelperBase<algorithmFPType, cpu, UnorderedRespHelperRandom<algorithmFPType, cpu> >
{
public:
    using Histogramm   = typename RespHelperBase<algorithmFPType, cpu, UnorderedRespHelperRandom<algorithmFPType, cpu> >::Histogramm;
    using ImpurityData = typename RespHelperBase<algorithmFPType, cpu, UnorderedRespHelperRandom<algorithmFPType, cpu> >::ImpurityData;
    using TSplitData   = typename RespHelperBase<algorithmFPType, cpu, UnorderedRespHelperRandom<algorithmFPType, cpu> >::TSplitData;

public:
    UnorderedRespHelperRandom(const dtrees::internal::IndexedFeatures * indexedFeatures, size_t nClasses,
                              const classification::training::internal::Hyperparameter * hp = nullptr)
        : RespHelperBase<algorithmFPType, cpu, UnorderedRespHelperRandom<algorithmFPType, cpu> >(indexedFeatures, nClasses, hp)
    {}

    size_t genRandomBinIdx(const IndexType iFeature, const size_t minidx, const size_t maxidx) const;

    int findSplitByHistDefault(int nDiffFeatMax, size_t n, size_t nMinSplitPart, const ImpurityData & curImpurity, TSplitData & split,
                               const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights, const IndexType iFeature) const;

    template <int K, bool noWeights>
    int findSplitFewClasses(int nDiffFeatMax, size_t n, size_t nMinSplitPart, const ImpurityData & curImpurity, TSplitData & split,
                            const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights, const IndexType iFeature) const;

    template <bool noWeights>
    bool findSplitOrderedFeature(const algorithmFPType * featureVal, const IndexType * aIdx, size_t n, size_t nMinSplitPart,
                                 const algorithmFPType accuracy, const ImpurityData & curImpurity, TSplitData & split,
                                 const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights) const;

    bool findSplitCategoricalFeature(const algorithmFPType * featureVal, const IndexType * aIdx, size_t n, size_t nMinSplitPart,
                                     const algorithmFPType accuracy, const ImpurityData & curImpurity, TSplitData & split,
                                     const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights) const;
};

template <typename algorithmFPType, CpuType cpu>
size_t UnorderedRespHelperRandom<algorithmFPType, cpu>::genRandomBinIdx(const IndexType iFeature, const size_t minidx, const size_t maxidx) const
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
int UnorderedRespHelperRandom<algorithmFPType, cpu>::findSplitByHistDefault(int nDiffFeatMax, size_t n, size_t nMinSplitPart,
                                                                            const ImpurityData & curImpurity, TSplitData & split,
                                                                            const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights,
                                                                            const IndexType iFeature) const
{
    auto nFeatIdx         = this->_idxFeatureBuf.get();
    auto featWeights      = this->_weightsFeatureBuf.get();
    auto nSamplesPerClass = this->_samplesPerClassBuf.get();

    algorithmFPType bestImpDecrease =
        split.impurityDecrease < 0 ? split.impurityDecrease : totalWeights * (split.impurityDecrease + algorithmFPType(1.) - curImpurity.var);

    //init histogram for the left part
    this->_histLeft.setAll(0);
    auto histLeft               = this->_histLeft.get();
    size_t nLeft                = 0;
    algorithmFPType leftWeights = 0.;
    int idxFeatureBestSplit     = -1; //index of best feature value in the array of sorted feature values

    size_t minidx = 0;
    size_t maxidx = nDiffFeatMax - 1;
    size_t idx;

    while ((minidx < maxidx) && isZero<IndexType, cpu>(nFeatIdx[minidx])) minidx++;
    while ((minidx < maxidx) && isZero<IndexType, cpu>(nFeatIdx[maxidx])) maxidx--;

    DAAL_ASSERT(minidx < maxidx); //if the if statement after minidx search doesn't activate, we have an issue

    if ((nFeatIdx[minidx] == n) //last split
        || ((n - nFeatIdx[minidx]) < nMinSplitPart) || ((totalWeights - featWeights[minidx]) < minWeightLeaf))
        return idxFeatureBestSplit;

    if (split.featureUnordered)
    {
        //find random index between minidx and maxidx
        RNGsInst<size_t, cpu> rng;
        rng.uniform(1, &idx, this->engineImpl->getState(), minidx, maxidx);

        //iterate idx down for FinalizeBestSplit (since it splits leftward)
        while ((minidx < idx) && isZero<IndexType, cpu>(nFeatIdx[idx])) idx--;
        DAAL_ASSERT(!(isZero<IndexType, cpu>(nFeatIdx[idx])));

        nLeft       = nFeatIdx[idx];
        leftWeights = featWeights[idx];

        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        //one against others
        for (size_t iClass = 0; iClass < this->_nClasses; ++iClass) histLeft[iClass] = nSamplesPerClass[idx * this->_nClasses + iClass];
    }
    else
    {
        idx = this->genRandomBinIdx(iFeature, minidx, maxidx);

        //iterate idx down for FinalizeBestSplit (since it splits leftward)
        while ((minidx < idx) && isZero<IndexType, cpu>(nFeatIdx[idx])) idx--;
        DAAL_ASSERT(!(isZero<IndexType, cpu>(nFeatIdx[idx])));

        for (size_t i = minidx; i <= idx; ++i)
        {
            if (isZero<IndexType, cpu>(nFeatIdx[i])) continue;
            nLeft += nFeatIdx[i];
            leftWeights += featWeights[i];

            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t iClass = 0; iClass < this->_nClasses; ++iClass) histLeft[iClass] += nSamplesPerClass[i * this->_nClasses + iClass];
        }
    }

    if (!(((n - nLeft) < nMinSplitPart) || ((totalWeights - leftWeights) < minWeightLeaf) || (nLeft < nMinSplitPart)
          || (leftWeights < minWeightLeaf)))
    {
        auto histTotal           = curImpurity.hist.get();
        algorithmFPType sumLeft  = 0;
        algorithmFPType sumRight = 0;
        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        //proximal impurity improvement
        for (size_t iClass = 0; iClass < this->_nClasses; ++iClass)
        {
            sumLeft += histLeft[iClass] * histLeft[iClass];
            sumRight += (histTotal[iClass] - histLeft[iClass]) * (histTotal[iClass] - histLeft[iClass]);
        }

        const algorithmFPType decrease = sumLeft / leftWeights + sumRight / (totalWeights - leftWeights);
        if (decrease > bestImpDecrease)
        {
            split.left.hist     = this->_histLeft;
            split.left.var      = sumLeft;
            split.nLeft         = nLeft;
            split.leftWeights   = leftWeights;
            idxFeatureBestSplit = idx;
            bestImpDecrease     = decrease;
        }
    }
    if (idxFeatureBestSplit >= 0)
    {
        split.impurityDecrease = curImpurity.var + bestImpDecrease / totalWeights - algorithmFPType(1);
        split.totalWeights     = totalWeights;
    }

    return idxFeatureBestSplit;
}

template <typename algorithmFPType, CpuType cpu>
template <int K, bool noWeights>
int UnorderedRespHelperRandom<algorithmFPType, cpu>::findSplitFewClasses(int nDiffFeatMax, size_t n, size_t nMinSplitPart,
                                                                         const ImpurityData & curImpurity, TSplitData & split,
                                                                         const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights,
                                                                         const IndexType iFeature) const
{
    auto nSamplesPerClass = this->_samplesPerClassBuf.get();
    auto nFeatIdx         = this->_idxFeatureBuf.get();

    algorithmFPType bestImpDecrease =
        split.impurityDecrease < 0 ? split.impurityDecrease : totalWeights * (split.impurityDecrease + algorithmFPType(1.) - curImpurity.var);

    //init histogram for the left part
    this->_histLeft.setAll(0);
    auto histLeft = this->_histLeft.get();
    size_t nLeft  = 0;
    algorithmFPType leftWeights(0);
    algorithmFPType minWeights(0);
    int idxFeatureBestSplit = -1; //index of best feature value in the array of sorted feature values

    size_t minidx = 0;
    size_t maxidx = nDiffFeatMax;
    size_t idx;

    //solve for the min and max indices of the histogram with data
    //when it comes across a nonzero bin, it will run the next step in prepping histLeft and nLeft

    //solve for the min non-zero index
    if (noWeights)
    {
        algorithmFPType thisNFeatIdx(0);
        for (size_t iC = 0; iC < K; ++iC) thisNFeatIdx += nSamplesPerClass[iC];
        while ((minidx < maxidx) && isZero<algorithmFPType, cpu>(thisNFeatIdx))
        {
            minidx++;

            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t iClass = 0; iClass < K; ++iClass)
            {
                thisNFeatIdx += nSamplesPerClass[minidx * K + iClass];
            }
        }
        nLeft      = thisNFeatIdx;
        minWeights = nLeft;
    }
    else
    {
        IndexType thisNFeatIdx = nFeatIdx[0];
        while ((minidx < maxidx) && isZero<IndexType, cpu>(thisNFeatIdx)) thisNFeatIdx = nFeatIdx[++minidx];
        nLeft = thisNFeatIdx;

        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t iClass = 0; iClass < K; ++iClass)
        {
            minWeights += nSamplesPerClass[minidx * K + iClass];
        }
    }

    DAAL_ASSERT(minidx < maxidx);

    if ((nLeft == n) //last split
        || ((n - nLeft) < nMinSplitPart) || ((totalWeights - minWeights) < minWeightLeaf))
        return idxFeatureBestSplit;

    //set histLeft
    PRAGMA_FORCE_SIMD
    PRAGMA_VECTOR_ALWAYS
    for (size_t iClass = 0; iClass < K; ++iClass) histLeft[iClass] = nSamplesPerClass[minidx * K + iClass];

    //solve for the max non-zero index
    if (noWeights)
    {
        algorithmFPType thisNFeatIdx(0);
        while ((minidx < maxidx) && isZero<algorithmFPType, cpu>(thisNFeatIdx))
        {
            maxidx--;
            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t iClass = 0; iClass < K; ++iClass)
            {
                thisNFeatIdx += nSamplesPerClass[maxidx * K + iClass];
            }
        }
    }
    else
    {
        IndexType thisNFeatIdx(0);
        while ((minidx < maxidx) && isZero<IndexType, cpu>(thisNFeatIdx)) thisNFeatIdx = nFeatIdx[--maxidx];
    }

    DAAL_ASSERT(minidx < maxidx); //if the if statement after minidx search doesn't activate, we have an issue.

    //randomly select a histogram split index
    if (split.featureUnordered)
    {
        RNGsInst<size_t, cpu> rng;
        rng.uniform(1, &idx, this->engineImpl->getState(), minidx, maxidx);
    }
    else
    {
        idx = this->genRandomBinIdx(iFeature, minidx, maxidx);
    }

    if (noWeights)
    {
        //iterate idx down to a bin with values for FinalizeBestSplit
        algorithmFPType thisNFeatIdx(0);

        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t iC = 0; iC < K; ++iC) thisNFeatIdx += nSamplesPerClass[idx * K + iC];
        while ((minidx < idx) && isZero<algorithmFPType, cpu>(thisNFeatIdx))
        {
            idx--;
            for (size_t iClass = 0; iClass < K; ++iClass)
            {
                thisNFeatIdx += nSamplesPerClass[idx * K + iClass];
            }
        }

        DAAL_ASSERT(!(isZero<algorithmFPType, cpu>(thisNFeatIdx)))

        if (split.featureUnordered) //only need last index
        {
            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t iClass = 0; iClass < K; ++iClass) histLeft[iClass] = nSamplesPerClass[idx * K + iClass];
        }
        else //sum over all to idx
        {
            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t i = minidx + 1; i <= idx; i++)
            {
                for (size_t iClass = 0; iClass < K; ++iClass) histLeft[iClass] += nSamplesPerClass[i * K + iClass];
            }
        }

        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t iClass = 0; iClass < K; ++iClass)
            leftWeights += histLeft[iClass]; //histleft is forced to float, and may cause issues with algorithmFPType = double
        nLeft = leftWeights;
    }
    else
    {
        //iterate idx down to a bin with values for FinalizeBestSplit
        IndexType thisNFeatIdx = nFeatIdx[idx];
        while ((minidx < idx) && isZero<IndexType, cpu>(thisNFeatIdx)) thisNFeatIdx = nFeatIdx[--idx];

        DAAL_ASSERT(!(isZero<IndexType, cpu>(thisNFeatIdx)));

        if (split.featureUnordered) //only need last index
        {
            nLeft = nFeatIdx[idx];
            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t iClass = 0; iClass < K; ++iClass) histLeft[iClass] = nSamplesPerClass[idx * K + iClass];
        }
        else //sum over all to idx
        {
            for (size_t i = minidx + 1; i <= idx; i++)
            {
                nLeft += nFeatIdx[i];
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t iClass = 0; iClass < K; ++iClass) histLeft[iClass] += nSamplesPerClass[i * K + iClass];
            }
        }

        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t iClass = 0; iClass < K; ++iClass) leftWeights += histLeft[iClass];
    }

    if (!(((n - nLeft) < nMinSplitPart) || ((totalWeights - leftWeights) < minWeightLeaf) || (nLeft < nMinSplitPart)
          || (leftWeights < minWeightLeaf)))
    {
        auto histTotal           = curImpurity.hist.get();
        algorithmFPType sumLeft  = 0;
        algorithmFPType sumRight = 0;

        //proximal impurity improvement
        for (size_t iClass = 0; iClass < K; ++iClass)
        {
            sumLeft += histLeft[iClass] * histLeft[iClass];
            sumRight += (histTotal[iClass] - histLeft[iClass]) * (histTotal[iClass] - histLeft[iClass]);
        }

        const algorithmFPType decrease = sumLeft / leftWeights + sumRight / (totalWeights - leftWeights);
        if (decrease > bestImpDecrease)
        {
            split.left.hist     = this->_histLeft;
            split.left.var      = sumLeft;
            split.nLeft         = nLeft;
            split.leftWeights   = leftWeights;
            idxFeatureBestSplit = idx;
            bestImpDecrease     = decrease;
        }
    }
    if (idxFeatureBestSplit >= 0)
    {
        split.impurityDecrease = curImpurity.var + bestImpDecrease / totalWeights - algorithmFPType(1);
        split.totalWeights     = totalWeights;
    }

    return idxFeatureBestSplit;
}

template <typename algorithmFPType, CpuType cpu>
template <bool noWeights>
bool UnorderedRespHelperRandom<algorithmFPType, cpu>::findSplitOrderedFeature(const algorithmFPType * featureVal, const IndexType * aIdx, size_t n,
                                                                              size_t nMinSplitPart, const algorithmFPType accuracy,
                                                                              const ImpurityData & curImpurity, TSplitData & split,
                                                                              const algorithmFPType minWeightLeaf, algorithmFPType totalWeights) const
{
    this->_impLeft.init(this->_nClasses);
    this->_impRight = curImpurity;

    const bool bBestFromOtherFeatures      = isPositive<algorithmFPType, cpu>(split.impurityDecrease);
    algorithmFPType vBestFromOtherFeatures = bBestFromOtherFeatures ? totalWeights * (curImpurity.var - split.impurityDecrease) : algorithmFPType(-1);

    bool bFound                 = false;
    IndexType iBest             = -1;
    algorithmFPType leftWeights = algorithmFPType(0);
    algorithmFPType v           = algorithmFPType(0);
    algorithmFPType idx;
    size_t i;

    //select random split index
    RNGsInst<algorithmFPType, cpu> rng;
    rng.uniform(1, &idx, this->engineImpl->getState(), featureVal[0],
                featureVal[n - 1]); //this strategy follows sklearn's implementation

    if (idx >= featureVal[n - nMinSplitPart]
        || idx < featureVal[nMinSplitPart - 1]) //check if sufficient samples will exist, and not a constant feature
    {
        return bFound;
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

    if (noWeights)
    {
        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (i = 0; i < r; ++i)
        {
            const ClassIndexType iClass = this->_aResponse[aIdx[i]].val;
            this->_impLeft.hist[iClass] += algorithmFPType(1);
        }
        leftWeights = i;
    }
    else
    {
        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (i = 0; i < r; ++i)
        {
            const ClassIndexType iClass = this->_aResponse[aIdx[i]].val;
            this->_impLeft.hist[iClass] += this->_aWeights[aIdx[i]].val;
            leftWeights += this->_aWeights[aIdx[i]].val;
        }
    }

    for (size_t iClass = 0; iClass < this->_nClasses; ++iClass)
    {
        this->_impRight.hist[iClass] -= this->_impLeft.hist[iClass];
    }

    this->calcGini(leftWeights, this->_impLeft);
    this->calcGini(totalWeights - leftWeights, this->_impRight);

#ifdef DEBUG_CHECK_IMPURITY
    checkImpurity(aIdx, leftWeights, this->_impLeft);
    checkImpurity(aIdx + i - 1, totalWeights - leftWeights, this->_impRight);
#endif

    if ((leftWeights >= minWeightLeaf) && ((totalWeights - leftWeights) >= minWeightLeaf)) //it is a valid split with enought leaf weights
    {
        //check if bFound condition below
        if (!isPositive<algorithmFPType, cpu>(this->_impLeft.var)) this->_impLeft.var = 0;   //set left impurity to 0 if negative
        if (!isPositive<algorithmFPType, cpu>(this->_impRight.var)) this->_impRight.var = 0; //set right impurity to 0 if negative

        v = leftWeights * this->_impLeft.var + (totalWeights - leftWeights) * this->_impRight.var; //calculate overall weighted Gini index

        if (!(bBestFromOtherFeatures
              && isGreater<algorithmFPType, cpu>(v, vBestFromOtherFeatures))) //if it has a better weighted gini overwite parameters
        {
            bFound             = true;
            split.left.var     = this->_impLeft.var;
            split.left.hist    = this->_impLeft.hist;
            iBest              = i; //because of the for loop i = idx + 1. (from the last i++)
            split.nLeft        = i; //meaning that nLeft is correct, and split.featureValue will be set correctly
            split.leftWeights  = leftWeights;
            split.totalWeights = totalWeights;
        }
    }

    if (bFound) //if new best found
    {
        DAAL_ASSERT(iBest > 0);
        const algorithmFPType impurityDecrease = curImpurity.var - v / totalWeights;
        DAAL_CHECK_STATUS_VAR(!(isZero<algorithmFPType, cpu>(impurityDecrease)));
        split.impurityDecrease = impurityDecrease;
#ifdef DEBUG_CHECK_IMPURITY
        checkImpurity(aIdx, split.nLeft, split.left);
#endif
        split.featureValue = idx;
        split.iStart       = 0;
        DAAL_ASSERT(split.nLeft >= nMinSplitPart);
        DAAL_ASSERT((n - split.nLeft) >= nMinSplitPart);
        DAAL_ASSERT(split.leftWeights >= minWeightLeaf);
        DAAL_ASSERT((split.totalWeights - split.leftWeights) >= minWeightLeaf);
    }
    return bFound;
}

template <typename algorithmFPType, CpuType cpu>
bool UnorderedRespHelperRandom<algorithmFPType, cpu>::findSplitCategoricalFeature(const algorithmFPType * featureVal, const IndexType * aIdx,
                                                                                  size_t n, size_t nMinSplitPart, const algorithmFPType accuracy,
                                                                                  const ImpurityData & curImpurity, TSplitData & split,
                                                                                  const algorithmFPType minWeightLeaf,
                                                                                  const algorithmFPType totalWeights) const
{
    DAAL_ASSERT(n >= 2 * nMinSplitPart);
    this->_impRight.init(this->_nClasses);
    bool bFound                       = false;
    const bool bBestFromOtherFeatures = !(split.impurityDecrease < 0);
    algorithmFPType vBest             = -1;
    IndexType iBest                   = -1;
    algorithmFPType min               = featureVal[0];
    algorithmFPType max               = featureVal[0];
    algorithmFPType idx;
    algorithmFPType first;

    for (size_t i = 1; i < n; ++i)
    {
        max = featureVal[i] > max ? featureVal[i] : max;
        min = featureVal[i] < min ? featureVal[i] : min;
    }

    first = min;

    RNGsInst<algorithmFPType, cpu> rng;
    rng.uniform(1, &idx, this->engineImpl->getState(), min, max); //this strategy follows sklearn's implementation

    for (size_t i = 1; i < n; ++i)
    {
        first = featureVal[i] <= idx && featureVal[i] > first ? featureVal[i] : first;
    }
    //first is the closest categorical feature less than the idx O(n) computation as ordering of featureVal is unknown.

    const algorithmFPType vBestFromOtherFeatures = bBestFromOtherFeatures ? totalWeights * (curImpurity.var - split.impurityDecrease) : -1;
    for (size_t i = 0; i < n - nMinSplitPart;)
    {
        if (featureVal[i] != first)
        {
            i++;
            continue;
        }
        this->_impLeft.init(this->_nClasses);
        auto weights                = this->_aWeights[aIdx[i]].val;
        size_t count                = 1;
        algorithmFPType leftWeights = weights;
        const algorithmFPType first = featureVal[i];
        ClassIndexType xi           = this->_aResponse[aIdx[i]].val;
        this->_impLeft.hist[xi]     = weights;
        const size_t iStart         = i;
        //there is an ordering to categorical features shown here that isn't described
        //its not clear if featureVal[i] == first will occur at a later point
        //but the for loop assumes that they could be grouped together in the array
        for (++i; (i < n) && (featureVal[i] == first); ++count, ++i)
        {
            weights = this->_aWeights[aIdx[i]].val;
            xi      = this->_aResponse[aIdx[i]].val;
            leftWeights += weights;
            this->_impLeft.hist[xi] += weights;
        }
        if ((count < nMinSplitPart) || ((n - count) < nMinSplitPart) || (leftWeights < minWeightLeaf)
            || ((totalWeights - leftWeights) < minWeightLeaf))
            continue;
        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < this->_nClasses; ++j) this->_impRight.hist[j] = curImpurity.hist[j] - this->_impLeft.hist[j];
        this->calcGini(leftWeights, this->_impLeft);
        this->calcGini(totalWeights - leftWeights, this->_impRight);
        const algorithmFPType v = leftWeights * this->_impLeft.var + (totalWeights - leftWeights) * this->_impRight.var;

        if (iBest < 0)
        {
            if (bBestFromOtherFeatures && isGreater<algorithmFPType, cpu>(v, vBestFromOtherFeatures)) continue;
        }
        else if (isGreater<algorithmFPType, cpu>(v, vBest))
            continue;
        iBest              = i;
        vBest              = v;
        split.left.var     = this->_impLeft.var;
        split.left.hist    = this->_impLeft.hist;
        split.nLeft        = count;
        split.leftWeights  = leftWeights;
        split.totalWeights = totalWeights;
        split.iStart       = iStart;
        split.featureValue = first;
        bFound             = true;
    }
    if (bFound)
    {
        const algorithmFPType impurityDecrease = curImpurity.var - vBest / totalWeights;
        DAAL_CHECK_STATUS_VAR(!(isZero<algorithmFPType, cpu>(impurityDecrease)));
        split.impurityDecrease = impurityDecrease;
        DAAL_ASSERT(split.nLeft >= nMinSplitPart);
        DAAL_ASSERT((n - split.nLeft) >= nMinSplitPart);
        DAAL_ASSERT(split.leftWeights >= minWeightLeaf);
        DAAL_ASSERT((split.totalWeights - split.leftWeights) >= minWeightLeaf);
    }
    return bFound;
}

//////////////////////////////////////////////////////////////////////////////////////////
// TreeThreadCtx class for classification
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class TreeThreadCtx : public TreeThreadCtxBase<algorithmFPType, cpu>
{
public:
    typedef TreeThreadCtxBase<algorithmFPType, cpu> super;
    TreeThreadCtx(algorithmFPType * _varImp = nullptr) : super(_varImp), _nClasses(0) {}
    bool init(const decision_forest::training::Parameter & par, const NumericTable * x, size_t nClasses)
    {
        DAAL_CHECK_STATUS_VAR(super::init(par, x));
        _nClasses = nClasses;
        using namespace decision_forest::training;
        if (par.resultsToCompute
            & (computeOutOfBagError | computeOutOfBagErrorPerObservation | computeOutOfBagErrorR2 | computeOutOfBagErrorDecisionFunction))
        {
            size_t sz    = sizeof(OOBClassificationData) * nClasses * x->getNumberOfRows();
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
            OOBClassificationData * dst       = (OOBClassificationData *)other.oobBuf;
            const OOBClassificationData * src = (const OOBClassificationData *)this->oobBuf;
            const size_t n                    = _nClasses * nSamples;
            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t i = 0; i < n; ++i)
            {
                dst[i] += src[i];
            }
        }
    }
    Status finalizeOOBError(const NumericTable * resp, algorithmFPType * res, algorithmFPType * resPerObs, algorithmFPType * resAccuracy,
                            algorithmFPType * resR2, algorithmFPType * resDecisionFunction, algorithmFPType * resPrediction) const;

private:
    size_t _nClasses;
};

template <typename algorithmFPType, CpuType cpu>
Status TreeThreadCtx<algorithmFPType, cpu>::finalizeOOBError(const NumericTable * resp, algorithmFPType * res, algorithmFPType * resPerObs,
                                                             algorithmFPType * resAccuracy, algorithmFPType * resR2,
                                                             algorithmFPType * resDecisionFunction, algorithmFPType * resPrediction) const
{
    DAAL_ASSERT(this->oobBuf);
    const size_t nSamples = resp->getNumberOfRows();
    ReadRows<algorithmFPType, cpu> y(const_cast<NumericTable *>(resp), 0, nSamples);
    DAAL_CHECK_BLOCK_STATUS(y);
    Atomic<size_t> nPredicted(0);
    Atomic<size_t> nError(0);
    const algorithmFPType eps = services::internal::EpsilonVal<algorithmFPType>::get();
    daal::threader_for(nSamples, nSamples, [&](size_t i) {
        const OOBClassificationData * ptr = ((const OOBClassificationData *)this->oobBuf) + i * _nClasses;
        const size_t classLabel(y.get()[i]);
        size_t maxIdx                = 0;
        algorithmFPType sum          = static_cast<algorithmFPType>(ptr[0]);
        OOBClassificationData maxVal = ptr[0];
        for (size_t j = 1; j < _nClasses; ++j)
        {
            sum += static_cast<algorithmFPType>(ptr[j]);
            if (maxVal < ptr[j])
            {
                maxVal = ptr[j];
                maxIdx = j;
            }
        }
        if (resDecisionFunction)
        {
            for (size_t j = 0; j < _nClasses; ++j)
            {
                resDecisionFunction[i * _nClasses + j] =
                    static_cast<algorithmFPType>(ptr[j]) / services::internal::max<cpu, algorithmFPType>(sum, eps);
            }
        }
        if (maxVal == 0)
        {
            //was not in OOB set of any tree and hence not predicted
            if (resPerObs) resPerObs[i] = algorithmFPType(-1);
            return;
        }
        if (res || resAccuracy)
        {
            nPredicted.inc();
            if (maxIdx != classLabel) nError.inc();
        }
        if (resPerObs) resPerObs[i] = algorithmFPType(maxIdx != classLabel);
    });
    if (res) *res = nPredicted.get() ? algorithmFPType(nError.get()) / algorithmFPType(nPredicted.get()) : 0;
    if (resAccuracy)
        *resAccuracy = nPredicted.get() ? algorithmFPType(1) - algorithmFPType(nError.get()) / algorithmFPType(nPredicted.get()) : algorithmFPType(1);
    return Status();
}

//////////////////////////////////////////////////////////////////////////////////////////
// TrainBatchTask for classification
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, typename BinIndexType, decision_forest::classification::training::Method method, typename helper,
          typename HyperparameterType, CpuType cpu>
class TrainBatchTask : public TrainBatchTaskBase<algorithmFPType, BinIndexType, helper, HyperparameterType, cpu>
{
    typedef TrainBatchTaskBase<algorithmFPType, BinIndexType, helper, HyperparameterType, cpu> super;

public:
    typedef TreeThreadCtx<algorithmFPType, cpu> ThreadCtxType;
    TrainBatchTask(HostAppIface * pHostApp, const NumericTable * x, const NumericTable * y, const NumericTable * w,
                   const decision_forest::training::Parameter & par, const dtrees::internal::FeatureTypes & featTypes,
                   const dtrees::internal::IndexedFeatures * indexedFeatures, const BinIndexType * binIndex, typename super::ThreadCtxType & ctx,
                   size_t dummy, const HyperparameterType * hyperparameter = nullptr)
        : super(pHostApp, x, y, w, par, featTypes, indexedFeatures, binIndex, ctx, dummy, hyperparameter)
    {
        if (!this->_nFeaturesPerNode)
        {
            size_t nF(daal::internal::MathInst<algorithmFPType, cpu>::sSqrt(x->getNumberOfColumns()));
            const_cast<size_t &>(this->_nFeaturesPerNode) = nF;
        }
    }
};

//////////////////////////////////////////////////////////////////////////////////////////
// ClassificationTrainBatchKernel
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, Method method, CpuType cpu, typename helper>
services::Status computeForSpecificHelper(HostAppIface * pHostApp, const NumericTable * x, const NumericTable * y, const NumericTable * w,
                                          decision_forest::classification::Model & m, Result & res,
                                          const decision_forest::classification::training::Parameter & par, bool memSave,
                                          const classification::training::internal::Hyperparameter * hyperparameter)
{
    typedef classification::training::internal::Hyperparameter HyperparameterType;
    ResultData rd(par, res.get(variableImportance).get(), res.get(outOfBagError).get(), res.get(outOfBagErrorPerObservation).get(),
                  res.get(outOfBagErrorAccuracy).get(), nullptr, res.get(outOfBagErrorDecisionFunction).get(), nullptr);
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
                s = computeImpl<algorithmFPType, uint8_t, cpu, daal::algorithms::decision_forest::classification::internal::ModelImpl,
                                TrainBatchTask<algorithmFPType, uint8_t, hist, helper, HyperparameterType, cpu> >(
                    pHostApp, x, y, w, *static_cast<daal::algorithms::decision_forest::classification::internal::ModelImpl *>(&m), rd, par,
                    par.nClasses, featTypes, &indexedFeatures, hyperparameter);
            else if (indexedFeatures.maxNumIndices() <= 65536)
                s = computeImpl<algorithmFPType, uint16_t, cpu, daal::algorithms::decision_forest::classification::internal::ModelImpl,
                                TrainBatchTask<algorithmFPType, uint16_t, hist, helper, HyperparameterType, cpu> >(
                    pHostApp, x, y, w, *static_cast<daal::algorithms::decision_forest::classification::internal::ModelImpl *>(&m), rd, par,
                    par.nClasses, featTypes, &indexedFeatures, hyperparameter);
            else
                s = computeImpl<
                    algorithmFPType, dtrees::internal::IndexedFeatures::IndexType, cpu,
                    daal::algorithms::decision_forest::classification::internal::ModelImpl,
                    TrainBatchTask<algorithmFPType, dtrees::internal::IndexedFeatures::IndexType, hist, helper, HyperparameterType, cpu> >(
                    pHostApp, x, y, w, *static_cast<daal::algorithms::decision_forest::classification::internal::ModelImpl *>(&m), rd, par,
                    par.nClasses, featTypes, &indexedFeatures, hyperparameter);
        }
        else
            s = computeImpl<algorithmFPType, dtrees::internal::IndexedFeatures::IndexType, cpu,
                            daal::algorithms::decision_forest::classification::internal::ModelImpl,
                            TrainBatchTask<algorithmFPType, dtrees::internal::IndexedFeatures::IndexType, hist, helper, HyperparameterType, cpu> >(
                pHostApp, x, y, w, *static_cast<daal::algorithms::decision_forest::classification::internal::ModelImpl *>(&m), rd, par, par.nClasses,
                featTypes, nullptr, hyperparameter);
    }
    else
    {
        if (!memSave)
        {
            s = indexedFeatures.init<algorithmFPType, cpu>(*x, &featTypes);
            DAAL_CHECK_STATUS_VAR(s);
            s = computeImpl<
                algorithmFPType, dtrees::internal::IndexedFeatures::IndexType, cpu,
                daal::algorithms::decision_forest::classification::internal::ModelImpl,
                TrainBatchTask<algorithmFPType, dtrees::internal::IndexedFeatures::IndexType, defaultDense, helper, HyperparameterType, cpu> >(
                pHostApp, x, y, w, *static_cast<daal::algorithms::decision_forest::classification::internal::ModelImpl *>(&m), rd, par, par.nClasses,
                featTypes, &indexedFeatures, hyperparameter);
        }
        else
            s = computeImpl<
                algorithmFPType, dtrees::internal::IndexedFeatures::IndexType, cpu,
                daal::algorithms::decision_forest::classification::internal::ModelImpl,
                TrainBatchTask<algorithmFPType, dtrees::internal::IndexedFeatures::IndexType, defaultDense, helper, HyperparameterType, cpu> >(
                pHostApp, x, y, w, *static_cast<daal::algorithms::decision_forest::classification::internal::ModelImpl *>(&m), rd, par, par.nClasses,
                featTypes, nullptr, hyperparameter);
    }

    if (s.ok()) res.impl()->setEngine(rd.updatedEngine);
    return s;
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status ClassificationTrainBatchKernel<algorithmFPType, method, cpu>::compute(
    HostAppIface * pHostApp, const NumericTable * x, const NumericTable * y, const NumericTable * w, decision_forest::classification::Model & m,
    Result & res, const decision_forest::classification::training::Parameter & par, const HyperparameterType * hp)
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
                                     RespHelperBase<algorithmFPType, cpu, UnorderedRespHelperBest<algorithmFPType, cpu> > >(
            pHostApp, x, y, w, m, res, par, par.memorySavingMode, hp);
    }
    else if (par.splitter == decision_forest::training::SplitterMode::random)
    {
        s = computeForSpecificHelper<algorithmFPType, method, cpu,
                                     RespHelperBase<algorithmFPType, cpu, UnorderedRespHelperRandom<algorithmFPType, cpu> > >(
            pHostApp, x, y, w, m, res, par, par.memorySavingMode || method == defaultDense, hp);
    }
    return s;
}

} /* namespace internal */
} /* namespace training */
} /* namespace classification */
} /* namespace decision_forest */
} /* namespace algorithms */
} /* namespace daal */

#endif
