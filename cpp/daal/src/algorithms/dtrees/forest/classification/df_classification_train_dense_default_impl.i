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
#include <iostream>

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
// UnorderedRespHelper
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class UnorderedRespHelper : public DataHelper<algorithmFPType, ClassIndexType, cpu>
{
public:
    typedef ClassIndexType TResponse;
    typedef DataHelper<algorithmFPType, ClassIndexType, cpu> super;
    typedef typename dtrees::internal::TreeImpClassification<> TreeType;
    typedef typename TreeType::NodeType NodeType;
    typedef typename dtrees::internal::TVector<float, cpu, dtrees::internal::ScalableAllocator<cpu> > Histogramm;

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
    UnorderedRespHelper(const dtrees::internal::IndexedFeatures * indexedFeatures, size_t nClasses)
        : super(indexedFeatures), _nClasses(nClasses), _histLeft(nClasses), _impLeft(nClasses), _impRight(nClasses)
    {}
    virtual bool init(const NumericTable * data, const NumericTable * resp, const IndexType * aSample,
                      const NumericTable * weights) DAAL_C11_OVERRIDE;
    void convertLeftImpToRight(size_t n, const ImpurityData & total, TSplitData & split)
    {
        computeRightHistogramm(total.hist, split.left.hist, split.left.hist);
        split.nLeft       = n - split.nLeft;
        split.leftWeights = split.totalWeights - split.leftWeights;
        calcGini(split.leftWeights, split.left);
    }

    template <bool noWeights>
    void calcImpurity(const IndexType * aIdx, size_t n, ImpurityData & imp, double & totalweights) const;
    bool findBestSplitForFeature(const algorithmFPType * featureVal, const IndexType * aIdx, size_t n, size_t nMinSplitPart,
                                 const algorithmFPType accuracy, const ImpurityData & curImpurity, TSplitData & split,
                                 const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights, 
                                 engines::internal::BatchBaseImpl * engineImpl) const
    {
        ::std::cout << "Feature\n";
        const bool noWeights = !this->_weights;
        if (noWeights)
        {
            return split.featureUnordered ? findBestSplitCategoricalFeature(featureVal, aIdx, n, nMinSplitPart, accuracy, curImpurity, split,
                                                                            minWeightLeaf, totalWeights, engineImpl) :
                                            findBestSplitOrderedFeature<true>(featureVal, aIdx, n, nMinSplitPart, accuracy, curImpurity, split,
                                                                              minWeightLeaf, totalWeights, engineImpl);
        }
        else
        {
            return split.featureUnordered ? findBestSplitCategoricalFeature(featureVal, aIdx, n, nMinSplitPart, accuracy, curImpurity, split,
                                                                            minWeightLeaf, totalWeights, engineImpl) :
                                            findBestSplitOrderedFeature<false>(featureVal, aIdx, n, nMinSplitPart, accuracy, curImpurity, split,
                                                                               minWeightLeaf, totalWeights, engineImpl);
        }
    }
    bool terminateCriteria(ImpurityData & imp, algorithmFPType impurityThreshold, size_t nSamples) const { return imp.value() < impurityThreshold; }

    template <typename BinIndexType>
    int findBestSplitForFeatureSorted(algorithmFPType * featureBuf, IndexType iFeature, const IndexType * aIdx, size_t n, size_t nMinSplitPart,
                                      const ImpurityData & curImpurity, TSplitData & split, const algorithmFPType minWeightLeaf,
                                      const algorithmFPType totalWeights, const BinIndexType * binIndex,
                                      engines::internal::BatchBaseImpl * engineImpl) const;
    template <typename BinIndexType>
    void computeHistFewClassesWithoutWeights(IndexType iFeature, const IndexType * aIdx, const BinIndexType * binIndex, size_t n) const;
    template <typename BinIndexType>
    void computeHistFewClassesWithWeights(IndexType iFeature, const IndexType * aIdx, const BinIndexType * binIndex, size_t n) const;
    template <typename BinIndexType>
    void computeHistManyClasses(IndexType iFeature, const IndexType * aIdx, const BinIndexType * binIndex, size_t n) const;

    int findBestSplitbyHistDefault(int nDiffFeatMax, size_t n, size_t nMinSplitPart, const ImpurityData & curImpurity, TSplitData & split,
                                   const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights,
                                   engines::internal::BatchBaseImpl * engineImpl) const;

    template <int K, bool noWeights>
    int findBestSplitFewClasses(int nDiffFeatMax, size_t n, size_t nMinSplitPart, const ImpurityData & curImpurity, TSplitData & split,
                                const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights,
                                engines::internal::BatchBaseImpl * engineImpl) const;

    template <bool noWeights>
    int findBestSplitFewClassesDispatch(int nDiffFeatMax, size_t n, size_t nMinSplitPart, const ImpurityData & curImpurity, TSplitData & split,
                                        const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights, 
                                        engines::internal::BatchBaseImpl * engineImpl) const;

    template <bool noWeights, typename BinIndexType>
    void finalizeBestSplit(const IndexType * aIdx, const BinIndexType * binIndex, size_t n, IndexType iFeature, size_t idxFeatureValueBestSplit,
                           TSplitData & bestSplit, IndexType * bestSplitIdx) const;
    void simpleSplit(const algorithmFPType * featureVal, const IndexType * aIdx, TSplitData & split) const;

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
            OOBClassificationData * ptr = ((OOBClassificationData *)oobBuf) + _nClasses * iRow;
            ptr[response]++;
        }
        return this->predictionError(response, *y.get());
    }

    void setLeafData(typename TreeType::NodeType::Leaf & node, const IndexType * idx, size_t n, ImpurityData & imp) const
    {
        DAAL_ASSERT(n > 0);
        node.count    = n;
        node.impurity = imp.var;
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < _nClasses; ++i)
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
        for (size_t i = 1; i < _nClasses; ++i)
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

private:
    size_t nClasses() const
    {
        return _nClasses;
    }
    void calcGini(double totalWeights, ImpurityData & imp) const
    {
        const double sqWeights = totalWeights * totalWeights;
        const double one       = double(1);
        const double cDiv      = isZero<double, cpu>(sqWeights) ? one : (one / sqWeights);
        double var             = one;
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < _nClasses; ++i) var -= cDiv * double(imp.hist[i]) * double(imp.hist[i]);
        imp.var = var;
        if (!isPositive<double, cpu>(imp.var)) imp.var = 0; //roundoff error
    }

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

    void computeRightHistogramm(const Histogramm & total, const Histogramm & left, Histogramm & right) const
    {
        auto histTotal = total.get();
        auto histRight = right.get();
        auto histLeft  = left.get();
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t iClass = 0; iClass < _nClasses; ++iClass) histRight[iClass] = histTotal[iClass] - histLeft[iClass];
    }

    template <bool noWeights>
    bool findBestSplitOrderedFeature(const algorithmFPType * featureVal, const IndexType * aIdx, size_t n, size_t nMinSplitPart,
                                     const algorithmFPType accuracy, const ImpurityData & curImpurity, TSplitData & split,
                                     const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights,
                                     engines::internal::BatchBaseImpl * engineImpl) const;
    bool findBestSplitCategoricalFeature(const algorithmFPType * featureVal, const IndexType * aIdx, size_t n, size_t nMinSplitPart,
                                         const algorithmFPType accuracy, const ImpurityData & curImpurity, TSplitData & split,
                                         const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights,
                                         engines::internal::BatchBaseImpl * engineImpl) const;

private:
    const size_t _nClasses;
    //set of buffers for indexed features processing, used in findBestSplitForFeatureIndexed only
    const size_t _nClassesThreshold = 8;
    mutable TVector<IndexType, cpu> _idxFeatureBuf;
    mutable TVector<algorithmFPType, cpu> _weightsFeatureBuf;
    mutable TVector<float, cpu> _samplesPerClassBuf;
    mutable Histogramm _histLeft;
    //work variables used in memory saving mode only
    mutable ImpurityData _impLeft;
    mutable ImpurityData _impRight;
};

#ifdef DEBUG_CHECK_IMPURITY
template <typename algorithmFPType, CpuType cpu>
void UnorderedRespHelper<algorithmFPType, cpu>::checkImpurity(const IndexType * ptrIdx, algorithmFPType totalWeights,
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

template <typename algorithmFPType, CpuType cpu>
bool UnorderedRespHelper<algorithmFPType, cpu>::init(const NumericTable * data, const NumericTable * resp, const IndexType * aSample,
                                                     const NumericTable * weights)
{
    DAAL_CHECK_STATUS_VAR(super::init(data, resp, aSample, weights));
    if (this->_indexedFeatures)
    {
        //init work buffers for the computation using indexed features
        const auto nDiffFeatMax = this->indexedFeatures().maxNumIndices();
        _idxFeatureBuf.reset(nDiffFeatMax);
        _weightsFeatureBuf.reset(nDiffFeatMax);
        _samplesPerClassBuf.reset(nClasses() * nDiffFeatMax);
        return _idxFeatureBuf.get() && _weightsFeatureBuf.get() && _samplesPerClassBuf.get();
    }
    return true;
}

template <typename algorithmFPType, CpuType cpu>
template <bool noWeights>
void UnorderedRespHelper<algorithmFPType, cpu>::calcImpurity(const IndexType * aIdx, size_t n, ImpurityData & imp, double & totalWeights) const
{
    imp.init(_nClasses);
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
    calcGini(totalWeights, imp);
}

template <typename algorithmFPType, CpuType cpu>
void UnorderedRespHelper<algorithmFPType, cpu>::simpleSplit(const algorithmFPType * featureVal, const IndexType * aIdx, TSplitData & split) const
{
    split.left.init(_nClasses);
    const ClassIndexType iClass = this->_aResponse[aIdx[0]].val;
    split.featureValue          = featureVal[0];
    split.iStart                = 0;
    split.left.hist[iClass]     = this->_aWeights[aIdx[0]].val;
    split.nLeft                 = 1;
    split.leftWeights           = this->_aWeights[aIdx[0]].val;
    split.totalWeights          = this->_aWeights[aIdx[0]].val + this->_aWeights[aIdx[1]].val;
}

template <typename algorithmFPType, CpuType cpu>
template <bool noWeights>
bool UnorderedRespHelper<algorithmFPType, cpu>::findBestSplitOrderedFeature(const algorithmFPType * featureVal, const IndexType * aIdx, size_t n,
                                                                            size_t nMinSplitPart, const algorithmFPType accuracy,
                                                                            const ImpurityData & curImpurity, TSplitData & split,
                                                                            const algorithmFPType minWeightLeaf, algorithmFPType totalWeights,
                                                                            engines::internal::BatchBaseImpl * engineImpl) const
{
    //init variables
    ::std::cout << "Ordered\n";
    size_t i;
    const bool bBestFromOtherFeatures      = isPositive<algorithmFPType, cpu>(split.impurityDecrease);        
    algorithmFPType vBestFromOtherFeatures = bBestFromOtherFeatures ? totalWeights * (curImpurity.var - split.impurityDecrease) : algorithmFPType(-1);
    
    bool bFound           = false;
    IndexType iBest       = -1;
    algorithmFPType leftWeights  = algorithmFPType(0);
    algorithmFPType rightWeights = algorithmFPType(0);
    algorithmFPType v = algorithmFPType(0);
    double lW, rW; //needed to interface with calcImpurity without changing it...


    //select random split index 
    if (n - nMinSplitPart + 1 >= nMinSplitPart)
    {    
        RNGs<size_t, cpu> rng;
        rng.uniform(1, &i, engineImpl->getState(), nMinSplitPart, n - nMinSplitPart + 1); //how to call this
    } 
    else
    {
        return bFound;
    }
    
    if(noWeights) //calculate Gini impurity for left and right
    {
        calcImpurity<true>(aIdx, i, _impLeft, lW);
        calcImpurity<true>(aIdx+i, n, _impRight, rW); 
    } 
    else
    {
        calcImpurity<false>(aIdx, i, _impLeft, lW);
        calcImpurity<false>(aIdx+i, n, _impRight, rW); //rightWeights is generally unused
    }
    leftWeights = lW; //double to float can possibly occur here
    rightWeights = rW;

#ifdef DEBUG_CHECK_IMPURITY
                checkImpurity(aIdx, leftWeights, _impLeft);
                checkImpurity(aIdx + i, totalWeights - leftWeights, _impRight);
#endif
    //check minWeightLeaf conditions for left and right
    if ((leftWeights > minWeightLeaf) && (rightWeights > minWeightLeaf)) //it is a valid split with enought leaf weights
    {
        //check if bFound condition below
        if (!isPositive<algorithmFPType, cpu>(_impLeft.var)) _impLeft.var = 0; //set left impurity to 0 if negative
        if (!isPositive<algorithmFPType, cpu>(_impRight.var)) _impRight.var = 0; //set right impurity to 0 if negative

        v = leftWeights * _impLeft.var + rightWeights * _impRight.var; //calculate overall weighted Gini index

        if (isGreater<algorithmFPType, cpu>(vBestFromOtherFeatures, v)) //if it has a better weighted gini overwite parameters
        {
            bFound             = true;  
            split.left.var     = _impLeft.var; 
            split.left.hist    = _impLeft.hist;
            iBest              = i; 
            split.nLeft        = i;
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
bool UnorderedRespHelper<algorithmFPType, cpu>::findBestSplitCategoricalFeature(const algorithmFPType * featureVal, const IndexType * aIdx, size_t n,
                                                                                size_t nMinSplitPart, const algorithmFPType accuracy,
                                                                                const ImpurityData & curImpurity, TSplitData & split,
                                                                                const algorithmFPType minWeightLeaf,
                                                                                const algorithmFPType totalWeights,
                                                                                engines::internal::BatchBaseImpl * engineImpl) const
{
    ::std::cout << "Categorical\n";
    //initialize variables
    DAAL_ASSERT(n >= 2 * nMinSplitPart);
    size_t i;
    _impRight.init(_nClasses);
    _impLeft.init(_nClasses);
    bool bFound                       = false;
    const bool bBestFromOtherFeatures = isPositive<algorithmFPType, cpu>(split.impurityDecrease);
    algorithmFPType vBest             = -1;
    IndexType iBest                   = -1;
    algorithmFPType leftWeights  = algorithmFPType(0);
    algorithmFPType rightWeights = algorithmFPType(0);
    algorithmFPType v = algorithmFPType(0);
    double lW, rW;
    
    //recover best Gini impurity change for current split
    const algorithmFPType vBestFromOtherFeatures = bBestFromOtherFeatures ? totalWeights * (curImpurity.var - split.impurityDecrease) : algorithmFPType(-1);

    //select random split index 
    if (n - nMinSplitPart + 1 >= nMinSplitPart)
    {    
        RNGs<size_t, cpu> rng;
        rng.uniform(1, &i, engineImpl->getState(), nMinSplitPart, n - nMinSplitPart + 1); //how to call this
    } 
    else
    {
        return bFound;
    }

    calcImpurity<false>(aIdx, i, _impLeft, lW);
    calcImpurity<false>(aIdx+i, n, _impRight, rW); //rightWeights is generally unused
    
    leftWeights = lW; //double to float can possibly occur here
    rightWeights = rW;

    //check minWeightLeaf conditions for left and right
    if ((leftWeights > minWeightLeaf) && (rightWeights > minWeightLeaf)) //it is a valid split with enought leaf weights
    {
        //check if bFound condition below
        if (!isPositive<algorithmFPType, cpu>(_impLeft.var)) _impLeft.var = 0; //set left impurity to 0 if negative
        if (!isPositive<algorithmFPType, cpu>(_impRight.var)) _impRight.var = 0; //set right impurity to 0 if negative

        v = leftWeights * _impLeft.var + rightWeights * _impRight.var; //calculate overall weighted Gini index

        if (isGreater<algorithmFPType, cpu>(vBestFromOtherFeatures, v)) //if it has a better weighted gini overwite parameters
        {
            bFound             = true;  
            split.left.var     = _impLeft.var; 
            split.left.hist    = _impLeft.hist;
            iBest              = i; 
            split.nLeft        = i;
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
template <typename BinIndexType>
void UnorderedRespHelper<algorithmFPType, cpu>::computeHistFewClassesWithoutWeights(IndexType iFeature, const IndexType * aIdx,
                                                                                    const BinIndexType * binIndex, size_t n) const
{
    const algorithmFPType one(1.0);
    const auto aResponse  = this->_aResponse.get();
    auto nSamplesPerClass = _samplesPerClassBuf.get();
    {
        for (size_t i = 0; i < n; ++i)
        {
            const IndexType iSample = aIdx[i];
            const auto & r          = aResponse[aIdx[i]];

            const BinIndexType idx      = binIndex[r.idx];
            const ClassIndexType iClass = r.val;
            nSamplesPerClass[idx * _nClasses + iClass] += one;
        }
    }
}

template <typename algorithmFPType, CpuType cpu>
template <typename BinIndexType>
void UnorderedRespHelper<algorithmFPType, cpu>::computeHistFewClassesWithWeights(IndexType iFeature, const IndexType * aIdx,
                                                                                 const BinIndexType * binIndex, size_t n) const
{
    const auto aResponse = this->_aResponse.get();
    const auto aWeights  = this->_aWeights.get();

    auto nFeatIdx         = _idxFeatureBuf.get();
    auto nSamplesPerClass = _samplesPerClassBuf.get();

    {
        for (size_t i = 0; i < n; ++i)
        {
            const IndexType iSample = aIdx[i];
            const auto & r          = aResponse[aIdx[i]];
            const BinIndexType idx  = binIndex[r.idx];
            ++nFeatIdx[idx];
            const auto weights          = aWeights[iSample].val;
            const ClassIndexType iClass = r.val;
            nSamplesPerClass[idx * _nClasses + iClass] += weights;
        }
    }
}

template <typename algorithmFPType, CpuType cpu>
template <typename BinIndexType>
void UnorderedRespHelper<algorithmFPType, cpu>::computeHistManyClasses(IndexType iFeature, const IndexType * aIdx, const BinIndexType * binIndex,
                                                                       size_t n) const
{
    const auto aResponse = this->_aResponse.get();
    const auto aWeights  = this->_aWeights.get();

    auto nFeatIdx         = _idxFeatureBuf.get();
    auto featWeights      = _weightsFeatureBuf.get();
    auto nSamplesPerClass = _samplesPerClassBuf.get();

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
            nSamplesPerClass[idx * _nClasses + iClass] += weights;
        }
    }
}

template <typename algorithmFPType, CpuType cpu>
int UnorderedRespHelper<algorithmFPType, cpu>::findBestSplitbyHistDefault(int nDiffFeatMax, size_t n, size_t nMinSplitPart,
                                                                          const ImpurityData & curImpurity, TSplitData & split,
                                                                          const algorithmFPType minWeightLeaf,
                                                                          const algorithmFPType totalWeights,
                                                                          engines::internal::BatchBaseImpl * engineImpl) const
{
    size_t idx;
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

    RNGs<size_t, cpu> rng;
    rng.uniform(1, &idx, engineImpl->getState(), 0, nDiffFeatMax - 1); //NEED TO DOUBLE CHECK ON nDiffFeatMax

    //only if split.featureUnordered is False, then a loop needs to start
    if (split.featureUnordered)
    {
        nLeft = nFeatIdx[idx];
        leftWeights = featWeights[idx];

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        //one against others
        for (size_t iClass = 0; iClass < _nClasses; ++iClass) histLeft[iClass] = nSamplesPerClass[idx * _nClasses + iClass];

    } 
    else
    {
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for(size_t i=0; i <= idx; ++i) //include idx for the proper split
        {
            nLeft += nFeatIdx[i];
            leftWeights += featWeights[i];
        }
        //split of for loop is to insure vectorization above
        for(size_t i=0; i <= idx; ++i) //include idx for the proper split
        {
            // if statement is in the case of sparsity
            if (nFeatIdx[i] > 0)
            {
                PRAGMA_IVDEP
                PRAGMA_VECTOR_ALWAYS
                for (size_t iClass = 0; iClass < _nClasses; ++iClass) histLeft[iClass] += nSamplesPerClass[i * _nClasses + iClass];
            }
            
        }


    }

    if (!((nLeft == n) || ((n - nLeft) < nMinSplitPart)
         || ((totalWeights - leftWeights) < minWeightLeaf)
         || ((nLeft < nMinSplitPart) || leftWeights < minWeightLeaf)))
    {

        auto histTotal           = curImpurity.hist.get(); //there is no reason to have this in the for loop
        algorithmFPType sumLeft  = 0;
        algorithmFPType sumRight = 0;
        PRAGMA_IVDEP
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
int UnorderedRespHelper<algorithmFPType, cpu>::findBestSplitFewClasses(int nDiffFeatMax, size_t n, size_t nMinSplitPart,
                                                                       const ImpurityData & curImpurity, TSplitData & split,
                                                                       const algorithmFPType minWeightLeaf, const algorithmFPType totalWeights,
                                                                       engines::internal::BatchBaseImpl * engineImpl) const
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

    size_t idx;
    RNGs<size_t, cpu> rng;
    rng.uniform(1, &idx, engineImpl->getState(), 0, nDiffFeatMax - 1); //NEED TO DOUBLE CHECK ON nDiffFeatMax

    for (size_t i = 0; i <= idx; ++i)
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

        if (!(i == idx)) continue; // only run when i==idx

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

    /*size_t idx;
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


    RNGs<size_t, cpu> rng;
    rng.uniform(1, &idx, engineImpl->getState(), 0, nDiffFeatMax - 1); //NEED TO DOUBLE CHECK ON nDiffFeatMax

    if (noWeights) ::std::cout << "noweights\n";

    if (split.featureUnordered)
    {
        ::std::cout << "unordered\n" ;

        if (noWeights)
        {
            //PRAGMA_IVDEP
            //PRAGMA_VECTOR_ALWAYS
            for (size_t iClass = 0; iClass < K; ++iClass)
            {
                nLeft += nSamplesPerClass[idx * K + iClass];
            }

            while((idx < nDiffFeatMax-1) && (nLeft == 0)){//find a plausable idx by searching through histogram to end
                idx++;
                for (size_t iClass = 0; iClass < K; ++iClass)
                {
                    nLeft += nSamplesPerClass[idx * K + iClass];
                }

            }
            if(nLeft == 0) return idxFeatureBestSplit; //this is a check that if idx = nDiffFeatMax -1, if nLeft is still 0 there is no split

            leftWeights = nLeft;
        }
        else
        {
            nLeft = nFeatIdx[idx];
            //while((idx < nDiffFeatMax-1) && (nLeft == 0)) nLeft = nFeatIdx[++idx]; //find a plausable idx by searching through histogram to end
            if(nLeft == 0) return idxFeatureBestSplit; //this is a check that if idx = nDiffFeatMax -1, if nLeft is still 0 there is no split
            
            //PRAGMA_IVDEP
            //PRAGMA_VECTOR_ALWAYS
            for (size_t iClass = 0; iClass < K; ++iClass)
            {
                leftWeights += nSamplesPerClass[idx * K + iClass];

            }
        }
    }
    else
    {

        ::std::cout << "ordered\n";
        size_t lim=K*(idx+1); //unrolled 2D to 1D [idx*k+i]

        if (noWeights)
        {
            //PRAGMA_IVDEP
            //PRAGMA_VECTOR_ALWAYS
            for (size_t i=0; i < lim; ++i)
            {
                nLeft += nSamplesPerClass[i];
            }
            leftWeights = nLeft;
        }
        else
        {        
            //PRAGMA_IVDEP
            //PRAGMA_VECTOR_ALWAYS
            for(size_t i =0; i <= idx; ++i)
            {
                nLeft += nFeatIdx[idx];
            }

            //PRAGMA_IVDEP
            //PRAGMA_VECTOR_ALWAYS
            for(size_t i =0; i < lim; ++i) 
            {
                leftWeights += nSamplesPerClass[i];
            }
        }
    }


    if (!((nLeft == n) || ((n - nLeft) < nMinSplitPart)
         || ((totalWeights - leftWeights) < minWeightLeaf)
         || ((nLeft < nMinSplitPart) || leftWeights < minWeightLeaf)))
    {

        if(split.featureUnordered)
        {
            //PRAGMA_IVDEP
            //PRAGMA_VECTOR_ALWAYS
            for (size_t iClass = 0; iClass < K; ++iClass) histLeft[iClass] = nSamplesPerClass[idx*K + iClass];
        }
        else
        {   
            for(size_t i=0; i <= idx; ++i)
            {
                //PRAGMA_IVDEP
                //PRAGMA_VECTOR_ALWAYS
                for (size_t iClass = 0; iClass < K; ++iClass) histLeft[iClass] += nSamplesPerClass[i*K + iClass];
            }

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
            idxFeatureBestSplit = idx;
            bestImpDecrease     = decrease;
        }
    }
    if (idxFeatureBestSplit >= 0)
    {
        split.impurityDecrease = curImpurity.var + bestImpDecrease / totalWeights - algorithmFPType(1);
        split.totalWeights     = totalWeights;
    }

    return idxFeatureBestSplit;*/
}

template <typename algorithmFPType, CpuType cpu>
template <bool noWeights>
int UnorderedRespHelper<algorithmFPType, cpu>::findBestSplitFewClassesDispatch(int nDiffFeatMax, size_t n, size_t nMinSplitPart,
                                                                               const ImpurityData & curImpurity, TSplitData & split,
                                                                               const algorithmFPType minWeightLeaf,
                                                                               const algorithmFPType totalWeights,
                                                                               engines::internal::BatchBaseImpl * engineImpl) const
{
    DAAL_ASSERT(_nClasses <= _nClassesThreshold);
    switch (_nClasses)
    {
    case 2: return findBestSplitFewClasses<2, noWeights>(nDiffFeatMax, n, nMinSplitPart, curImpurity, split, minWeightLeaf, totalWeights, engineImpl);
    case 3: return findBestSplitFewClasses<3, noWeights>(nDiffFeatMax, n, nMinSplitPart, curImpurity, split, minWeightLeaf, totalWeights, engineImpl);
    case 4: return findBestSplitFewClasses<4, noWeights>(nDiffFeatMax, n, nMinSplitPart, curImpurity, split, minWeightLeaf, totalWeights, engineImpl);
    case 5: return findBestSplitFewClasses<5, noWeights>(nDiffFeatMax, n, nMinSplitPart, curImpurity, split, minWeightLeaf, totalWeights, engineImpl);
    case 6: return findBestSplitFewClasses<6, noWeights>(nDiffFeatMax, n, nMinSplitPart, curImpurity, split, minWeightLeaf, totalWeights, engineImpl);
    case 7: return findBestSplitFewClasses<7, noWeights>(nDiffFeatMax, n, nMinSplitPart, curImpurity, split, minWeightLeaf, totalWeights, engineImpl);
    case 8: return findBestSplitFewClasses<8, noWeights>(nDiffFeatMax, n, nMinSplitPart, curImpurity, split, minWeightLeaf, totalWeights, engineImpl);
    }
    return -1;
}

template <typename algorithmFPType, CpuType cpu>
template <typename BinIndexType>
int UnorderedRespHelper<algorithmFPType, cpu>::findBestSplitForFeatureSorted(algorithmFPType * featureBuf, IndexType iFeature, const IndexType * aIdx,
                                                                             size_t n, size_t nMinSplitPart, const ImpurityData & curImpurity,
                                                                             TSplitData & split, const algorithmFPType minWeightLeaf,
                                                                             const algorithmFPType totalWeights, const BinIndexType * binIndex,
                                                                             engines::internal::BatchBaseImpl * engineImpl) const
{

    ::std::cout << "Sorted\n";
    const auto nDiffFeatMax = this->indexedFeatures().numIndices(iFeature);
    _samplesPerClassBuf.setValues(nClasses() * nDiffFeatMax, 0);

    int idxFeatureBestSplit = -1; //index of best feature value in the array of sorted feature values

    if (_nClasses <= _nClassesThreshold)
    {
        if (!this->_weights)
        {
            // nSamplesPerClass - computed. nFeatIdx and featWeights - no
            computeHistFewClassesWithoutWeights(iFeature, aIdx, binIndex, n);
            idxFeatureBestSplit =
                findBestSplitFewClassesDispatch<true>(nDiffFeatMax, n, nMinSplitPart, curImpurity, split, minWeightLeaf, totalWeights, engineImpl);
        }
        else
        {
            // nSamplesPerClass and nFeatIdx - computed, featWeights - no
            _idxFeatureBuf.setValues(nDiffFeatMax, algorithmFPType(0));
            computeHistFewClassesWithWeights(iFeature, aIdx, binIndex, n);
            idxFeatureBestSplit =
                findBestSplitFewClassesDispatch<false>(nDiffFeatMax, n, nMinSplitPart, curImpurity, split, minWeightLeaf, totalWeights, engineImpl);
        }
    }
    else
    {
        // nSamplesPerClass, nFeatIdx and featWeights - computed
        _weightsFeatureBuf.setValues(nDiffFeatMax, algorithmFPType(0));
        _idxFeatureBuf.setValues(nDiffFeatMax, algorithmFPType(0));
        computeHistManyClasses(iFeature, aIdx, binIndex, n);
        idxFeatureBestSplit = findBestSplitbyHistDefault(nDiffFeatMax, n, nMinSplitPart, curImpurity, split, minWeightLeaf, totalWeights, engineImpl);
    }

    return idxFeatureBestSplit;
}

template <typename algorithmFPType, CpuType cpu>
template <bool noWeights, typename BinIndexType>
void UnorderedRespHelper<algorithmFPType, cpu>::finalizeBestSplit(const IndexType * aIdx, const BinIndexType * binIndex, size_t n, IndexType iFeature,
                                                                  size_t idxFeatureValueBestSplit, TSplitData & bestSplit,
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
    bestSplit.featureValue = (this->getValue(iFeature, iRowSplitVal) + this->getValue(iFeature, iNext)) / (algorithmFPType)2.;
    if (bestSplit.featureValue == this->getValue(iFeature, iNext)) bestSplit.featureValue = this->getValue(iFeature, iRowSplitVal);
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
            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for (size_t i = 0, n = _nClasses * nSamples; i < n; ++i) dst[i] += src[i];
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
template <typename algorithmFPType, typename BinIndexType, decision_forest::classification::training::Method method, CpuType cpu>
class TrainBatchTask : public TrainBatchTaskBase<algorithmFPType, BinIndexType, UnorderedRespHelper<algorithmFPType, cpu>, cpu>
{
    typedef TrainBatchTaskBase<algorithmFPType, BinIndexType, UnorderedRespHelper<algorithmFPType, cpu>, cpu> super;

public:
    typedef TreeThreadCtx<algorithmFPType, cpu> ThreadCtxType;
    TrainBatchTask(HostAppIface * pHostApp, const NumericTable * x, const NumericTable * y, const NumericTable * w,
                   const decision_forest::training::Parameter & par, const dtrees::internal::FeatureTypes & featTypes,
                   const dtrees::internal::IndexedFeatures * indexedFeatures, const BinIndexType * binIndex, typename super::ThreadCtxType & ctx,
                   size_t dummy)
        : super(pHostApp, x, y, w, par, featTypes, indexedFeatures, binIndex, ctx, dummy)
    {
        if (!this->_nFeaturesPerNode)
        {
            size_t nF(daal::internal::Math<algorithmFPType, cpu>::sSqrt(x->getNumberOfColumns()));
            const_cast<size_t &>(this->_nFeaturesPerNode) = nF;
        }
    }
};

//////////////////////////////////////////////////////////////////////////////////////////
// ClassificationTrainBatchKernel
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status ClassificationTrainBatchKernel<algorithmFPType, method, cpu>::compute(
    HostAppIface * pHostApp, const NumericTable * x, const NumericTable * y, const NumericTable * w, decision_forest::classification::Model & m,
    Result & res, const decision_forest::classification::training::Parameter & par)
{
    ResultData rd(par, res.get(variableImportance).get(), res.get(outOfBagError).get(), res.get(outOfBagErrorPerObservation).get(),
                  res.get(outOfBagErrorAccuracy).get(), nullptr, res.get(outOfBagErrorDecisionFunction).get(), nullptr);
    services::Status s;
    dtrees::internal::FeatureTypes featTypes;
    DAAL_CHECK(featTypes.init(*x), ErrorMemoryAllocationFailed);
    dtrees::internal::IndexedFeatures indexedFeatures;
    if (method == hist)
    {
        if (!par.memorySavingMode)
        {
            BinParams prm(par.maxBins, par.minBinSize);
            s = indexedFeatures.init<algorithmFPType, cpu>(*x, &featTypes, &prm);
            DAAL_CHECK_STATUS_VAR(s);
            if (indexedFeatures.maxNumIndices() <= 256)
                s = computeImpl<algorithmFPType, uint8_t, cpu, daal::algorithms::decision_forest::classification::internal::ModelImpl,
                                TrainBatchTask<algorithmFPType, uint8_t, hist, cpu> >(
                    pHostApp, x, y, w, *static_cast<daal::algorithms::decision_forest::classification::internal::ModelImpl *>(&m), rd, par,
                    par.nClasses, featTypes, indexedFeatures);
            else if (indexedFeatures.maxNumIndices() <= 65536)
                s = computeImpl<algorithmFPType, uint16_t, cpu, daal::algorithms::decision_forest::classification::internal::ModelImpl,
                                TrainBatchTask<algorithmFPType, uint16_t, hist, cpu> >(
                    pHostApp, x, y, w, *static_cast<daal::algorithms::decision_forest::classification::internal::ModelImpl *>(&m), rd, par,
                    par.nClasses, featTypes, indexedFeatures);
            else
                s = computeImpl<algorithmFPType, dtrees::internal::IndexedFeatures::IndexType, cpu,
                                daal::algorithms::decision_forest::classification::internal::ModelImpl,
                                TrainBatchTask<algorithmFPType, dtrees::internal::IndexedFeatures::IndexType, hist, cpu> >(
                    pHostApp, x, y, w, *static_cast<daal::algorithms::decision_forest::classification::internal::ModelImpl *>(&m), rd, par,
                    par.nClasses, featTypes, indexedFeatures);
        }
        else
            s = computeImpl<algorithmFPType, dtrees::internal::IndexedFeatures::IndexType, cpu,
                            daal::algorithms::decision_forest::classification::internal::ModelImpl,
                            TrainBatchTask<algorithmFPType, dtrees::internal::IndexedFeatures::IndexType, hist, cpu> >(
                pHostApp, x, y, w, *static_cast<daal::algorithms::decision_forest::classification::internal::ModelImpl *>(&m), rd, par, par.nClasses,
                featTypes, indexedFeatures);
    }
    else
    {
        if (!par.memorySavingMode)
        {
            s = indexedFeatures.init<algorithmFPType, cpu>(*x, &featTypes);
            DAAL_CHECK_STATUS_VAR(s);
        }
        s = computeImpl<algorithmFPType, dtrees::internal::IndexedFeatures::IndexType, cpu,
                        daal::algorithms::decision_forest::classification::internal::ModelImpl,
                        TrainBatchTask<algorithmFPType, dtrees::internal::IndexedFeatures::IndexType, defaultDense, cpu> >(
            pHostApp, x, y, w, *static_cast<daal::algorithms::decision_forest::classification::internal::ModelImpl *>(&m), rd, par, par.nClasses,
            featTypes, indexedFeatures);
    }

    if (s.ok()) res.impl()->setEngine(rd.updatedEngine);
    return s;
}

} /* namespace internal */
} /* namespace training */
} /* namespace classification */
} /* namespace decision_forest */
} /* namespace algorithms */
} /* namespace daal */

#endif
