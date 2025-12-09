/* file: gbt_train_aux.i */
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
//  Implementation of auxiliary functions for gradient boosted trees training
//  (defaultDense) method.
//--
*/

#ifndef __GBT_TRAIN_AUX_I__
#define __GBT_TRAIN_AUX_I__

#include "src/algorithms/dtrees/dtrees_model_impl.h"
#include "src/algorithms/dtrees/dtrees_train_data_helper.i"
#include "src/algorithms/dtrees/gbt/gbt_internal.h"
#include "src/threading/threading.h"
#include "src/algorithms/dtrees/gbt/gbt_model_impl.h"
#include "services/daal_atomic_int.h"
#include "src/externals/service_service.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace training
{
namespace internal
{
using namespace daal::algorithms::dtrees::training::internal;
using namespace daal::algorithms::gbt::internal;

template <CpuType cpu>
void deleteTables(gbt::internal::GbtDecisionTree ** aTbl, HomogenNumericTable<double> ** aTblImp, HomogenNumericTable<int> ** aTblSmplCnt, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        delete aTbl[i];
        delete aTblImp[i];
        delete aTblSmplCnt[i];
        aTbl[i]        = nullptr;
        aTblImp[i]     = nullptr;
        aTblSmplCnt[i] = nullptr;
    }
}

template <CpuType cpu>
size_t getNBlocksForOpt(size_t nThreads, size_t n)
{
    if (nThreads <= 1 || n < 128000) return 1;
    const size_t blockSize = 512;
    size_t nBlocks         = n / blockSize;
    if (nBlocks % blockSize) nBlocks++;
    return nBlocks;
}

//////////////////////////////////////////////////////////////////////////////////////////
// Data helper class for regression
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class OrderedRespHelper : public DataHelperBase<algorithmFPType, cpu>
{
public:
    typedef DataHelperBase<algorithmFPType, cpu> super;

public:
    OrderedRespHelper(const dtrees::internal::IndexedFeatures * indexedFeatures) : super(indexedFeatures), _aIdxToRow(nullptr) {}
    const algorithmFPType * y() const { return _y.get(); }

    services::Status init(const NumericTable * data, const NumericTable * resp, const IndexType * aIdxToRow)
    {
        super::init(data, resp);
        _y.reset(data->getNumberOfRows());
        DAAL_CHECK_MALLOC(_y.get());
        ReadRows<algorithmFPType, cpu> bd(const_cast<NumericTable *>(resp), 0, _y.size());
        services::internal::tmemcpy<algorithmFPType, cpu>(_y.get(), bd.get(), _y.size());
        setIdxToRowMapping(aIdxToRow);
        return services::Status();
    }
    void setIdxToRowMapping(const IndexType * aIdxToRow) { _aIdxToRow = aIdxToRow; }

    void getColumnValues(size_t iCol, const IndexType * aIdx, size_t n, algorithmFPType * aVal) const
    {
        if (this->_dataDirect)
        {
            for (size_t i = 0; i < n; ++i) aVal[i] = this->_dataDirect[aIdx[i] * this->_nCols + iCol];
        }
        else
        {
            data_management::BlockDescriptor<algorithmFPType> bd;
            for (size_t i = 0; i < n; ++i)
            {
                this->_data->getBlockOfColumnValues(iCol, aIdx[i], 1, readOnly, bd);
                aVal[i] = *bd.getBlockPtr();
                this->_data->releaseBlockOfColumnValues(bd);
            }
        }
    }

    bool hasDiffFeatureValues(IndexType iFeature, const int * aIdx, size_t n) const
    {
        if (this->indexedFeatures().numIndices(iFeature) == 1) return false; //single value only
        const IndexedFeatures::IndexType * indexedFeature = this->indexedFeatures().data(iFeature);
        size_t i                                          = 1;

        const IndexedFeatures::IndexType idx0 = indexedFeature[aIdx[0]];
        for (; i < n; ++i)
        {
            const IndexedFeatures::IndexType idx = indexedFeature[aIdx[i]];
            if (idx != idx0) break;
        }
        return (i != n);
    }

protected:
    TArray<algorithmFPType, cpu> _y;
    const IndexType * _aIdxToRow; //for the methods that take in array of indices, this is the
                                  // mapping of the index to the row, if required
};

//////////////////////////////////////////////////////////////////////////////////////////
// Service class, pair (gradient, hessian) of algorithmFPType values
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
struct gh
{
    typedef double intermSummFPType; // needed due to templating shared with forests
    algorithmFPType g;               //gradient
    algorithmFPType h;               //hessian
    gh() : g(0), h(0) {}
    gh(algorithmFPType _g, algorithmFPType _h) : g(_g), h(_h) {}
    gh(const gh & o) : g(o.g), h(o.h) {}
    gh(const gh & total, const gh & part) : g(total.g - part.g), h(total.h - part.h) {}
    gh & operator=(const gh & o)
    {
        g = o.g;
        h = o.h;
        return *this;
    }
    void reset(algorithmFPType _g, algorithmFPType _h)
    {
        g = _g;
        h = _h;
    }
    void add(const gh & o)
    {
        g += o.g;
        h += o.h;
    }
    algorithmFPType value(algorithmFPType regLambda) const { return (g / (h + regLambda)) * g; }
};

//////////////////////////////////////////////////////////////////////////////////////////
// Impurity data
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
using ImpurityData = gh<algorithmFPType, cpu>;

//////////////////////////////////////////////////////////////////////////////////////////
// Base class for loss function L(y,f), where y is a response value,
// f is its current approximation
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class LossFunction : public Base
{
public:
    virtual void getGradients(size_t n, size_t nRows, const algorithmFPType * y, const algorithmFPType * f, const IndexType * sampleInd,
                              algorithmFPType * gh) = 0;
};

//////////////////////////////////////////////////////////////////////////////////////////
// Service class, the sum of pairs (gradient, hessian) corresponding to the same value of indexed feature
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
struct ghSum : public gh<algorithmFPType, cpu>
{
    ghSum() : gh<algorithmFPType, cpu>(), n(0) {}

    inline ghSum<algorithmFPType, cpu> & operator+=(const ghSum<algorithmFPType, cpu> & other) // TODO: remove
    {
        this->g += other.g;
        this->h += other.h;
        this->n += other.n;
        return *this;
    }

    algorithmFPType n;
    algorithmFPType dummy;
};

//////////////////////////////////////////////////////////////////////////////////////////
// Base memory helper class
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class MemHelperBase : public Base
{
protected:
    MemHelperBase(size_t nFeaturesIdx) : _nFeaturesIdx(nFeaturesIdx) {}

public:
    typedef gh<algorithmFPType, cpu> ghType;
    typedef ghSum<algorithmFPType, cpu> ghSumType;
    typedef TVector<IndexType, cpu> IndexTypeVector;
    typedef TVector<ghType, cpu> ghTypeVector;
    typedef TVector<ghSumType, cpu> ghSumTypeVector;
    typedef TVector<algorithmFPType, cpu> algorithmFPTypeVector;

    virtual bool init() = 0;
    //get buffer for the indices of features to be used for the split at the current level
    virtual IndexType * getFeatureSampleBuf() = 0;
    //release the buffer
    virtual void releaseFeatureSampleBuf(IndexType * buf) = 0;

    //get buffer for ghSums to be used for the split of an indexed feature at the current level
    virtual ghSumType * getGHSumBuf(size_t size) = 0;

    //get buffer for the feature values to be used for the split at the current level
    virtual algorithmFPTypeVector * getFeatureValueBuf(size_t size) = 0;
    //release the buffer
    virtual void releaseFeatureValueBuf(algorithmFPTypeVector * buf) = 0;

    //get buffer for the indexes of the sorted feature values to be used for the split at the current level
    virtual IndexTypeVector * getSortedFeatureIdxBuf(size_t size) = 0;
    //release the buffer
    virtual void releaseSortedFeatureIdxBuf(IndexTypeVector * p) = 0;

protected:
    const size_t _nFeaturesIdx;
};

//////////////////////////////////////////////////////////////////////////////////////////
// Implementation of memory helper for sequential version
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class MemHelperSeq : public MemHelperBase<algorithmFPType, cpu>
{
public:
    typedef MemHelperBase<algorithmFPType, cpu> super;
    MemHelperSeq(size_t nFeaturesIdx, size_t nDiffFeaturesMax, size_t nFeatureValuesMax)
        : super(nFeaturesIdx), _featureSample(nFeaturesIdx), _aGHSum(nDiffFeaturesMax), _aFeatureValue(nFeatureValuesMax)
    {}

    virtual bool init() DAAL_C11_OVERRIDE
    {
        return (!_featureSample.size() || _featureSample.get()) && //not required to allocate or allocated
               (!_aGHSum.size() || _aGHSum.get()) && (!_aFeatureValue.size() || _aFeatureValue.get());
    }

    virtual IndexType * getFeatureSampleBuf() DAAL_C11_OVERRIDE { return _featureSample.get(); }
    virtual void releaseFeatureSampleBuf(IndexType * buf) DAAL_C11_OVERRIDE {}

    virtual typename super::ghSumType * getGHSumBuf(size_t size) DAAL_C11_OVERRIDE
    {
        if (_aGHSum.size() < size) _aGHSum.reset(size);
        return _aGHSum.get();
    }

    //get buffer for the feature values to be used for the split at the current level
    virtual typename super::algorithmFPTypeVector * getFeatureValueBuf(size_t size) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(_aFeatureValue.size() >= size);
        return &_aFeatureValue;
    }
    //release the buffer
    virtual void releaseFeatureValueBuf(typename super::algorithmFPTypeVector * buf) DAAL_C11_OVERRIDE {}

    virtual typename super::IndexTypeVector * getSortedFeatureIdxBuf(size_t size) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(false);
        return nullptr;
    } //should never be called

    virtual void releaseSortedFeatureIdxBuf(typename super::IndexTypeVector * p) DAAL_C11_OVERRIDE {}

protected:
    typename super::IndexTypeVector _featureSample;
    typename super::ghSumTypeVector _aGHSum;
    typename super::algorithmFPTypeVector _aFeatureValue;
};

//////////////////////////////////////////////////////////////////////////////////////////
// Service class, keeps an array in ls and resizes it in local()
//////////////////////////////////////////////////////////////////////////////////////////
template <typename VectorType>
class lsVector : public ls<VectorType *>
{
public:
    typedef ls<VectorType *> super;
    explicit lsVector() : super([=]() -> VectorType * { return new VectorType(); }) {}
    ~lsVector()
    {
        this->reduce([](VectorType * ptr) {
            if (ptr) delete ptr;
        });
    }
    VectorType * local(size_t size)
    {
        auto ptr = super::local();
        if (ptr && (ptr->size() < size))
        {
            ptr->reset(size);
            if (!ptr->get())
            {
                this->release(ptr);
                ptr = nullptr;
            }
        }
        return ptr;
    }
};

//////////////////////////////////////////////////////////////////////////////////////////
// Service class, keeps an array in tls and resizes it in local()
//////////////////////////////////////////////////////////////////////////////////////////
template <typename VectorType>
class tlsVector : public tls<VectorType *>
{
public:
    typedef tls<VectorType *> super;
    explicit tlsVector() : super([=]() -> VectorType * { return new VectorType(); }) {}
    ~tlsVector()
    {
        this->reduce([](VectorType * ptr) {
            if (ptr) delete ptr;
        });
    }
    VectorType * local(size_t size)
    {
        auto ptr = super::local();
        if (ptr && (ptr->size() < size))
        {
            ptr->reset(size);
            if (!ptr->get()) ptr = nullptr;
        }
        return ptr;
    }
};

//////////////////////////////////////////////////////////////////////////////////////////
// Implementation of memory helper for threaded version
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class MemHelperThr : public MemHelperBase<algorithmFPType, cpu>
{
public:
    typedef MemHelperBase<algorithmFPType, cpu> super;
    MemHelperThr(size_t nFeaturesIdx)
        : super(nFeaturesIdx),
          _lsFeatureSample([=]() -> IndexType * { return services::internal::service_scalable_calloc<IndexType, cpu>(this->_nFeaturesIdx); })
    {}
    ~MemHelperThr()
    {
        _lsFeatureSample.reduce([](IndexType * ptr) {
            if (ptr) services::internal::service_scalable_free<IndexType, cpu>(ptr);
        });
    }

public:
    virtual bool init() DAAL_C11_OVERRIDE { return true; }
    virtual IndexType * getFeatureSampleBuf() DAAL_C11_OVERRIDE { return _lsFeatureSample.local(); }

    virtual void releaseFeatureSampleBuf(IndexType * p) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(p);
        _lsFeatureSample.release(p);
    }

    virtual typename super::ghSumType * getGHSumBuf(size_t size) DAAL_C11_OVERRIDE
    {
        auto ptr = _tlsGHSum.local(size);
        return ptr ? ptr->get() : nullptr;
    }

    //get buffer for the feature values to be used for the split at the current level
    virtual typename super::algorithmFPTypeVector * getFeatureValueBuf(size_t size) DAAL_C11_OVERRIDE { return _lsFeatureValueBuf.local(size); }

    //release the buffer
    virtual void releaseFeatureValueBuf(typename super::algorithmFPTypeVector * p) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(p);
        _lsFeatureValueBuf.release(p);
    }

    virtual typename super::IndexTypeVector * getSortedFeatureIdxBuf(size_t size) DAAL_C11_OVERRIDE { return _lsSortedFeatureIdxBuf.local(size); }

    virtual void releaseSortedFeatureIdxBuf(typename super::IndexTypeVector * p) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(p);
        _lsSortedFeatureIdxBuf.release(p);
    }

protected:
    ls<IndexType *> _lsFeatureSample;
    tlsVector<typename super::ghSumTypeVector> _tlsGHSum;
    lsVector<typename super::algorithmFPTypeVector> _lsFeatureValueBuf;
    lsVector<typename super::IndexTypeVector> _lsSortedFeatureIdxBuf;
};

//////////////////////////////////////////////////////////////////////////////////////////
// Job to be performed in one node
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
struct SplitJob
{
public:
    using TreeType     = gbt::internal::TreeImpRegression<>;
    using NodeType     = TreeType::NodeType;
    using ImpurityType = ImpurityData<algorithmFPType, cpu>;

    SplitJob(const SplitJob & o) : iStart(o.iStart), n(o.n), level(o.level), imp(o.imp), res(o.res) {}
    SplitJob(size_t _iStart, size_t _n, size_t _level, const ImpurityType & _imp, NodeType::Base *& _res)
        : iStart(_iStart), n(_n), level(_level), imp(_imp), res(_res)
    {}

public:
    const size_t iStart;
    const size_t n;
    const size_t level;
    const ImpurityType imp;
    NodeType::Base *& res;

    bool doMerged;
    size_t prevStart;
    size_t prevN;
    size_t iFeaturePrev;
    size_t splitPointPrev;
    NodeType::Base * prevRes;
};

//////////////////////////////////////////////////////////////////////////////////////////
// Base tree builder class
//////////////////////////////////////////////////////////////////////////////////////////
class TreeBuilderBase : public Base
{
public:
    virtual services::Status init()                                                     = 0;
    virtual services::Status run(gbt::internal::GbtDecisionTree *& pRes, HomogenNumericTable<double> *& pTblImp,
                                 HomogenNumericTable<int> *& pTblSmplCnt, size_t iTree) = 0;
};

class GbtTask
{
public:
    virtual GbtTask * execute() = 0;
    virtual void operator()() { execute(); };
    virtual void getNextTasks(GbtTask ** newTasks, size_t & nTasks) {};
    virtual ~GbtTask() {};
};

template <typename algorithmFPType, typename BinIndexType, CpuType cpu>
class TrainBatchTaskBaseXBoost;
template <typename algorithmFPType, CpuType cpu>
class MemHelperBase;

template <CpuType cpu>
struct EmptyResult
{
    template <typename DataType>
    void release(DataType & data)
    {}
};

template <typename PartialResult, CpuType cpu>
struct MergedResult
{
    template <typename DataType>
    void release(DataType & data)
    {
        for (size_t i = 0; i < res.size(); ++i)
        {
            res[i].release(data);
        }
        res.~TVector<PartialResult, cpu, ScalableAllocator<cpu> >();
        service_scalable_free<void, cpu>(this);
    }
    MergedResult(size_t size) : res(size) {}
    TVector<PartialResult, cpu, ScalableAllocator<cpu> > res;
};

template <CpuType cpu>
struct LoopHelper
{
    template <typename Func>
    static void run(bool inParallel, size_t nBlocks, Func func)
    {
        if (inParallel)
        {
            daal::threader_for(nBlocks, nBlocks, [&](size_t i) { func(i); });
        }
        else
        {
            for (size_t i = 0; i < nBlocks; ++i) func(i);
        }
    }
};

template <typename T, CpuType cpu>
class GroupOfStorages
{
public:
    GroupOfStorages(size_t nElems) : storages(nElems)
    {
        for (size_t i = 0; i < nElems; i++) new (&storages[i]) BuffersStorage();
    }

    ~GroupOfStorages()
    {
        for (size_t i = 0; i < storages.size(); i++) storages[i].~BuffersStorage();
    }

    void add(size_t idx, size_t size, size_t nElem)
    {
        if (nElem && size) storages[idx].init(size, nElem);
    }

    class BuffersStorage
    {
    public:
        BuffersStorage() {}

        void init(size_t bufferSize, size_t NElem)
        {
            _capacity   = NElem;
            _curIdx     = 0;
            _bufferSize = bufferSize;

            buffers.resize(NElem);
            T * ptr = allocate(NElem);

            for (size_t i = 0; i < NElem; ++i)
            {
                buffers[i] = ptr + i * _bufferSize;
            }
        }

        ~BuffersStorage() { destroy(); }

        T * getBlockFromStorage()
        {
            AUTOLOCK(_mutex);

            if (_curIdx == _capacity)
            {
                addBlocks(6);
            }
            return buffers[_curIdx++];
        }

        size_t size() { return _curIdx; }

        size_t bufferSize() { return _bufferSize; }

        void returnBlockToStorage(T * ptr)
        {
            if (!!ptr)
            {
                AUTOLOCK(_mutex);
                buffers[--_curIdx] = ptr;
            }
        }

    protected:
        void addBlocks(size_t nNewBlocks) // no thread-safe
        {
            T * ptr = allocate(nNewBlocks);

            buffers.resize(nNewBlocks + _capacity);
            for (size_t i = 0; i < nNewBlocks; ++i)
            {
                buffers[i + _capacity] = ptr + i * _bufferSize;
            }

            _capacity = nNewBlocks + _capacity;
        }

        T * allocate(size_t nBlocks)
        {
            alloc.pushBack(service_scalable_calloc<T, cpu>(nBlocks * _bufferSize));
            return alloc[alloc.size() - 1];
        }

        void destroy()
        {
            for (size_t i = 0; i < alloc.size(); ++i)
            {
                service_scalable_free<T, cpu>(alloc[i]);
                alloc[i] = nullptr;
            }
        }

        daal::Mutex _mutex;
        TVector<T *, cpu, ScalableAllocator<cpu> > buffers;
        TVector<T *, cpu, ScalableAllocator<cpu> > alloc;

        size_t _capacity;
        size_t _curIdx;
        size_t _bufferSize;
    };

    BuffersStorage & get(size_t idx) { return storages[idx]; }

    size_t size() { return storages.size(); }

    TVector<BuffersStorage, cpu, ScalableAllocator<cpu> > storages;
};

template <typename T, CpuType cpu>
class GHSumsStorage
{
public:
    GHSumsStorage(size_t nGH, size_t nInitElems) : _nGH(nGH), _capacity(nInitElems), _curIdx(0) { allocate(_capacity); }

    ~GHSumsStorage() { destroy(); }

    T * getBlockFromStorage()
    {
        AUTOLOCK(_mutex);

        if (_curIdx == _capacity)
        {
            addBlocks(2);
        }
        return alloc[_curIdx++];
    }

    void returnBlockToStorage(T * ptr)
    {
        if (!!ptr)
        {
            AUTOLOCK(_mutex);
            alloc[--_curIdx] = ptr;
        }
    }

protected:
    void addBlocks(size_t nNewBlocks) // no thread-safe
    {
        allocate(nNewBlocks);
        _capacity = nNewBlocks + _capacity;
    }

    void allocate(size_t nBlocks)
    {
        for (size_t i = 0; i < nBlocks; ++i)
        {
            alloc.pushBack(new (service_scalable_calloc<T, cpu>(1)) T(_nGH));
        }
    }

    void destroy()
    {
        for (size_t i = 0; i < alloc.size(); ++i)
        {
            alloc[i]->~T();
            service_scalable_free<T, cpu>(alloc[i]);
            alloc[i] = nullptr;
        }
    }

    daal::Mutex _mutex;
    size_t _nGH;
    TVector<T *, cpu, ScalableAllocator<cpu> > alloc;
    size_t _capacity;
    size_t _curIdx;
};

template <typename GHSumType, CpuType cpu>
struct GHSumForTLS
{
    GHSumForTLS(GHSumType * p) : ghSum(p), isInitilized(false) {}

    GHSumType * ghSum;
    bool isInitilized;
};

template <typename T, typename algorithmFPType, CpuType cpu,
          typename Allocator = services::internal::ScalableMalloc<ghSum<algorithmFPType, cpu>, cpu> >
class TlsGHSumMerge : public daal::tls<T *>
{
public:
    using super     = daal::tls<T *>;
    using GHSumType = ghSum<algorithmFPType, cpu>;

    TlsGHSumMerge(size_t n) : super([=]() -> T * { return new (service_scalable_calloc<T, cpu>(1)) T(Allocator::allocate(n)); }) {}

    ~TlsGHSumMerge()
    {
        this->reduce([](T * ptr) -> void {
            Allocator::deallocate(ptr->ghSum);
            ptr->~T();
            service_scalable_free<T, cpu>(ptr);
        });
    }

    void reduceTo(algorithmFPType ** res, size_t & size)
    {
        size = 0;
        this->reduce([&](T * ptr) -> void {
            if (!ptr->isInitilized)
            {
                return;
            }

            res[size++] = (algorithmFPType *)ptr->ghSum;
        });
    }

    void release()
    {
        this->reduce([](T * ptr) -> void { ptr->isInitilized = false; });
    }
};

template <typename algorithmFPType, typename BinIndexType, CpuType cpu>
struct GlobalStorages
{
    using GHSumType = ghSum<algorithmFPType, cpu>;
    using TlsType   = TlsGHSumMerge<GHSumForTLS<GHSumType, cpu>, algorithmFPType, cpu>;

    GlobalStorages(size_t nFeatures, size_t nStor, size_t nUniq, size_t nGlobal)
        : singleGHSums(nStor), GHForCols(nUniq, nGlobal), nUniquesArr(nFeatures)
    {}

    GroupOfStorages<GHSumType, cpu> singleGHSums;
    GHSumsStorage<TlsType, cpu> GHForCols;
    TVector<size_t, cpu, ScalableAllocator<cpu> > nUniquesArr;
    size_t nDiffFeatMax;

    BinIndexType * newFI;
};

template <typename algorithmFPType, typename RowIndexType, typename BinIndexType, CpuType cpu>
class SharedDataForTree
{
public:
    using MemHelperType = MemHelperBase<algorithmFPType, cpu>;
    using CtxType       = TrainBatchTaskBaseXBoost<algorithmFPType, BinIndexType, cpu>;
    using TreeType      = gbt::internal::TreeImpRegression<>;

    SharedDataForTree(CtxType & _ctx, RowIndexType * _bestSplitIdxBuf, RowIndexType * _aIdx, MemHelperType * _memHelper, size_t _iTree,
                      TreeType & _tree, daal::Mutex & _mtAlloc)
        : ctx(_ctx), bestSplitIdxBuf(_bestSplitIdxBuf), aIdx(_aIdx), memHelper(_memHelper), iTree(_iTree), tree(_tree), mtAlloc(_mtAlloc)
    {}

    GlobalStorages<algorithmFPType, BinIndexType, cpu> * GH_SUMS_BUF;
    CtxType & ctx;
    int * aIdx;
    MemHelperType * memHelper;
    size_t iTree;
    RowIndexType * bestSplitIdxBuf;
    TreeType & tree;
    daal::Mutex & mtAlloc;
};

} /* namespace internal */
} /* namespace training */
} /* namespace gbt */
} /* namespace algorithms */
} /* namespace daal */

#endif
