/* file: svm_train_common.h */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef __SVM_TRAIN_COMMON_H__
#define __SVM_TRAIN_COMMON_H__

#include "src/data_management/service_numeric_table.h"
#include "src/services/service_profiler.h"
#include "src/services/service_utils.h"
#include "src/algorithms/service_hash_table.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace internal
{
using namespace daal::services::internal;
using namespace daal::internal;

enum SVMVectorStatus
{
    free     = 0x0,
    up       = 0x1,
    low      = 0x2,
    shrink   = 0x4,
    positive = 0x8,
    negative = 0x10
};

enum class SignNuType
{
    none,
    positive,
    negative
};

template <typename algorithmFPType, CpuType cpu>
struct HelperTrainSVM
{
    DAAL_FORCEINLINE static bool isUpper(const algorithmFPType y, const algorithmFPType alpha, const algorithmFPType C,
                                         SignNuType signNuType = SignNuType::none)
    {
        return checkLabel(y, signNuType) && ((y > 0 && alpha < C) || (y < 0 && alpha > 0));
    }
    DAAL_FORCEINLINE static bool isLower(const algorithmFPType y, const algorithmFPType alpha, const algorithmFPType C,
                                         SignNuType signNuType = SignNuType::none)
    {
        return checkLabel(y, signNuType) && ((y > 0 && alpha > 0) || (y < 0 && alpha < C));
    }

    DAAL_FORCEINLINE static algorithmFPType WSSi(size_t nActiveVectors, const algorithmFPType * grad, const char * I, int & Bi,
                                                 SignNuType signNuType = SignNuType::none);

    DAAL_FORCEINLINE static void WSSjLocal(const size_t jStart, const size_t jEnd, const algorithmFPType * KiBlock,
                                           const algorithmFPType * kernelDiag, const algorithmFPType * grad, const char * I,
                                           const algorithmFPType GMin, const algorithmFPType Kii, const algorithmFPType tau, int & Bj,
                                           algorithmFPType & GMax, algorithmFPType & GMax2, algorithmFPType & delta,
                                           SignNuType signNuType = SignNuType::none);

    DAAL_FORCEINLINE static bool checkLabel(const algorithmFPType y, SignNuType signNuType = SignNuType::none)
    {
        return (signNuType == SignNuType::none) || ((signNuType == SignNuType::positive) && (y > 0))
               || ((signNuType == SignNuType::negative) && (y < 0));
    }

private:
    DAAL_FORCEINLINE static void WSSjLocalBaseline(const size_t jStart, const size_t jEnd, const algorithmFPType * KiBlock,
                                                   const algorithmFPType * kernelDiag, const algorithmFPType * grad, const char * I,
                                                   const algorithmFPType GMin, const algorithmFPType Kii, const algorithmFPType tau, int & Bj,
                                                   algorithmFPType & GMax, algorithmFPType & GMax2, algorithmFPType & delta,
                                                   SignNuType signNuType = SignNuType::none);

    DAAL_FORCEINLINE static char getSign(SignNuType signNuType)
    {
        char sign = positive | negative;
        if (signNuType == SignNuType::positive)
        {
            sign = positive;
        }
        else if (signNuType == SignNuType::negative)
        {
            sign = negative;
        }
        return sign;
    }
};

template <CpuType cpu, typename TKey>
class LRUCache
{
public:
    LRUCache(const size_t capacity) : _capacity(capacity), _hashmap(capacity)
    {
        _freeIndexCache = -1;
        _count          = 0;
        _head           = nullptr;
        _tail           = nullptr;
    }

    ~LRUCache()
    {
        LRUNode * curr = _head;
        while (curr)
        {
            LRUNode * next = curr->next;
            delete curr;
            curr = next;
        }
    }

    void put(const TKey key);
    int64_t getFreeIndex() const { return _freeIndexCache; }
    int64_t get(const TKey key);

private:
    class LRUNode
    {
    public:
        DAAL_NEW_DELETE();
        static LRUNode * create(const TKey key, int64_t value)
        {
            auto val = new LRUNode(key, value);
            if (val) return val;
            delete val;
            return nullptr;
        }

        TKey getKey() const { return key_; }
        int64_t getValue() const { return value_; }

        void setKey(const TKey key) { key_ = key; }
        void setValue(const int64_t value) { value_ = value; }

    public:
        LRUNode * next;
        LRUNode * prev;

    private:
        LRUNode(const TKey key, int64_t value) : key_(key), value_(value), next(nullptr), prev(nullptr) {}
        TKey key_;
        int64_t value_;
    };

    const size_t _capacity;
    algorithms::internal::HashTable<cpu, TKey, LRUNode *> _hashmap;
    LRUNode * _head;
    LRUNode * _tail;
    size_t _count;
    int64_t _freeIndexCache;

private:
    void enqueue(LRUNode * node);
    int64_t dequeue();
};

template <typename algorithmFPType, CpuType cpu>
class SubDataTaskBase
{
public:
    DAAL_NEW_DELETE();
    virtual ~SubDataTaskBase() {}

    virtual services::Status copyDataByIndices(const uint32_t * wsIndices, const size_t nSubsetVectors, const NumericTablePtr & xTable) = 0;

    NumericTablePtr getTableData() const { return _dataTable; }

protected:
    SubDataTaskBase(const size_t nMaxSubsetVectors, const size_t dataSize) : _nMaxSubsetVectors(nMaxSubsetVectors), _data(dataSize) {}
    SubDataTaskBase(const size_t nMaxSubsetVectors) : _nMaxSubsetVectors(nMaxSubsetVectors) {}

    bool isValid() const { return _data.get(); }

protected:
    size_t _nMaxSubsetVectors;
    TArray<algorithmFPType, cpu> _data;
    NumericTablePtr _dataTable;
};

template <typename algorithmFPType, CpuType cpu>
class SubDataTaskCSR : public SubDataTaskBase<algorithmFPType, cpu>
{
public:
    using super = SubDataTaskBase<algorithmFPType, cpu>;
    static SubDataTaskCSR * create(const NumericTablePtr & xTable, const size_t nMaxSubsetVectors)
    {
        auto val = new SubDataTaskCSR(xTable, nMaxSubsetVectors);
        if (val && val->isValid()) return val;
        delete val;
        val = nullptr;
        return nullptr;
    }

private:
    bool isValid() const { return super::isValid() && _colIndices.get() && this->_dataTable.get(); }

    SubDataTaskCSR(const NumericTablePtr & xTable, const size_t nMaxSubsetVectors) : super(nMaxSubsetVectors)
    {
        const size_t p                        = xTable->getNumberOfColumns();
        const size_t nRows                    = xTable->getNumberOfRows();
        CSRNumericTableIface * const csrIface = dynamic_cast<CSRNumericTableIface * const>(const_cast<NumericTable *>(xTable.get()));
        if (!csrIface) return;
        const size_t maxDataSize = csrIface->getDataSize();
        this->_data.reset(maxDataSize);
        _colIndices.reset(maxDataSize + nMaxSubsetVectors + 1);
        _rowOffsets = _colIndices.get() + maxDataSize;

        if (this->_data.get())
            this->_dataTable = CSRNumericTable::create(this->_data.get(), _colIndices.get(), _rowOffsets, p, nMaxSubsetVectors,
                                                       CSRNumericTableIface::CSRIndexing::oneBased);
    }

    virtual services::Status copyDataByIndices(const uint32_t * wsIndices, const size_t nSubsetVectors, const NumericTablePtr & xTable);

private:
    TArray<size_t, cpu> _colIndices;
    size_t * _rowOffsets;
};

template <typename algorithmFPType, CpuType cpu>
class SubDataTaskDense : public SubDataTaskBase<algorithmFPType, cpu>
{
public:
    using super = SubDataTaskBase<algorithmFPType, cpu>;
    static SubDataTaskDense * create(const size_t nFeatures, const size_t nMaxSubsetVectors)
    {
        auto val = new SubDataTaskDense(nFeatures, nMaxSubsetVectors);
        if (val && val->isValid()) return val;
        delete val;
        val = nullptr;
        return nullptr;
    }

    virtual services::Status copyDataByIndices(const uint32_t * wsIndices, const size_t nSubsetVectors, const NumericTablePtr & xTable);

private:
    bool isValid() const { return super::isValid() && this->_dataTable.get(); }

    SubDataTaskDense(const size_t nFeatures, const size_t nMaxSubsetVectors) : super(nMaxSubsetVectors, nFeatures * nMaxSubsetVectors)
    {
        services::Status status;
        if (this->_data.get())
            this->_dataTable = HomogenNumericTableCPU<algorithmFPType, cpu>::create(this->_data.get(), nFeatures, nMaxSubsetVectors, &status);
    }
};

template <typename algorithmFPType, CpuType cpu>
using SubDataTaskBasePtr = services::SharedPtr<SubDataTaskBase<algorithmFPType, cpu> >;

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
