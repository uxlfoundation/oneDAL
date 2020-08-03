/* file: service_numeric_table.h */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
//  CPU-specified homogeneous numeric table
//--
*/

#ifndef __SERVICE_NUMERIC_TABLE_H__
#define __SERVICE_NUMERIC_TABLE_H__

#include "data_management/data/homogen_numeric_table.h"
#include "data_management/data/soa_numeric_table.h"
#include "data_management/data/csr_numeric_table.h"
#include "data_management/data/symmetric_matrix.h"
#include "service/kernel/service_defines.h"
#include "externals/service_memory.h"
#include "service/kernel/service_arrays.h"

using namespace daal::data_management;
using namespace daal::data_management::internal;

#define DEFINE_TABLE_BLOCK_EX(BlockType, targetVariable, ...)    \
    BlockType<algorithmFPType, cpu> targetVariable(__VA_ARGS__); \
    DAAL_CHECK_BLOCK_STATUS(targetVariable);

#define DEFINE_TABLE_BLOCK(BlockType, targetVariable, table) DEFINE_TABLE_BLOCK_EX(BlockType, targetVariable, table, 0, (table)->getNumberOfRows())

namespace daal
{
namespace internal
{
template <CpuType cpu>
class NumericTableFeatureCPU : public NumericTableFeature
{
public:
    NumericTableFeatureCPU() : NumericTableFeature() {}
    virtual ~NumericTableFeatureCPU() {}
};

template <CpuType cpu>
class NumericTableDictionaryCPU : public NumericTableDictionary
{
public:
    NumericTableDictionaryCPU(size_t nfeat)
    {
        _nfeat         = 0;
        _dict          = nullptr;
        _featuresEqual = DictionaryIface::equal;
        if (nfeat) setNumberOfFeatures(nfeat);
    };

    NumericTableDictionaryCPU(size_t nfeat, FeaturesEqual featuresEqual, services::Status & st)
    {
        _nfeat         = 0;
        _dict          = 0;
        _featuresEqual = featuresEqual;
        if (nfeat)
        {
            st |= setNumberOfFeatures(nfeat);
        }
    };

    static services::SharedPtr<NumericTableDictionaryCPU<cpu> > create(size_t nfeat, FeaturesEqual featuresEqual = notEqual,
                                                                       services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(NumericTableDictionaryCPU, DAAL_TEMPLATE_ARGUMENTS(cpu), nfeat, featuresEqual);
    }

    services::Status setAllFeatures(const NumericTableFeature & defaultFeature) DAAL_C11_OVERRIDE
    {
        if (_featuresEqual == DictionaryIface::equal)
        {
            if (_nfeat > 0)
            {
                _dict[0] = defaultFeature;
            }
        }
        else
        {
            for (size_t i = 0; i < _nfeat; i++)
            {
                _dict[i] = defaultFeature;
            }
        }
        return services::Status();
    }

    template <typename featureType>
    services::Status setAllFeatures()
    {
        NumericTableFeatureCPU<cpu> defaultFeature;
        defaultFeature.template setType<featureType>();
        return setAllFeatures(defaultFeature);
    }

    services::Status setNumberOfFeatures(size_t nfeat) DAAL_C11_OVERRIDE
    {
        _nfeat = nfeat;
        if (_featuresEqual == DictionaryIface::equal)
        {
            _dict = new NumericTableFeatureCPU<cpu>[1];
        }
        else
        {
            _dict = new NumericTableFeatureCPU<cpu>[_nfeat];
        }
        return services::Status();
    }
};

template <typename T, CpuType cpu>
class HomogenNumericTableCPU
{};

template <CpuType cpu>
class HomogenNumericTableCPU<float, cpu> : public HomogenNumericTable<float>
{
public:
    HomogenNumericTableCPU(float * const ptr, size_t featnum, size_t obsnum, services::Status & st)
        : HomogenNumericTable<float>(services::SharedPtr<NumericTableDictionaryCPU<cpu> >(new NumericTableDictionaryCPU<cpu>(featnum)), st)
    {
        _ptr       = services::SharedPtr<byte>((byte *)ptr, services::EmptyDeleter());
        _memStatus = userAllocated;

        NumericTableFeature df;
        df.setType<float>();
        st.add(_ddict->setAllFeatures(df));

        st.add(setNumberOfRows(obsnum));
    }

    static services::SharedPtr<HomogenNumericTableCPU<float, cpu> > create(float * const ptr, size_t featnum, size_t obsnum,
                                                                           services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(HomogenNumericTableCPU, DAAL_TEMPLATE_ARGUMENTS(float, cpu), ptr, featnum, obsnum);
    }

    HomogenNumericTableCPU(size_t featnum, size_t obsnum, services::Status & st)
        : HomogenNumericTable<float>(services::SharedPtr<NumericTableDictionaryCPU<cpu> >(new NumericTableDictionaryCPU<cpu>(featnum)), st)
    {
        st.add(setNumberOfRows(obsnum));

        NumericTableFeature df;
        df.setType<float>();
        st.add(_ddict->setAllFeatures(df));

        st.add(allocateDataMemory());
    }

    static services::SharedPtr<HomogenNumericTableCPU<float, cpu> > create(size_t featnum, size_t obsnum, services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(HomogenNumericTableCPU, DAAL_TEMPLATE_ARGUMENTS(float, cpu), featnum, obsnum);
    }

    virtual ~HomogenNumericTableCPU() {}

    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<double>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<float>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<int>(vector_idx, vector_num, rwflag, block);
    }

    services::Status releaseBlockOfRows(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE { return releaseTBlock<double>(block); }
    services::Status releaseBlockOfRows(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE { return releaseTBlock<float>(block); }
    services::Status releaseBlockOfRows(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE { return releaseTBlock<int>(block); }

    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<double>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<float>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<int>(feature_idx, vector_idx, value_num, rwflag, block);
    }

    services::Status releaseBlockOfColumnValues(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE { return releaseTFeature<double>(block); }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE { return releaseTFeature<float>(block); }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE { return releaseTFeature<int>(block); }
};

template <CpuType cpu>
class HomogenNumericTableCPU<double, cpu> : public HomogenNumericTable<double>
{
public:
    HomogenNumericTableCPU(double * const ptr, size_t featnum, size_t obsnum, services::Status & st)
        : HomogenNumericTable<double>(services::SharedPtr<NumericTableDictionaryCPU<cpu> >(new NumericTableDictionaryCPU<cpu>(featnum)), st)
    {
        NumericTableFeature df;
        df.setType<double>();
        st.add(_ddict->setAllFeatures(df));

        _ptr       = services::SharedPtr<byte>((byte *)ptr, services::EmptyDeleter());
        _memStatus = userAllocated;
        st.add(setNumberOfRows(obsnum));
    }

    static services::SharedPtr<HomogenNumericTableCPU<double, cpu> > create(double * const ptr, size_t featnum, size_t obsnum,
                                                                            services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(HomogenNumericTableCPU, DAAL_TEMPLATE_ARGUMENTS(double, cpu), ptr, featnum, obsnum);
    }

    HomogenNumericTableCPU(size_t featnum, size_t obsnum, services::Status & st)
        : HomogenNumericTable<double>(services::SharedPtr<NumericTableDictionaryCPU<cpu> >(new NumericTableDictionaryCPU<cpu>(featnum)), st)
    {
        st.add(setNumberOfRows(obsnum));

        NumericTableFeature df;
        df.setType<double>();
        st.add(_ddict->setAllFeatures(df));

        st.add(allocateDataMemory());
    }

    static services::SharedPtr<HomogenNumericTableCPU<double, cpu> > create(size_t featnum, size_t obsnum, services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(HomogenNumericTableCPU, DAAL_TEMPLATE_ARGUMENTS(double, cpu), featnum, obsnum);
    }

    virtual ~HomogenNumericTableCPU() {}

    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<double>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<float>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<int>(vector_idx, vector_num, rwflag, block);
    }

    services::Status releaseBlockOfRows(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE { return releaseTBlock<double>(block); }
    services::Status releaseBlockOfRows(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE { return releaseTBlock<float>(block); }
    services::Status releaseBlockOfRows(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE { return releaseTBlock<int>(block); }

    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<double>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<float>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<int>(feature_idx, vector_idx, value_num, rwflag, block);
    }

    services::Status releaseBlockOfColumnValues(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE { return releaseTFeature<double>(block); }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE { return releaseTFeature<float>(block); }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE { return releaseTFeature<int>(block); }
};

template <CpuType cpu>
class HomogenNumericTableCPU<int, cpu> : public HomogenNumericTable<int>
{
public:
    HomogenNumericTableCPU(int * const ptr, size_t featnum, size_t obsnum, services::Status & st)
        : HomogenNumericTable<int>(services::SharedPtr<NumericTableDictionaryCPU<cpu> >(new NumericTableDictionaryCPU<cpu>(featnum)), st)
    {
        NumericTableFeature df;
        df.setType<int>();
        st.add(_ddict->setAllFeatures(df));

        _ptr       = services::SharedPtr<byte>((byte *)ptr, services::EmptyDeleter());
        _memStatus = userAllocated;
        st.add(setNumberOfRows(obsnum));
    }

    static services::SharedPtr<HomogenNumericTableCPU<int, cpu> > create(int * const ptr, size_t featnum, size_t obsnum,
                                                                         services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(HomogenNumericTableCPU, DAAL_TEMPLATE_ARGUMENTS(int, cpu), ptr, featnum, obsnum);
    }

    HomogenNumericTableCPU(size_t featnum, size_t obsnum, services::Status & st)
        : HomogenNumericTable<int>(services::SharedPtr<NumericTableDictionaryCPU<cpu> >(new NumericTableDictionaryCPU<cpu>(featnum)), st)
    {
        NumericTableFeature df;
        df.setType<int>();
        st.add(_ddict->setAllFeatures(df));

        st.add(setNumberOfRows(obsnum));
        st.add(allocateDataMemory());
    }

    static services::SharedPtr<HomogenNumericTableCPU<int, cpu> > create(size_t featnum, size_t obsnum, services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(HomogenNumericTableCPU, DAAL_TEMPLATE_ARGUMENTS(int, cpu), featnum, obsnum);
    }

    virtual ~HomogenNumericTableCPU() {}

    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<double>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<float>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<int>(vector_idx, vector_num, rwflag, block);
    }

    services::Status releaseBlockOfRows(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE { return releaseTBlock<double>(block); }
    services::Status releaseBlockOfRows(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE { return releaseTBlock<float>(block); }
    services::Status releaseBlockOfRows(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE { return releaseTBlock<int>(block); }

    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<double>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<float>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<int>(feature_idx, vector_idx, value_num, rwflag, block);
    }

    services::Status releaseBlockOfColumnValues(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE { return releaseTFeature<double>(block); }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE { return releaseTFeature<float>(block); }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE { return releaseTFeature<int>(block); }

private:
    NumericTableDictionary * _cpuDict;
};

template <CpuType cpu>
class SOANumericTableCPU : public SOANumericTable
{
public:
    SOANumericTableCPU(size_t nColumns, size_t nRows, DictionaryIface::FeaturesEqual featuresEqual, services::Status & st)
        : SOANumericTable(nColumns, nRows, featuresEqual, st)
    {}

    static services::SharedPtr<SOANumericTableCPU<cpu> > create(size_t nColumns = 0, size_t nRows = 0,
                                                                DictionaryIface::FeaturesEqual featuresEqual = DictionaryIface::notEqual,
                                                                services::Status * stat                      = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(SOANumericTableCPU, DAAL_TEMPLATE_ARGUMENTS(cpu), nColumns, nRows, featuresEqual);
    }

    SOANumericTableCPU(NumericTableDictionaryPtr ddict, size_t nRows, AllocationFlag memoryAllocationFlag, services::Status & st)
        : SOANumericTable(ddict, nRows, memoryAllocationFlag)
    {}

    static services::SharedPtr<SOANumericTableCPU<cpu> > create(NumericTableDictionaryPtr ddict, size_t nRows,
                                                                AllocationFlag memoryAllocationFlag = notAllocate, services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(SOANumericTableCPU, DAAL_TEMPLATE_ARGUMENTS(cpu), ddict, nRows, memoryAllocationFlag);
    }

    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<double>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<float>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<int>(vector_idx, vector_num, rwflag, block);
    }

    services::Status releaseBlockOfRows(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE { return releaseTBlock<double>(block); }
    services::Status releaseBlockOfRows(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE { return releaseTBlock<float>(block); }
    services::Status releaseBlockOfRows(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE { return releaseTBlock<int>(block); }

    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<double>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<float>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<int>(feature_idx, vector_idx, value_num, rwflag, block);
    }

    services::Status releaseBlockOfColumnValues(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE { return releaseTFeature<double>(block); }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE { return releaseTFeature<float>(block); }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE { return releaseTFeature<int>(block); }

    virtual ~SOANumericTableCPU() {}

protected:
    template <typename T>
    services::Status getTBlock(size_t idx, size_t nrows, ReadWriteMode rwFlag, BlockDescriptor<T> & block)
    {
        size_t ncols = getNumberOfColumns();
        size_t nobs  = getNumberOfRows();
        block.setDetails(0, idx, rwFlag);

        if (idx >= nobs)
        {
            block.resizeBuffer(ncols, 0);
            return services::Status();
        }

        nrows = (idx + nrows < nobs) ? nrows : nobs - idx;

        if (!block.resizeBuffer(ncols, nrows))
        {
            return services::Status(services::ErrorMemoryAllocationFailed);
        }

        if (!(block.getRWFlag() & (int)readOnly)) return services::Status();

        T * buffer    = block.getBlockPtr();
        bool computed = false;

        if (isHomogeneous())
        {
            NumericTableFeature & f = (*_ddict)[0];

            if ((int)data_management::internal::getConversionDataType<T>() == (int)f.indexType)
            {
                DAAL_CHECK(_arrOffsets, services::ErrorNullPtr)
                T const * ptrMin = (T *)(_arrays[_index].get()) + idx;
                computed         = data_management::internal::getVector<T>()(nrows, ncols, buffer, ptrMin, _arrOffsets);
            }
        }
        if (!computed)
        {
            size_t di = 32;
            T lbuf[32];

            for (size_t i = 0; i < nrows; i += di)
            {
                if (i + di > nrows)
                {
                    di = nrows - i;
                }

                for (size_t j = 0; j < ncols; ++j)
                {
                    NumericTableFeature & f = (*_ddict)[j];

                    char * ptr = (char *)_arrays[j].get() + (idx + i) * f.typeSize;

                    data_management::internal::getVectorUpCast(f.indexType, data_management::internal::getConversionDataType<T>())(di, ptr, lbuf);

                    for (size_t k = 0; k < di; ++k)
                    {
                        buffer[(i + k) * ncols + j] = lbuf[k];
                    }
                }
            }
        }

        return services::Status();
    }

    template <typename T>
    services::Status releaseTBlock(BlockDescriptor<T> & block)
    {
        if (block.getRWFlag() & (int)writeOnly)
        {
            size_t ncols = getNumberOfColumns();
            size_t nrows = block.getNumberOfRows();
            size_t idx   = block.getRowsOffset();
            T lbuf[32];

            size_t di = 32;

            T * blockPtr = block.getBlockPtr();

            for (size_t i = 0; i < nrows; i += di)
            {
                if (i + di > nrows)
                {
                    di = nrows - i;
                }

                for (size_t j = 0; j < ncols; j++)
                {
                    NumericTableFeature & f = (*_ddict)[j];

                    char * ptr = (char *)_arrays[j].get() + (idx + i) * f.typeSize;

                    for (size_t ii = 0; ii < di; ii++)
                    {
                        lbuf[ii] = blockPtr[(i + ii) * ncols + j];
                    }

                    data_management::internal::getVectorDownCast(f.indexType, data_management::internal::getConversionDataType<T>())(di, lbuf, ptr);
                }
            }
        }
        block.reset();
        return services::Status();
    }

    template <typename T>
    services::Status getTFeature(size_t feat_idx, size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T> & block)
    {
        size_t ncols = getNumberOfColumns();
        size_t nobs  = getNumberOfRows();
        block.setDetails(feat_idx, idx, rwFlag);

        if (idx >= nobs)
        {
            block.resizeBuffer(1, 0);
            return services::Status();
        }

        nrows = (idx + nrows < nobs) ? nrows : nobs - idx;

        NumericTableFeature & f = (*_ddict)[feat_idx];

        if (features::internal::getIndexNumType<T>() == f.indexType)
        {
            block.setPtr(&(_arrays[feat_idx]), _arrays[feat_idx].get() + idx * f.typeSize, 1, nrows);
        }
        else
        {
            byte * location = _arrays[feat_idx].get() + idx * f.typeSize;

            if (!block.resizeBuffer(1, nrows))
            {
                return services::Status(services::ErrorMemoryAllocationFailed);
            }

            if (!(block.getRWFlag() & (int)readOnly)) return services::Status();

            data_management::internal::getVectorUpCast(f.indexType, data_management::internal::getConversionDataType<T>())(nrows, location,
                                                                                                                           block.getBlockPtr());
        }
        return services::Status();
    }

    template <typename T>
    services::Status releaseTFeature(BlockDescriptor<T> & block)
    {
        if (block.getRWFlag() & (int)writeOnly)
        {
            size_t feat_idx = block.getColumnsOffset();

            NumericTableFeature & f = (*_ddict)[feat_idx];

            if (features::internal::getIndexNumType<T>() != f.indexType)
            {
                char * ptr = (char *)_arrays[feat_idx].get() + block.getRowsOffset() * f.typeSize;

                data_management::internal::getVectorDownCast(f.indexType, data_management::internal::getConversionDataType<T>())(
                    block.getNumberOfRows(), block.getBlockPtr(), ptr);
            }
        }
        block.reset();
        return services::Status();
    }
};

template <typename algorithmFPType, typename algorithmFPAccessType, CpuType cpu, ReadWriteMode mode, typename NumericTableType>
class GetRows
{
public:
    GetRows(NumericTableType & data, size_t iStartFrom, size_t nRows) : _data(&data) { getBlock(iStartFrom, nRows); }
    GetRows(NumericTableType * data, size_t iStartFrom, size_t nRows) : _data(data), _toReleaseFlag(false)
    {
        if (_data)
        {
            getBlock(iStartFrom, nRows);
        }
    }
    GetRows(NumericTableType * data = nullptr) : _data(data), _toReleaseFlag(false) {}
    GetRows(NumericTableType & data) : _data(&data), _toReleaseFlag(false) {}
    ~GetRows() { release(); }
    algorithmFPAccessType * get() { return _data ? _block.getBlockPtr() : nullptr; }
    algorithmFPAccessType * next(size_t iStartFrom, size_t nRows)
    {
        if (!_data) return nullptr;

        if (_toReleaseFlag)
        {
            _status = _data->releaseBlockOfRows(_block);
        }

        return getBlock(iStartFrom, nRows);
    }
    algorithmFPAccessType * set(NumericTableType * data, size_t iStartFrom, size_t nRows)
    {
        release();
        if (data)
        {
            _data = data;
            return getBlock(iStartFrom, nRows);
        }
        return nullptr;
    }
    algorithmFPAccessType * set(NumericTableType & data, size_t iStartFrom, size_t nRows)
    {
        release();
        _data = &data;
        return getBlock(iStartFrom, nRows);
    }
    void release()
    {
        if (_toReleaseFlag)
        {
            _data->releaseBlockOfRows(_block);
            _toReleaseFlag = false;
        }
        _data = nullptr;
        _status.clear();
    }
    const services::Status & status() const { return _status; }

private:
    algorithmFPAccessType * getBlock(size_t iStartFrom, size_t nRows)
    {
        _status        = _data->getBlockOfRows(iStartFrom, nRows, mode, _block);
        _toReleaseFlag = _status.ok();
        return _block.getBlockPtr();
    }

private:
    NumericTableType * _data;
    BlockDescriptor<algorithmFPType> _block;
    services::Status _status;
    bool _toReleaseFlag;
};

template <typename algorithmFPType, CpuType cpu, typename NumericTableType = NumericTable>
using ReadRows = GetRows<algorithmFPType, const algorithmFPType, cpu, readOnly, NumericTableType>;

template <typename algorithmFPType, CpuType cpu, typename NumericTableType = NumericTable>
using WriteRows = GetRows<algorithmFPType, algorithmFPType, cpu, readWrite, NumericTableType>;

template <typename algorithmFPType, CpuType cpu, typename NumericTableType = NumericTable>
using WriteOnlyRows = GetRows<algorithmFPType, algorithmFPType, cpu, writeOnly, NumericTableType>;

using daal::services::internal::TArray;
using daal::services::internal::TArrayCalloc;
using daal::services::internal::TArrayScalable;
using daal::services::internal::TArrayScalableCalloc;

using daal::services::internal::TNArray;

template <typename algorithmFPType, typename algorithmFPAccessType, CpuType cpu, ReadWriteMode mode>
class GetRowsCSR
{
public:
    GetRowsCSR(CSRNumericTableIface & data, size_t iStartFrom, size_t nRows, bool toOneBaseRowIndices = false)
        : _data(&data), _toReleaseFlag(false), _toOneBaseRowIndices(toOneBaseRowIndices)
    {
        getBlock(iStartFrom, nRows);
    }
    GetRowsCSR(CSRNumericTableIface * data, size_t iStartFrom, size_t nRows, bool toOneBaseRowIndices = false)
        : _data(data), _toReleaseFlag(false), _toOneBaseRowIndices(toOneBaseRowIndices)
    {
        if (_data)
        {
            getBlock(iStartFrom, nRows);
        }
    }
    GetRowsCSR(CSRNumericTableIface * data = nullptr) : _data(data), _toReleaseFlag(false), _toOneBaseRowIndices(false) {}
    ~GetRowsCSR() { release(); }

    const algorithmFPAccessType * values() const { return _data ? _block.getBlockValuesPtr() : nullptr; }
    const size_t * cols() const { return _data ? _block.getBlockColumnIndicesPtr() : nullptr; }
    const size_t * rows() const { return _data ? _toOneBaseRowIndices ? _rowOffsets.get() : _block.getBlockRowIndicesPtr() : nullptr; }
    algorithmFPAccessType * values() { return _data ? _block.getBlockValuesPtr() : nullptr; }
    size_t * cols() { return _data ? _block.getBlockColumnIndicesPtr() : nullptr; }
    size_t * rows() { return _data ? _toOneBaseRowIndices ? _rowOffsets.get() : _block.getBlockRowIndicesPtr() : nullptr; }

    void next(size_t iStartFrom, size_t nRows, const bool toOneBaseRowIndices = false)
    {
        _toOneBaseRowIndices = toOneBaseRowIndices;
        if (_data)
        {
            if (_toReleaseFlag)
            {
                _status = _data->releaseSparseBlock(_block);
            }
            getBlock(iStartFrom, nRows);
        }
    }
    void set(CSRNumericTableIface * data, size_t iStartFrom, size_t nRows, const bool toOneBaseRowIndices = false)
    {
        _toOneBaseRowIndices = toOneBaseRowIndices;
        release();
        if (data)
        {
            _data = data;
            getBlock(iStartFrom, nRows);
        }
    }

    void release()
    {
        if (_toReleaseFlag)
        {
            _data->releaseSparseBlock(_block);
            _toReleaseFlag = false;
        }
        _data = nullptr;
        _status.clear();
    }

    const services::Status & status() const { return _status; }
    size_t size() { return _block.getDataSize(); }

private:
    void getBlock(size_t iStartFrom, size_t nRows)
    {
        _status        = _data->getSparseBlock(iStartFrom, nRows, mode, _block);
        _toReleaseFlag = _status.ok();

        if (_toOneBaseRowIndices)
        {
            if (_rowOffsets.size() < nRows + 1)
            {
                _rowOffsets.reset(nRows + 1);
            }
            const size_t * const rows = _block.getBlockRowIndicesPtr();
            _rowOffsets[0]            = 1;
            for (size_t i = 0; i < nRows; ++i)
            {
                const size_t nNonZeroValuesInRow = rows[i + 1] - rows[i];
                _rowOffsets[i + 1]               = _rowOffsets[i] + nNonZeroValuesInRow;
            }
        }
    }

private:
    CSRNumericTableIface * _data;
    CSRBlockDescriptor<algorithmFPType> _block;
    TArray<size_t, cpu> _rowOffsets;
    services::Status _status;
    bool _toOneBaseRowIndices;
    bool _toReleaseFlag;
};

template <typename algorithmFPType, CpuType cpu>
using ReadRowsCSR = GetRowsCSR<algorithmFPType, const algorithmFPType, cpu, readOnly>;

template <typename algorithmFPType, CpuType cpu>
using WriteRowsCSR = GetRowsCSR<algorithmFPType, algorithmFPType, cpu, readWrite>;

template <typename algorithmFPType, CpuType cpu>
using WriteOnlyRowsCSR = GetRowsCSR<algorithmFPType, algorithmFPType, cpu, writeOnly>;

template <typename algorithmFPType, typename algorithmFPAccessType, CpuType cpu, ReadWriteMode mode, typename NumericTableType>
class GetColumns
{
public:
    GetColumns(NumericTableType & data, size_t iCol, size_t iStartFrom, size_t n) : _data(&data)
    {
        _status = _data->getBlockOfColumnValues(iCol, iStartFrom, n, mode, _block);
    }
    GetColumns(NumericTableType * data, size_t iCol, size_t iStartFrom, size_t n) : _data(data)
    {
        if (_data) _status = _data->getBlockOfColumnValues(iCol, iStartFrom, n, mode, _block);
    }
    GetColumns() : _data(nullptr) {}
    ~GetColumns() { release(); }
    algorithmFPAccessType * get() { return _data ? _block.getBlockPtr() : nullptr; }
    algorithmFPAccessType * next(size_t iCol, size_t iStartFrom, size_t n)
    {
        if (!_data) return nullptr;
        _status = _data->releaseBlockOfColumnValues(_block);
        _status |= _data->getBlockOfColumnValues(iCol, iStartFrom, n, mode, _block);
        return _block.getBlockPtr();
    }
    algorithmFPAccessType * set(NumericTableType * data, size_t iCol, size_t iStartFrom, size_t n)
    {
        release();
        if (data)
        {
            _data   = data;
            _status = _data->getBlockOfColumnValues(iCol, iStartFrom, n, mode, _block);
            return _block.getBlockPtr();
        }
        return nullptr;
    }
    void release()
    {
        if (_data)
        {
            _data->releaseBlockOfColumnValues(_block);
            _data = nullptr;
            _status.clear();
        }
    }
    const services::Status & status() const { return _status; }

private:
    NumericTableType * _data;
    BlockDescriptor<algorithmFPType> _block;
    services::Status _status;
};

template <typename algorithmFPType, CpuType cpu, typename NumericTableType = NumericTable>
using ReadColumns = GetColumns<algorithmFPType, const algorithmFPType, cpu, readOnly, NumericTableType>;

template <typename algorithmFPType, CpuType cpu, typename NumericTableType = NumericTable>
using WriteColumns = GetColumns<algorithmFPType, algorithmFPType, cpu, readWrite, NumericTableType>;

template <typename algorithmFPType, CpuType cpu, typename NumericTableType = NumericTable>
using WriteOnlyColumns = GetColumns<algorithmFPType, algorithmFPType, cpu, writeOnly, NumericTableType>;

template <typename algorithmFPType, typename algorithmFPAccessType, CpuType cpu, ReadWriteMode mode, typename NumericTableType>
class GetPacked
{
public:
    GetPacked(NumericTableType & data)
    {
        _data = dynamic_cast<PackedArrayNumericTableIface *>(&data);
        if (!_data)
        {
            _status = services::Status(services::ErrorIncorrectTypeOfNumericTable);
            return;
        }
        _status = _data->getPackedArray(mode, _block);
    }
    GetPacked(NumericTableType * data)
    {
        _data = dynamic_cast<PackedArrayNumericTableIface *>(data);
        if (!_data)
        {
            _status = services::Status(services::ErrorIncorrectTypeOfNumericTable);
            return;
        }
        _status = _data->getPackedArray(mode, _block);
    }
    GetPacked() : _data(nullptr) {}
    ~GetPacked() { release(); }
    algorithmFPAccessType * get() { return _data ? _block.getBlockPtr() : nullptr; }
    algorithmFPAccessType * set(NumericTableType * data)
    {
        release();
        PackedArrayNumericTableIface * ptr = dynamic_cast<PackedArrayNumericTableIface *>(data);
        if (!ptr)
        {
            _status = services::Status(services::ErrorIncorrectTypeOfNumericTable);
            return nullptr;
        }
        if (ptr)
        {
            _data   = ptr;
            _status = _data->getPackedArray(mode, _block);
            return _block.getBlockPtr();
        }
        return nullptr;
    }
    void release()
    {
        if (_data)
        {
            _data->releasePackedArray(_block);
            _data = nullptr;
            _status.clear();
        }
    }
    const services::Status & status() const { return _status; }

private:
    PackedArrayNumericTableIface * _data;
    BlockDescriptor<algorithmFPType> _block;
    services::Status _status;
};

template <typename algorithmFPType, CpuType cpu, typename NumericTableType = NumericTable>
using ReadPacked = GetPacked<algorithmFPType, const algorithmFPType, cpu, readOnly, NumericTableType>;

template <typename algorithmFPType, CpuType cpu, typename NumericTableType = NumericTable>
using WritePacked = GetPacked<algorithmFPType, algorithmFPType, cpu, readWrite, NumericTableType>;

template <typename algorithmFPType, CpuType cpu, typename NumericTableType = NumericTable>
using WriteOnlyPacked = GetPacked<algorithmFPType, algorithmFPType, cpu, writeOnly, NumericTableType>;

template <typename algorithmFPType>
services::Status createSparseTable(const NumericTablePtr & inputTable, CSRNumericTablePtr & resTable);

} // namespace internal
} // namespace daal

#endif
