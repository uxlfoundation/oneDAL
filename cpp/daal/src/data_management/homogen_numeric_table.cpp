/** file homogen_numeric_table.cpp */
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

#include "data_management/data/homogen_numeric_table.h"

using namespace daal;
using namespace daal::data_management;

template <typename DataType>
HomogenNumericTable<DataType>::HomogenNumericTable(DictionaryIface::FeaturesEqual featuresEqual, size_t nColumns, size_t nRows,
                                                   NumericTable::AllocationFlag memoryAllocationFlag, const DataType & constValue,
                                                   services::Status & st)
    : NumericTable(nColumns, nRows, featuresEqual, st)
{
    _layout = aos;

    NumericTableFeature df;
    df.setType<DataType>();

    st |= _ddict->setAllFeatures(df);

    if (memoryAllocationFlag == doAllocate)
    {
        st |= allocateDataMemoryImpl();
    }

    st |= assign<DataType>(constValue);
}

#define DAAL_INSTANTIATE_FUNCTION(T)                                                                                                  \
    template HomogenNumericTable<T>::HomogenNumericTable(DictionaryIface::FeaturesEqual featuresEqual, size_t nColumns, size_t nRows, \
                                                         NumericTable::AllocationFlag memoryAllocationFlag, const T & constValue,     \
                                                         services::Status & st)

DAAL_INSTANTIATE_FUNCTION(float);
DAAL_INSTANTIATE_FUNCTION(double);
DAAL_INSTANTIATE_FUNCTION(int);
DAAL_INSTANTIATE_FUNCTION(unsigned int);
DAAL_INSTANTIATE_FUNCTION(DAAL_INT64);
DAAL_INSTANTIATE_FUNCTION(DAAL_UINT64);
DAAL_INSTANTIATE_FUNCTION(char);
DAAL_INSTANTIATE_FUNCTION(unsigned char);
DAAL_INSTANTIATE_FUNCTION(short);
DAAL_INSTANTIATE_FUNCTION(unsigned short);
DAAL_INSTANTIATE_FUNCTION(unsigned long);
DAAL_INSTANTIATE_FUNCTION(long);
