/** file merged_numeric_table.cpp */
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

#include "data_management/data/merged_numeric_table.h"

namespace daal
{
namespace data_management
{
namespace interface1
{
MergedNumericTable::MergedNumericTable() : NumericTable(0, 0), _tables(new DataCollection) {}

MergedNumericTable::MergedNumericTable(NumericTablePtr table) : NumericTable(0, 0), _tables(new DataCollection)
{
    this->_status |= addNumericTable(table);
}

MergedNumericTable::MergedNumericTable(NumericTablePtr first, NumericTablePtr second) : NumericTable(0, 0), _tables(new DataCollection)
{
    this->_status |= addNumericTable(first);
    this->_status |= addNumericTable(second);
}

MergedNumericTable::MergedNumericTable(services::Status & st) : NumericTable(0, 0), _tables(new DataCollection)
{
    if (!_tables)
    {
        st.add(services::ErrorMemoryAllocationFailed);
    }
    this->_status |= st;
}

MergedNumericTable::MergedNumericTable(const NumericTablePtr & table, services::Status & st) : NumericTable(0, 0), _tables(new DataCollection)
{
    if (!_tables)
    {
        st.add(services::ErrorMemoryAllocationFailed);
    }
    st |= addNumericTable(table);
    this->_status |= st;
}

MergedNumericTable::MergedNumericTable(const NumericTablePtr & first, const NumericTablePtr & second, services::Status & st)
    : NumericTable(0, 0), _tables(new DataCollection)
{
    if (!_tables)
    {
        st.add(services::ErrorMemoryAllocationFailed);
    }
    st |= addNumericTable(first);
    st |= addNumericTable(second);
    this->_status |= st;
}

services::SharedPtr<MergedNumericTable> MergedNumericTable::create(services::Status * stat)
{
    DAAL_DEFAULT_CREATE_IMPL(MergedNumericTable);
}

services::SharedPtr<MergedNumericTable> MergedNumericTable::create(const NumericTablePtr & nestedTable, services::Status * stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(MergedNumericTable, nestedTable);
}

services::SharedPtr<MergedNumericTable> MergedNumericTable::create(const NumericTablePtr & first, const NumericTablePtr & second,
                                                                   services::Status * stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(MergedNumericTable, first, second);
}

} // namespace interface1
} // namespace data_management
} // namespace daal
