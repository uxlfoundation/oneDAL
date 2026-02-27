/** file row_merged_numeric_table.cpp */
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

#include "data_management/data/row_merged_numeric_table.h"

namespace daal
{
namespace data_management
{
namespace interface1
{
RowMergedNumericTable::RowMergedNumericTable() : NumericTable(0, 0), _tables(new DataCollection) {}

RowMergedNumericTable::RowMergedNumericTable(NumericTablePtr table) : NumericTable(0, 0), _tables(new DataCollection)
{
    this->_status |= addNumericTable(table);
}

services::SharedPtr<RowMergedNumericTable> RowMergedNumericTable::create(services::Status * stat)
{
    DAAL_DEFAULT_CREATE_IMPL(RowMergedNumericTable);
}

services::SharedPtr<RowMergedNumericTable> RowMergedNumericTable::create(const NumericTablePtr & nestedTable, services::Status * stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(RowMergedNumericTable, nestedTable);
}

RowMergedNumericTable::RowMergedNumericTable(services::Status & st) : NumericTable(0, 0), _tables(new DataCollection)
{
    if (!_tables)
    {
        st.add(services::ErrorMemoryAllocationFailed);
    }
    this->_status |= st;
}

RowMergedNumericTable::RowMergedNumericTable(const NumericTablePtr & table, services::Status & st) : NumericTable(0, 0), _tables(new DataCollection)
{
    if (!_tables)
    {
        st.add(services::ErrorMemoryAllocationFailed);
    }
    st |= addNumericTable(table);
    this->_status |= st;
}

} // namespace interface1
} // namespace data_management
} // namespace daal
