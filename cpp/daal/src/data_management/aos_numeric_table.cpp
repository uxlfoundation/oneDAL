/** file aos_numeric_table.cpp */
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

#include "data_management/data/aos_numeric_table.h"

namespace daal
{
namespace data_management
{
namespace interface1
{
AOSNumericTable::AOSNumericTable(size_t structSize, size_t ncol, size_t nrow) : NumericTable(ncol, nrow)
{
    _layout     = aos;
    _structSize = structSize;

    initOffsets();
}

AOSNumericTable::AOSNumericTable(size_t structSize, size_t ncol, size_t nrow, services::Status & st)
    : NumericTable(ncol, nrow, DictionaryIface::notEqual, st) //?
{
    _layout     = aos;
    _structSize = structSize;

    st |= initOffsets();
}

services::SharedPtr<AOSNumericTable> AOSNumericTable::create(size_t structSize, size_t ncol, size_t nrow, services::Status * stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(AOSNumericTable, structSize, ncol, nrow);
}

} // namespace interface1
} // namespace data_management
} // namespace daal
