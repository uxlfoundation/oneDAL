/** file csr_numeric_table.cpp */
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

#include "data_management/data/csr_numeric_table.h"

namespace daal
{
namespace data_management
{
namespace interface1
{
#define DAAL_IMPL_CSRBLOCKDESCRIPTORCONSTRUCTOR(T)                                                                                    \
    template <>                                                                                                                       \
    CSRBlockDescriptor<T>::CSRBlockDescriptor()                                                                                       \
        : _rows_capacity(0), _values_capacity(0), _ncols(0), _nrows(0), _rowsOffset(0), _rwFlag(0), _rawPtr(0), _pPtr(0), _nvalues(0) \
    {}

DAAL_IMPL_CSRBLOCKDESCRIPTORCONSTRUCTOR(float)
DAAL_IMPL_CSRBLOCKDESCRIPTORCONSTRUCTOR(double)
DAAL_IMPL_CSRBLOCKDESCRIPTORCONSTRUCTOR(int)
DAAL_IMPL_CSRBLOCKDESCRIPTORCONSTRUCTOR(unsigned int)
DAAL_IMPL_CSRBLOCKDESCRIPTORCONSTRUCTOR(DAAL_INT64)
DAAL_IMPL_CSRBLOCKDESCRIPTORCONSTRUCTOR(DAAL_UINT64)
DAAL_IMPL_CSRBLOCKDESCRIPTORCONSTRUCTOR(char)
DAAL_IMPL_CSRBLOCKDESCRIPTORCONSTRUCTOR(unsigned char)
DAAL_IMPL_CSRBLOCKDESCRIPTORCONSTRUCTOR(short)
DAAL_IMPL_CSRBLOCKDESCRIPTORCONSTRUCTOR(unsigned short)
DAAL_IMPL_CSRBLOCKDESCRIPTORCONSTRUCTOR(unsigned long)

} // namespace interface1
} // namespace data_management
} // namespace daal
