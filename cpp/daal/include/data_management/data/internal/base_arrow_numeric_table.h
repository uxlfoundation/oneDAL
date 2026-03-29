/* file: base_arrow_numeric_table.h */
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

#ifndef __BASE_ARROW_NUMERIC_TABLE_H__
#define __BASE_ARROW_NUMERIC_TABLE_H__

#include "data_management/data/numeric_table.h"
#include "data_management/data/internal/conversion.h"

namespace daal
{
namespace data_management
{
namespace interface1
{
/**
 * @ingroup numeric_tables
 * @{
 */
/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__ARROWNUMERICTABLE"></a>
 *  \brief Base class that provides methods to access data stored as a Apache Arrow table.
 */
class DAAL_EXPORT ArrowNumericTable : public NumericTable
{
public:
    /**
     *  Returns whether the numeric table is mutable
     *  \return true if the numeric table is mutable, false otherwise
     */
    virtual bool isMutable() const = 0;

protected:
    ArrowNumericTable(size_t featnum, size_t obsnum, services::Status & st) : NumericTable(featnum, obsnum, DictionaryIface::notEqual, st) {}
};
typedef services::SharedPtr<ArrowNumericTable> ArrowNumericTablePtr;

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__BASEARROWIMMUTABLENUMERICTABLE"></a>
 *  \brief Base class that provides methods to access data stored as a immutable Apache Arrow table.
 */
class DAAL_EXPORT BaseArrowImmutableNumericTable : public ArrowNumericTable
{
    DECLARE_SERIALIZABLE_TAG()

public:
    bool isMutable() const override { return false; }

protected:
    BaseArrowImmutableNumericTable(size_t featnum, size_t obsnum, services::Status & st) : ArrowNumericTable(featnum, obsnum, st) {}
};
typedef services::SharedPtr<BaseArrowImmutableNumericTable> BaseArrowImmutableNumericTablePtr;

/** @} */
} // namespace interface1

using interface1::ArrowNumericTable;
using interface1::ArrowNumericTablePtr;
using interface1::BaseArrowImmutableNumericTable;
using interface1::BaseArrowImmutableNumericTablePtr;

} // namespace data_management
} // namespace daal

#endif
