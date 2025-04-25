/** file error_id.h */
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
//  Data types for error handling in oneAPI Data Analytics Library (oneDAL).
//--
*/

#ifndef __ERROR_ID__
#define __ERROR_ID__

#include "services/error_indexes.h"
#include "services/daal_defines.h"
#include "services/daal_memory.h"

namespace daal
{
namespace services
{
namespace interface1
{
/**
 * @ingroup error_handling
 * @{
 */
/**
* <a name="DAAL-CLASS-SERVICES__ERRORDETAIL"></a>
* \brief Base for error detail classes
*/
class DAAL_EXPORT ErrorDetail
{
public:
    DAAL_NEW_DELETE();

    /**
    * Constructs error detail from error identifier
    * \param[in] id    Error identifier
    */
    ErrorDetail(ErrorDetailID id) : _id(id), _next(NULL) {}

    /**
    * Destructor
    */
    virtual ~ErrorDetail() {}

    /**
    * Returns identifier of an error detail
    * \return identifier of an error detail
    */
    ErrorDetailID id() const { return _id; }

    /**
    * Returns copy of this object
    * \return copy of this object
    */
    virtual ErrorDetail * clone() const = 0;

    /**
    * Adds description of the error detail to the given string
    * \param[in] str String to add descrition to
    */
    virtual void describe(char * str) const = 0;

    /**
    * Access to the pointer of the next detail
    * \return pointer to the next detail
    */
    const ErrorDetail * next() const { return _next; }

protected:
    /**
    * Access to the pointer of the next detail
    * \return pointer to the next detail
    */
    ErrorDetail * next() { return _next; }

    /**
    * Set pointer to the next detail
    * \param[in] ptr Pointer to the next detail
    */
    void addNext(ErrorDetail * ptr) { _next = ptr; }

private:
    const ErrorDetailID _id;
    ErrorDetail * _next;
    friend class Error;
};
/** @} */

} // namespace interface1
using interface1::ErrorDetail;

} // namespace services
} // namespace daal
#endif
