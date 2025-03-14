/* file: library_version_info.h */
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
//  oneAPI Data Analytics Library (oneDAL) version information.
//--
*/

#ifndef __LIBRARY_VERSION_INFO_H__
#define __LIBRARY_VERSION_INFO_H__

#define __INTEL_DAAL_BUILD_DATE 21990101

#define __INTEL_DAAL__        2199
#define __INTEL_DAAL_MINOR__  9
#define __INTEL_DAAL_UPDATE__ 9
#define __INTEL_DAAL_STATUS__ 'A'

#define INTEL_DAAL_VERSION (__INTEL_DAAL__ * 10000 + __INTEL_DAAL_MINOR__ * 100 + __INTEL_DAAL_UPDATE__)

#define __INTEL_DAAL_MAJOR_BINARY__ 999
#define __INTEL_DAAL_MINOR_BINARY__ 999

#define INTEL_DAAL_BINARY_VERSION (__INTEL_DAAL_MAJOR_BINARY__ * 1000 + __INTEL_DAAL_MINOR_BINARY__)

#include "services/base.h"

namespace daal
{
/**
 * @defgroup services Services
 * \copydoc daal::services
 * @{
 */
namespace services
{
namespace interface1
{
/**
 * @defgroup library_version_info Extracting Version Information
 * \brief Provides information about the version of oneAPI Data Analytics Library.
 * @ingroup services
 * @{
 */
/**
 * <a name="DAAL-CLASS-SERVICES__LIBRARYVERSIONINFO"></a>
 * \brief Provides information about the version of oneAPI Data Analytics Library.
 * <!-- \n<a href="DAAL-REF-LIBRARYVERSIONINFO-STRUCTURE">LibraryVersionInfo structure details and Optimization Notice</a> -->
 */
class DAAL_EXPORT LibraryVersionInfo : public Base
{
public:
    const int majorVersion;     /*!< Major library version */
    const int minorVersion;     /*!< Minor library version */
    const int updateVersion;    /*!< Update library version */
    const char * productStatus; /*!< Product library status */
    const char * build;         /*!< Library build */
    const char * build_rev;     /*!< Library build revision */
    const char * name;          /*!< Library name */
    const char * processor;     /*!< Instruction set supported by the processor */

    LibraryVersionInfo();
    ~LibraryVersionInfo();
};
/** @} */
} // namespace interface1
using interface1::LibraryVersionInfo;

} // namespace services
/** @} */
} // namespace daal
#endif
