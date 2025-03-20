/* file: library_version_info.cpp */
/*******************************************************************************
* Copyright 2015 Intel Corporation
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
!  Content:
!    oneAPI Data Analytics Library (oneDAL) version information
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-LIBRARY_VERSION_INFO"></a>
 * \example library_version_info.cpp
 */

#include "daal.h"
#include <iostream>

using namespace daal::services;

int main() {
    LibraryVersionInfo ver;

    std::cout << "Major version:          " << ver.majorVersion << std::endl;
    std::cout << "Minor version:          " << ver.minorVersion << std::endl;
    std::cout << "Update version:         " << ver.updateVersion << std::endl;
    std::cout << "Product status:         " << ver.productStatus << std::endl;
    std::cout << "Build:                  " << ver.build << std::endl;
    std::cout << "Build revision:         " << ver.build_rev << std::endl;
    std::cout << "Name:                   " << ver.name << std::endl;
    std::cout << "Processor optimization: " << ver.processor << std::endl;

    return 0;
}
