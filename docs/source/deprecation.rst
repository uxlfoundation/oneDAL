.. Copyright 2023 Intel Corporation
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..     http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.


Deprecation Notice
==================

This page provides information about the deprecations of a specific oneAPI Data Analytics Library (oneDAL) functionality.

Java* Interfaces
****************

**Deprecation:** The Java* interfaces in the oneDAL library are marked as deprecated. The future releases of the oneDAL library may no longer include support for these Java* interfaces.

**Reasons for deprecation:** The ongoing efforts to optimize oneDAL resources and focus strongly on the most widely used features.

**Alternatives:** Intel(R) Optimized Analytics Package* (OAP) project for the Spark* users.
The project offers a comprehensive set of optimized libraries, including the OAP* MLlib* component. For more information, visit https://github.com/oap-project/oap-mllib.

Compression Functionality
*************************

**Deprecation:** The compression functionality in the oneDAL library is deprecated. Starting with the 2024.0 release, oneDAL will not support the compression functionality.

**Reasons for deprecation:** The ongoing efforts to optimize oneDAL resources and focus strongly on the most widely used features.

**Alternatives:** The external compression mechanics with optimized into your application implementation. For example, Intel(R) IPP.

ABI Compatibility
*****************

**Deprecation:** ABI compatibility is to be broken as part of the 2024.0 release of oneDAL. The library's major version is to be incremented to two to enforce the relinking of existing applications.

**Reasons for deprecation:**  The clean-up process of the deprecated functionality, interfaces, and symbols.

**Alternatives:** Relink to a newer version.

macOS* Support
**************

**Deprecation:** macOS* support is deprecated for oneDAL. The 2023.x releases are the last to provide it.

**Reasons for deprecation:**  No modern X86 macOS*-based systems are to be released.

**Alternatives:** The 2023.x version on macOS*.
