<!--
******************************************************************************
* Copyright contributors to the oneDAL project
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
*******************************************************************************/-->

## Coverity Patch Overview

This repository includes a patch file located at `.github/patches/coverity`, which is specifically designed to improve the effectiveness of Coverity static analysis scans.

### Purpose

The patch reflects necessary modifications to Coverity configuration files to increase the number of **Captured Compilation Units (CCUs)** to at least **85%**, which is the minimum threshold required by [scan.coverity.com](https://scan.coverity.com) to perform a successful analysis.

### Background

Coverity Build Tool currently lacks full support for the latest versions of **Intel DPCPP compiler**, which is commonly used in modern C++ projects, especially those involving SYCL or heterogeneous computing. As a result, many compilation units are not captured during the default build process, leading to failure to meet Coverity scanning requirements and server side analysis rejection.

### Solution

The patch introduces configuration changes and workarounds that help Coverity recognize and process a greater portion of the build, despite the limitations with DPCPP. These changes may include:

- Adjustments to compiler flags
- Substitutions or wrappers for unsupported compiler calls
- Environment variable tweaks
- Additional build script logic

By applying this patch, the project can meet the CCU threshold and ensure that Coverity performs a meaningful and complete static analysis.

### Support

The patch is expected (not required) to be reviewed after each Coverity Build Tool version upgrade (~once an year) for relevance of the changes it contains, some of which may become obsolete due to improved Intel compiler support in future versions of the Coverity Build Tool.
