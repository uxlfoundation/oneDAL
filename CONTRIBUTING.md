<!--
******************************************************************************
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
*******************************************************************************/-->

# How to Contribute
We welcome community contributions to Intel(R) oneAPI Data Analytics Library. You can:

- Submit your changes directly with a [pull request](https://github.com/uxlfoundation/oneDAL/pulls).
- Log a bug or make a feature request with an [issue](https://github.com/uxlfoundation/oneDAL/issues).

Refer to our guidelines on [pull requests](#pull-requests) and [issues](#issues) before you proceed.

## Contacting maintainers
You may reach out to Intel project maintainers privately at onedal.maintainers@intel.com.
[Codeowners](https://github.com/uxlfoundation/oneDAL/blob/main/.github/CODEOWNERS) configuration defines specific maintainers for corresponding code sections, however it's currently limited to Intel members. With further migration to UXL we will be changing this, but here are non-Intel contacts:

For ARM specifics you may contact: [@rakshithgb-fujitsu](https://github.com/rakshithgb-fujitsu/)

For RISC-V specifics you may contact: [@keeranroth](https://github.com/keeranroth/)

## Issues

Use [GitHub issues](https://github.com/uxlfoundation/oneDAL/issues) to:
- report an issue
- make a feature request

**Note**: To report a vulnerability, refer to [Intel vulnerability reporting policy](https://www.intel.com/content/www/us/en/security-center/default.html).

## Pull Requests

To contribute your changes directly to the repository, do the following:
- Make sure you can build the product and run all the examples with your patch.
- Product uses bazel for validation and your changes should pass tests. Please add new tests as necessary. [Bazel Guide for oneDAL](https://github.com/uxlfoundation/oneDAL/tree/main/dev/bazel)
- Make sure your code is in line with our [coding style](#code-style) as `clang-format` is one of the checks in our public CI.
- For a larger feature, provide a relevant example, and tests.
- [Document](#documentation-guidelines) your code.
- [Submit](https://github.com/uxlfoundation/oneDAL/pulls) a pull request into the `main` branch.

Public and private CIs are enabled for the repository. Your PR should pass all of our checks. We will review your contribution and, if any additional fixes or modifications are necessary, we may give some feedback to guide you. When accepted, your pull request will be merged into our GitHub* repository.

## Code Style

### ClangFormat

**Prerequisites:** ClangFormat `9.0.0` or later

Our repository contains [clang-format configurations](https://github.com/uxlfoundation/oneDAL/blob/main/.clang-format) that you should use on your code. To do this, run:

```
clang-format style=file <your file>
```

Refer to [ClangFormat documentation](https://clang.llvm.org/docs/ClangFormat.html) for more information.

### editorconfig-checker

We also recommend using [editorconfig-checker](https://github.com/editorconfig-checker/editorconfig-checker) to ensure your code adheres to the project's coding style. EditorConfig-Checker is a command-line tool that checks your code against the rules defined in the [.editorconfig](https://github.com/uxlfoundation/oneDAL/blob/main/.editorconfig) file.

To use it, follow these steps:

1. Install the tool by following the instructions in the [official documentation](https://github.com/editorconfig-checker/editorconfig-checker#installation).
2. Navigate to the root directory of your project.
3. Run the following command to check your code:

```
editorconfig-checker
```

### Coding Guidelines

For your convenience we also added [coding guidelines](https://uxlfoundation.github.io/oneDAL/contribution/coding_guide.html) with examples and detailed descriptions of the coding style oneDAL follows. We encourage you to consult them when writing your code.

## Custom Components

### CPU Features Dispatching

oneDAL provides binaries that can contain code targeting different architectural extensions of a base instruction set architecture (ISA). For example, code paths can exist for Intel(R) SSE2, Intel(R) AVX2, Intel(R) AVX-512, etc. extensions, on top of the x86-64 base architecture.
When run on a specific hardware implementation like Haswell, Skylake-X, etc., oneDAL chooses the code path which is most suitable for that implementation.
Contributors should leverage [CPU Features Dispatching](http://uxlfoundation.github.io/oneDAL/contribution/cpu_features.html) mechanism to implement the code of the algorithms that can perform most optimally on various hardware implementations.

### Threading Layer

In the source code of the algorithms, oneDAL does not use threading primitives directly. All the threading primitives used within oneDAL form are called the [threading layer](http://uxlfoundation.github.io/oneDAL/contribution/threading.html). Contributors should leverage the primitives from the layer to implement parallel algorithms.

## Documentation Guidelines

oneDAL uses `Doxygen` for inline comments in public header files that are used to build the API reference and  `reStructuredText` for the Developer Guide. See [oneDAL documentation](https://uxlfoundation.github.io/oneDAL/) for reference.

---
**Note:** oneDAL is licensed under [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

---
