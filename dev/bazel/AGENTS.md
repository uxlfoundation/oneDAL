# Copyright contributors to the oneDAL project
#
<!--
  ~ Licensed under the Apache License, Version 2.0 (the "License");
  ~ you may not use this file except in compliance with the License.
  ~ You may obtain a copy of the License at
  ~
  ~     http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS,
  ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ~ See the License for the specific language governing permissions and
  ~ limitations under the License.
-->

# Bazel Build System - AI Agents Context

> **Purpose**: This file provides context for AI agents working with the Bazel build system in oneDAL, explaining the build configuration, rules, and development patterns.

## 🏗️ Bazel Build System Overview

Bazel is the **primary build system** for oneDAL development, providing fast, reliable builds with automatic dependency management and multi-platform support.

### Key Characteristics
- **Primary Build System**: Used for development and CI/CD
- **Dependency Management**: Automatic dependency resolution
- **Multi-platform**: Linux, Windows, macOS support
- **Incremental Builds**: Fast incremental compilation
- **Hermetic Builds**: Reproducible build environments

## 📁 Directory Structure

```
dev/bazel/
├── BUILD                    # Root Bazel configuration
├── cc/                      # C++ build rules and configurations
│   ├── BUILD               # C++ build configuration
│   ├── common.bzl          # Common build rules
│   ├── compile.bzl         # Compilation rules
│   └── ...                 # Other build configurations
├── config/                  # Build configurations and toolchains
│   ├── BUILD               # Configuration build rules
│   ├── config.bzl          # Configuration management
│   └── ...                 # Configuration files
├── deps/                    # External dependencies
│   ├── BUILD               # Dependency build rules
│   ├── catch2.BUILD        # Catch2 testing framework
│   ├── ccl.bzl             # CCL dependency
│   └── ...                 # Other dependencies
└── toolchains/              # Compiler toolchain configurations
    ├── BUILD                # Toolchain build rules
    ├── action_names.bzl     # Action definitions
    ├── cc_toolchain_config_lnx.bzl # Linux toolchain config
    └── ...                  # Other toolchain configs
```

## 🔧 Core Configuration Files

### Root Configuration
- **[MODULE.bazel](MODULE.bazel)** - Root module configuration
- **[.bazelrc](.bazelrc)** - Bazel configuration options
- **[dev/bazel/BUILD](BUILD)** - Root build configuration

### Build Rules
- **[dev/bazel/cc/BUILD](cc/BUILD)** - C++ build configuration
- **[dev/bazel/cc/common.bzl](cc/common.bzl)** - Common build rules
- **[dev/bazel/cc/compile.bzl](cc/compile.bzl)** - Compilation rules

### Dependencies
- **[dev/bazel/deps/BUILD](deps/BUILD)** - Dependency management
- **[dev/bazel/deps/catch2.BUILD](deps/catch2.BUILD)** - Testing framework
- **[dev/bazel/deps/ccl.bzl](deps/ccl.bzl)** - CCL dependency

## 🎯 Build Configuration

### Module Configuration
```python
# MODULE.bazel
module(
    name = "onedal",
    version = "1.0.0",
    compatibility_level = 1,
)

bazel_dep(name = "rules_cc", version = "0.0.9")
bazel_dep(name = "catch2", version = "3.4.0")
```

### Root Build Configuration
```python
# dev/bazel/BUILD
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "onedal_core",
    srcs = glob([
        "//cpp/daal/src/**/*.cpp",
        "//cpp/oneapi/dal/src/**/*.cpp",
    ]),
    hdrs = glob([
        "//cpp/daal/include/**/*.h",
        "//cpp/oneapi/dal/include/**/*.hpp",
    ]),
    deps = [
        "//dev/bazel/deps:tbb",
        "//dev/bazel/deps:mkl",
    ],
)
```

## 🔧 Build Rules and Patterns

### 1. C++ Library Target
```python
cc_library(
    name = "library_name",
    srcs = glob(["src/**/*.cpp"]),
    hdrs = glob(["include/**/*.h"]),
    deps = [
        "//path/to:dependency",
        "//dev/bazel/deps:external_lib",
    ],
    visibility = ["//visibility:public"],
    copts = [
        "-std=c++17",
        "-O3",
        "-march=native",
    ],
)
```

### 2. C++ Test Target
```python
cc_test(
    name = "library_test",
    srcs = glob(["test/**/*.cpp"]),
    deps = [
        ":library_name",
        "//dev/bazel/deps:catch2",
    ],
    copts = [
        "-std=c++17",
        "-g",
    ],
)
```

### 3. C++ Binary Target
```python
cc_binary(
    name = "example_binary",
    srcs = glob(["examples/**/*.cpp"]),
    deps = [
        ":library_name",
        "//dev/bazel/deps:external_lib",
    ],
    copts = [
        "-std=c++17",
        "-O2",
    ],
)
```

## 🔧 Dependency Management

### External Dependencies
```python
# dev/bazel/deps/BUILD
cc_library(
    name = "tbb",
    srcs = glob(["tbb/src/**/*.cpp"]),
    hdrs = glob(["tbb/include/**/*.h"]),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mkl",
    srcs = glob(["mkl/lib/**/*.so"]),
    hdrs = glob(["mkl/include/**/*.h"]),
    visibility = ["//visibility:public"],
)
```

### Dependency Rules
```python
# dev/bazel/deps/ccl.bzl
def ccl_deps():
    native.new_local_repository(
        name = "ccl",
        path = "path/to/ccl",
        build_file = "//dev/bazel/deps:ccl.BUILD",
    )
```

## 🔧 Toolchain Configuration

### Linux Toolchain
```python
# dev/bazel/toolchains/cc_toolchain_config_lnx.bzl
def _impl(ctx):
    toolchain = cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        toolchain_identifier = "linux-toolchain",
        host_system_name = "linux",
        target_system_name = "linux",
        target_cpu = "k8",
        target_libc = "glibc_2.19",
        compiler = "gcc",
        abi_version = "gcc",
        abi_libc_version = "glibc_2.19",
    )
    return toolchain

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {},
    provides = [CcToolchainConfigInfo],
)
```

## 🎯 Development Guidelines

### 1. Build Target Naming
- **Libraries**: Use descriptive names (e.g., `daal_core`, `oneapi_dal`)
- **Tests**: Append `_test` suffix (e.g., `daal_core_test`)
- **Examples**: Use descriptive names (e.g., `kmeans_example`)

### 2. Dependency Management
- **Internal Dependencies**: Use relative paths (e.g., `//cpp/daal:daal`)
- **External Dependencies**: Use dependency rules (e.g., `//dev/bazel/deps:tbb`)
- **Visibility**: Set appropriate visibility levels

### 3. Build Configuration
- **Compiler Flags**: Use `copts` for compiler options
- **Linker Flags**: Use `linkopts` for linker options
- **Platform Specific**: Use `select()` for platform-specific options

## 🔍 Common Build Patterns

### 1. Conditional Compilation
```python
cc_library(
    name = "platform_specific",
    srcs = select({
        "//dev/bazel/config:linux": ["src/linux.cpp"],
        "//dev/bazel/config:windows": ["src/windows.cpp"],
        "//conditions:default": ["src/default.cpp"],
    }),
    deps = [":common"],
)
```

### 2. Feature Detection
```python
cc_library(
    name = "feature_detection",
    srcs = ["src/feature_detection.cpp"],
    deps = [":common"],
    copts = select({
        "//dev/bazel/config:avx512": ["-mavx512f", "-mavx512dq"],
        "//dev/bazel/config:avx2": ["-mavx2"],
        "//conditions:default": [],
    }),
)
```

### 3. Testing Configuration
```python
cc_test(
    name = "comprehensive_test",
    srcs = glob(["test/**/*.cpp"]),
    deps = [
        ":library_name",
        "//dev/bazel/deps:catch2",
        "//dev/bazel/deps:gtest",
    ],
    size = "large",
    timeout = "long",
)
```

## 🚫 Common Pitfalls to Avoid

### 1. Build Configuration
- **Don't hardcode** platform-specific paths
- **Don't ignore** dependency conflicts
- **Do use** proper visibility settings

### 2. Dependency Management
- **Don't mix** different dependency management approaches
- **Don't ignore** version compatibility
- **Do use** proper dependency rules

### 3. Toolchain Configuration
- **Don't assume** toolchain availability
- **Don't ignore** platform differences
- **Do test** on target platforms

## 🧪 Testing and Validation

### Build Validation
- **All Platforms**: Ensure builds work on all supported platforms
- **Dependencies**: Validate dependency resolution
- **Toolchains**: Test with different compiler versions

### CI/CD Integration
- **Bazel Builds**: Primary CI/CD build system
- **Test Execution**: Run all tests in CI/CD
- **Performance**: Monitor build performance

## 🔧 Development Tools

### Required Tools
- **Bazel**: 5.0+ for modern Bazel features
- **Python**: 3.7+ for build rule development
- **Compilers**: GCC 7+, Clang 6+, MSVC 2017+

### Optional Tools
- **Bazel Buildifier**: Format BUILD files
- **Bazel Query**: Analyze build dependencies
- **Bazel Aquery**: Analyze build actions

## 📖 Further Reading

- **[dev/AGENTS.md](../AGENTS.md)** - Overall development tools context
- **[dev/make/AGENTS.md](../make/AGENTS.md)** - Make build system details
- **[cpp/AGENTS.md](../../cpp/AGENTS.md)** - C++ implementation context
- **[docs/AGENTS.md](../../docs/AGENTS.md)** - Documentation guidelines

---

**Note**: Bazel is the primary build system for oneDAL development. Use it for consistent, fast builds with automatic dependency management.
