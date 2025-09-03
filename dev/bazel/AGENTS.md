
# Bazel Build System - AI Agents Context

> **Purpose**: Context for AI agents working with the Bazel build system in oneDAL development and testing.

## 🏗️ Bazel Build System Overview

Bazel is the **development and testing build system** for oneDAL, providing fast, reliable builds with automatic dependency management.

### Key Characteristics
- **Development Build System**: Used for development and CI/CD
- **Dependency Management**: Automatic dependency resolution
- **Multi-platform**: Linux, Windows, macOS support
- **Incremental Builds**: Fast incremental compilation
- **Hermetic Builds**: Reproducible build environments

## 📁 Structure
```
dev/bazel/
├── BUILD           # Root Bazel configuration
├── cc/             # C++ build rules and configurations
├── config/         # Build configurations and toolchains
├── deps/           # External dependencies
└── toolchains/     # Compiler toolchain configurations
```

## 🎯 Configuration Files
- **[MODULE.bazel](MODULE.bazel)** - Root module configuration
- **[.bazelrc](.bazelrc)** - Bazel configuration options
- **[dev/bazel/BUILD](BUILD)** - Root build configuration
- **[dev/bazel/cc/BUILD](cc/BUILD)** - C++ build configuration
- **[dev/bazel/deps/BUILD](deps/BUILD)** - Dependency management

### Module Configuration
```python
# MODULE.bazel
module(
    name = "onedal",
    version = "1.0.0",
)

bazel_dep(name = "rules_cc", version = "0.0.9")
bazel_dep(name = "catch2", version = "3.4.0")
```

## 🔧 Build Rules and Patterns

### C++ Library Target
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
    copts = ["-std=c++17", "-O3"],
)
```

### C++ Test Target
```python
cc_test(
    name = "library_test",
    srcs = glob(["test/**/*.cpp"]),
    deps = [
        ":library_name",
        "//dev/bazel/deps:catch2",
    ],
    copts = ["-std=c++17", "-g"],
)
```

## 🔧 Common Commands

```bash
# Build entire project
bazel build //...

# Build specific target
bazel build //cpp/daal:daal

# Run tests
bazel test //...

# Clean build
bazel clean --expunge
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

## 🎯 Development Guidelines

### Build Target Naming
- **Libraries**: Use descriptive names (e.g., `daal_core`, `oneapi_dal`)
- **Tests**: Append `_test` suffix (e.g., `daal_core_test`)
- **Examples**: Use descriptive names (e.g., `kmeans_example`)

### Dependencies
- **Internal**: Use relative paths (e.g., `//cpp/daal:daal`)
- **External**: Use dependency rules (e.g., `//dev/bazel/deps:tbb`)
- **Visibility**: Set appropriate visibility levels

## 🔍 Common Patterns

### Conditional Compilation
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

### Feature Detection
```python
cc_library(
    name = "feature_detection",
    srcs = ["src/feature_detection.cpp"],
    copts = select({
        "//dev/bazel/config:avx512": ["-mavx512f"],
        "//dev/bazel/config:avx2": ["-mavx2"],
        "//conditions:default": [],
    }),
)
```

## 🚫 Common Pitfalls
- **Build Configuration**: Don't hardcode platform-specific paths
- **Dependencies**: Don't mix different dependency management approaches  
- **Toolchains**: Don't assume toolchain availability, test on target platforms

## 🧪 Testing and Validation
- **Build Validation**: Ensure builds work on all supported platforms
- **Dependencies**: Validate dependency resolution
- **CI/CD Integration**: Primary build system for CI/CD

## 🔧 Required Tools
- **Bazel**: 5.0+ for modern Bazel features
- **Python**: 3.7+ for build rule development
- **Compilers**: GCC 7+, Clang 6+, MSVC 2017+

## 📖 Further Reading
- **[dev/AGENTS.md](../AGENTS.md)** - Development tools context
- **[cpp/AGENTS.md](../../cpp/AGENTS.md)** - C++ implementation context
- **[docs/AGENTS.md](../../docs/AGENTS.md)** - Documentation guidelines
