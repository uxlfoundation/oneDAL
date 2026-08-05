
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
├── rules/          # Bazel BUILD-time rules (mpi_test, ccl_test)
└── toolchains/     # Compiler toolchain configurations

third_party/
├── repo.bzl        # Shared repo-rule factory for prebuilt-libs deps
├── <dep>/
│   ├── workspace.bzl        # External repo declaration
│   ├── <dep>.tpl.BUILD      # BUILD template for the external repo
│   └── <dep>.win.tpl.BUILD  # Windows variant (optional)
└── catch2/catch2.BUILD      # Overlay for http_archive-fetched deps
```

## 🎯 Configuration Files
- **[MODULE.bazel](MODULE.bazel)** - Root module configuration; declares external deps via `//third_party/<dep>:workspace.bzl`
- **[.bazelrc](.bazelrc)** - Bazel configuration options
- **[dev/bazel/BUILD](BUILD)** - Root build configuration
- **[dev/bazel/cc/BUILD](cc/BUILD)** - C++ build configuration
- **[third_party/](../../third_party)** - One folder per external C++ dep (MKL, TBB, MPI, oneCCL, oneDPL, OpenCL, OpenMP, OpenBLAS, onedal_release)

### Module Configuration
```python
# MODULE.bazel
module(
    name = "onedal",
    version = "1.0.0",
)

bazel_dep(name = "rules_cc", version = "0.2.18")
bazel_dep(name = "catch2", version = "3.9.1")
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
        "@mkl//:mkl_core",         # external dep from //third_party/mkl
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
        "@catch2//:catch2",
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
Each external C++ dep lives in `//third_party/<dep>/` with three files:
- `workspace.bzl` - declares the repo rule (usually via `repos.prebuilt_libs_repo_rule`)
- `<dep>.tpl.BUILD` - BUILD template evaluated inside the external repo; defines `cc_library` targets like `@<dep>//:headers` and `@<dep>//:<dep>_core`
- `<dep>.win.tpl.BUILD` - Windows variant (optional; only some deps ship different layouts)

Consumers reference deps by external label (e.g. `@mkl//:mkl_core`), not by any local `//dev/bazel/...` alias.

## 🎯 Development Guidelines

### Build Target Naming
- **Libraries**: Use descriptive names (e.g., `daal_core`, `oneapi_dal`)
- **Tests**: Append `_test` suffix (e.g., `daal_core_test`)
- **Examples**: Use descriptive names (e.g., `kmeans_example`)

### Dependencies
- **Internal**: Use relative paths (e.g., `//cpp/daal:daal`)
- **External**: Use external labels (e.g., `@tbb//:tbb`); the repo is declared in `//third_party/<dep>/workspace.bzl` and wired in `MODULE.bazel`
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
