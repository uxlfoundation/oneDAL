
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
- **[MODULE.bazel](../../MODULE.bazel)** - Root module configuration
- **[.bazelrc](../../.bazelrc)** - Bazel configuration options
- **[dev/bazel/BUILD](BUILD)** - Root build configuration
- **[dev/bazel/cc/BUILD](cc/BUILD)** - C++ build configuration
- **[dev/bazel/deps/BUILD](deps/BUILD)** - Dependency management

### Module Configuration
`MODULE.bazel` declares a handful of `bazel_dep`s (`platforms`, `bazel_skylib`, `rules_cc`, `rules_shell`, `fmt`). Everything else is a repository rule: catch2 is an `http_archive` with `build_file = "//dev/bazel/deps:catch2.BUILD"`, and MKL, TBB, OpenBLAS, MPI, CCL, DPL and OpenCL come from the `dev/bazel/deps/*.bzl` rules. Read `MODULE.bazel` for current versions rather than copying them from here.

## 🔧 Build Rules and Patterns

oneDAL does not use bare `cc_library` / `cc_test`. All modules go through the macros in `@onedal//dev/bazel:dal.bzl` (`dal_module`, `dal_test_suite`) and `@onedal//dev/bazel:daal.bzl` (`daal_module`). Compiler flags, CPU dispatch and threading are toolchain-owned (`dev/bazel/flags.bzl`); never hand-write `copts`.

```python
load("@onedal//dev/bazel:dal.bzl", "dal_module", "dal_test_suite")

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "core",
    auto = True,                       # globs sources by convention, excludes test/
    dal_deps = ["@onedal//cpp/oneapi/dal:core"],
    extra_deps = ["@onedal//cpp/daal/src/algorithms/pca:kernel"],
)

dal_test_suite(
    name = "interface_tests",
    srcs = glob(["test/*.cpp"]),
    hdrs = glob(["test/*.hpp"]),
    dal_deps = [":pca"],
    framework = "catch2",              # injects the catch2 main; don't add a catch2 dep by hand
)

dal_test_suite(
    name = "tests",                    # aggregate target CI invokes
    tests = [":backend_tests", ":interface_tests"],
)
```

Reference implementation: `cpp/oneapi/dal/algo/pca/BUILD`.

## 🔧 Common Commands

```bash
# Build the oneAPI core
bazel build //cpp/oneapi/dal:core

# Test one algorithm, CPU only
bazel test --config=host //cpp/oneapi/dal/algo/pca:tests

# Build the release tree
bazel build //:release
```

Always scope targets and pass `--config`; with `--config` unset, tests include DPC++ targets that need the Intel DPC++ compiler. `dev/bazel/README.md` lists every config.

## 🔧 Dependency Management

External libraries are referenced by their repository labels, e.g. `@mkl//:mkl_core`, `@tbb//:tbb`, `@openblas//:openblas`, `@mpi//:mpi`. `dev/bazel/deps/BUILD` declares no targets; the `*.tpl.BUILD` files next to it are the BUILD templates for those repositories.

## 🎯 Development Guidelines

### Dependencies
- **Internal**: Full `@onedal//` labels in `dal_deps` (e.g., `@onedal//cpp/oneapi/dal:core`)
- **DAAL kernels**: `extra_deps` (e.g., `@onedal//cpp/daal/src/algorithms/pca:kernel`)
- **Visibility**: Set appropriate visibility levels

## 🚫 Common Pitfalls
- **Build Configuration**: Don't hardcode platform-specific paths
- **Dependencies**: Don't mix different dependency management approaches  
- **Toolchains**: Don't assume toolchain availability, test on target platforms

## 🧪 Testing and Validation
- **Build Validation**: Ensure builds work on all supported platforms
- **Dependencies**: Validate dependency resolution
- **CI/CD Integration**: Primary build system for CI/CD

## 🔧 Required Tools
- **Bazel**: version pinned in `.bazelversion`; `.ci/env/bazelisk.sh` installs a matching launcher

## 📖 Further Reading
- **[dev/AGENTS.md](../AGENTS.md)** - Development tools context
- **[cpp/AGENTS.md](../../cpp/AGENTS.md)** - C++ implementation context
- **[docs/AGENTS.md](../../docs/AGENTS.md)** - Documentation guidelines
