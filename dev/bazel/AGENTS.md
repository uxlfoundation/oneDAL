
# Bazel Build System - AI Agents Context

> **Purpose**: Context for AI agents working with the Bazel build system in oneDAL development and testing.

## 🏗️ Bazel Build System Overview

Bazel is the **development and testing build system** for oneDAL, providing fast, reliable builds with automatic dependency management.

### Key Characteristics
- **Development Build System**: Used for development and CI/CD
- **Dependency Management**: Automatic dependency resolution
- **Multi-platform**: Linux and Windows toolchains
- **Incremental Builds**: Fast incremental compilation
- **Hermetic Builds**: Reproducible build environments

## 📁 Structure
```
dev/bazel/
├── BUILD           # Empty; makes dev/bazel a package so its .bzl files can be loaded
├── cc/             # C++ build rules (compile, link, common)
├── config/         # Build settings, platforms, TargetInfo, ISA table
├── deps/           # External dependency repositories
├── tests/          # analysis.bzl (graph checks) and the two sh_test scripts that inspect built libraries
└── toolchains/     # Compiler toolchain configurations
```

## 🎯 Configuration Files
- **[MODULE.bazel](../../MODULE.bazel)** - Root module configuration
- **[.bazelrc](../../.bazelrc)** - Bazel configuration options
- **[dev/bazel/cc/BUILD](cc/BUILD)** - C++ build configuration
- **[dev/bazel/deps/BUILD](deps/BUILD)** - Dependency management

### Module Configuration
`MODULE.bazel` declares a handful of `bazel_dep`s (`platforms`, `bazel_skylib`, `rules_cc`, `rules_shell`, `fmt`). Everything else is a repository rule: catch2 is an `http_archive` with `build_file = "//dev/bazel/deps:catch2.BUILD"`, and MKL, TBB, OpenBLAS, MPI, CCL, DPL and OpenCL come from the `dev/bazel/deps/*.bzl` rules. Read those files for current versions rather than copying them from here.

## 🔧 Build Rules and Patterns

oneDAL does not use bare `cc_library` / `cc_test`. All modules go through the macros in `@onedal//dev/bazel:dal.bzl` (`dal_module`, `dal_test_suite`) and `@onedal//dev/bazel:daal.bzl` (`daal_module`). Compiler flags, CPU dispatch and threading are toolchain-owned (`dev/bazel/flags.bzl`); never hand-write `copts`. The ISAs live in `ISA_TABLE` (`dev/bazel/config/config.bzl`); rules read the target OS and arch from `TargetInfo` (`target(ctx)`, attrs merged from `TARGET_ATTRS`) instead of declaring constraint attributes.

```python
load("@onedal//dev/bazel:dal.bzl", "dal_module", "dal_test_suite")

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "core",
    auto = True,                       # globs sources by convention, excludes test/
    dal_deps = ["@onedal//cpp/oneapi/dal:core"],
    extra_deps = ["@onedal//cpp/daal/src/algorithms/pca:kernel"],
)

dal_module(
    name = "pca",
    dal_deps = [":core"],
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
    tests = [":interface_tests"],
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

Always scope targets, and pass `--config` to `bazel test` and `bazel run`; with it unset, tests include DPC++ targets, which are skipped when the toolchain has no DPC++ compiler (`--release_dpc` defaults to whether it has one: `icpx` in `PATH` on Linux, icx on Windows). `dev/bazel/README.md` lists every config.

For the default CPU build on x86-64 Linux, Bazel downloads MKL, TBB and the other dependencies with fallback archives at the versions pinned in `dev/bazel/deps/requirements_lock.txt` (PyPI wheels), `dev/bazel/deps/gcc.bzl` (conda) and `dev/bazel/deps/*.bzl`, so the host only needs Bazelisk, gcc or clang, `unzip` and `unzstd`. The `*ROOT` variables in the README point a dependency at a local install instead; OpenBLAS (`ref` backend), OpenRNG and `@onedal_release` have no fallback and always need one. A release or packaging job that must ship DPC++ passes `--release_dpc=true` (or `--config=release-dpc`), since the default follows the toolchain's DPC++ compiler.

## 🔧 Dependency Management

External libraries are referenced by their repository labels, e.g. `@mkl//:mkl_core`, `@tbb//:tbb`, `@openblas//:openblas`, `@mpi//:mpi`. `dev/bazel/deps/BUILD` declares no targets, only exports the lock; the `*.tpl.BUILD` files next to it are the BUILD templates for those repositories.

## 🎯 Development Guidelines

### Dependencies
- **Internal**: Full `@onedal//` labels in `dal_deps` (e.g., `@onedal//cpp/oneapi/dal:core`)
- **DAAL kernels**: `extra_deps` (e.g., `@onedal//cpp/daal/src/algorithms/pca:kernel`)
- **Visibility**: Set appropriate visibility levels

## 📝 Rules for Changes

- `//conditions:default` in a `select()` means every platform not listed, macOS included, not just Linux. Put Linux-only flags under `@platforms//os:linux`.
- Don't add `allow_empty = True` to a glob that must match; it hides packaging mistakes. The optional globs in `dal.bzl`, `daal.bzl` and `deps/*.tpl.BUILD` need it.
- Don't hardcode the workspace name in test paths; use `${TEST_WORKSPACE}`.
- No `use_default_shell_env = True`; it breaks hermeticity.
- Don't write globs that match several `.so` variants of one library; they produce duplicate link inputs.
- Remove unused `load()` symbols.
- Don't hardcode platform-specific paths; reference external libraries by their repository labels.
- Product and library binary versions of the Bazel build live only in `dev/bazel/version.bzl`. Until Make is removed, `makefile.ver` carries the same values; bump both. The build date is `--build_date=YYYYMMDD`, which the release jobs pass with `--config=release`; that config fails without it.
- No shell, batch or PowerShell in build actions: use `expand_template`, `write`, `symlink` or `run` with arguments. The one exception, `toolchains/tools/dpc_link_win.ps1`, is tracked in `TODO.md`.
- Checks of the configured graph are Starlark analysis tests in `dev/bazel/tests/analysis.bzl` (`bazel test //dev/bazel/tests:all`), not scripts that run `bazel` again.

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
