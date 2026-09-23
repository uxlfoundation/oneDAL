
# oneDAL Repository - AI Agents Context Guide

> **Purpose**: Comprehensive context for AI agents working with the oneDAL repository structure, coding standards, and development guidelines.

## 🎯 Repository Overview

**oneDAL** (oneAPI Data Analytics Library) is a high-performance C++ library for machine learning algorithms, providing both traditional DAAL interfaces and modern oneAPI interfaces with SYCL support for GPU acceleration.

**Integration Note**: oneDAL works with [scikit-learn-intelex](https://github.com/intel/scikit-learn-intelex). They share common validation aspects and provide accelerated machine learning capabilities together.

### Key Characteristics
- **Language**: Modern C++ (17+)
- **Architecture**: Dual interface system (DAAL + oneAPI)
- **Build Systems**: Make (production), CMake (integration), Bazel (development/testing)
- **Targets**: CPU (SIMD optimized), GPU (SYCL), Distributed (MPI)
- **License**: Apache License 2.0

## 🏗️ Repository Structure

```
oneDAL/
├── cpp/            # Library sources
│   ├── daal/       # DAAL interface and CPU kernels
│   └── oneapi/     # oneAPI interface (C++ and DPC++)
├── dev/            # Build tooling: Bazel rules (dev/bazel), Make fragments (dev/make)
├── examples/       # Examples for the DAAL and oneAPI interfaces
├── samples/        # Distributed (MPI/CCL) samples
├── docs/           # Documentation sources
├── data/           # Datasets used by examples and tests
├── deploy/         # Packaging and environment scripts
├── cmake/          # CMake config templates for installed releases
├── conda-recipe/   # Conda package recipe
└── .ci/            # CI pipelines, environment setup and build/test scripts
```

## 🔗 Context Files for AI Agents

Specialized AGENTS.md files for detailed context:

### Core Implementation
- **[cpp/AGENTS.md](cpp/AGENTS.md)** - C++ implementation details and patterns
- **[cpp/daal/AGENTS.md](cpp/daal/AGENTS.md)** - Traditional DAAL interface context
- **[cpp/oneapi/AGENTS.md](cpp/oneapi/AGENTS.md)** - Modern oneAPI interface context

### Build Systems & Development
- **[dev/AGENTS.md](dev/AGENTS.md)** - Development tools and build system context
- **[dev/bazel/AGENTS.md](dev/bazel/AGENTS.md)** - Bazel build system specifics
- **[dev/make/AGENTS.md](dev/make/AGENTS.md)** - Make build fragments

### Documentation, Examples & Infrastructure
- **[docs/AGENTS.md](docs/AGENTS.md)** - Documentation structure and guidelines
- **[examples/AGENTS.md](examples/AGENTS.md)** - Example code patterns and usage
- **[deploy/AGENTS.md](deploy/AGENTS.md)** - Deployment and distribution context
- **[.ci/AGENTS.md](.ci/AGENTS.md)** - CI/CD infrastructure context
- **[.github/AGENTS.md](.github/AGENTS.md)** - Workflow constraints (`nightly-build.yml`)

## 📋 Critical Development Rules

### Code Style and Standards
- **ClangFormat**: Configs are per source tree (`cpp/daal/`, `cpp/oneapi/`, `examples/*/`, `samples/*/`, `dev/l0_tools/`); there is no root `.clang-format`
- **EditorConfig**: Follow `.editorconfig` rules
- **Modern C++**: C++17 (no C++20/23 features)
- **STL**: Leverage standard library containers and algorithms
- **RAII**: Follow Resource Acquisition Is Initialization principles

### Architecture Patterns
- **Interface Design**: Follow existing DAAL/oneAPI patterns
- **Memory Management**: Use smart pointers and RAII
- **Threading**: Use oneDAL threading layer, not direct primitives
- **CPU Features**: Implement CPU feature dispatching for optimizations

### Testing and Validation
- **Build Tests**: All changes must pass build system validation
- **Examples**: Ensure examples build and run correctly
- **Documentation**: Update relevant documentation

## 🚀 Quick Start for AI Agents

1. **Understand Context**: Read relevant AGENTS.md file for your task
2. **Follow Patterns**: Study existing code in similar areas
3. **Respect Standards**: Apply coding guidelines consistently
4. **Test Thoroughly**: Ensure changes work with build system

### 🔄 Cross-Repository Considerations
- **scikit-learn-intelex integration impact**
- **API compatibility preservation**
- **Performance consistency maintenance**

## 📝 Rules for Changes

These come from recurring maintainer review comments. Directory-specific rules are in the nearest `AGENTS.md`.

- Comments describe the code as it will be once merged. Don't reference discarded approaches, narrate the change, or mention "this PR".
- One PR, one logical change. Drive-by fixes, renames and mechanical changes (formatting, generated code, mass renames) go in their own PRs.
- Search before adding a helper, constant table or validation routine. Extend the existing one and name it in the PR description.
- Don't add a lock, critical section, guard or redundant check unless you can name the failure it prevents.
- A bug fix comes with a test that fails without the fix.
- Don't hardcode versions, URLs or paths that have a source of truth (`makefile.ver`, `MODULE.bazel`, `.github/renovate.json`).
- New files use the header `Copyright contributors to the oneDAL project`. Leave existing headers alone.
- ASCII only in source, comments and docs. Keep each file's existing line endings.
- Scripts with a `#!/bin/sh` shebang use POSIX `sh` only. `.bat` files follow `cmd.exe` quoting; don't mix PowerShell and CMD syntax.
- Bash scripts start with `set -euo pipefail`, use `mkdir -p`, and don't silence failures with a bare `|| true`.

## ✅ Verification Before You Push

### Format and style (blocking: Azure `FormatterChecks`)

```bash
pip install pre-commit && pre-commit install   # one-time
pre-commit run --all-files
editorconfig-checker
```

The pre-commit hook checks the same files as CI, with the same clang-format version (20.1.8; other versions format differently). To run exactly what CI runs, without modifying files: `CLANG_FORMAT_EXE=clang-format-20 .ci/scripts/clang-format.sh`.

### Tests (Bazel)

```bash
bazel test --config=host //cpp/oneapi/dal/algo/<algo>:tests   # one algorithm, CPU only
bazel test --config=host //cpp/oneapi/dal:tests               # oneAPI interface, CPU only
bazel test --config=dpc --device=gpu //cpp/oneapi/dal:tests   # DPC++ on GPU
```

Without `--config`, Bazel builds and runs all tests, including DPC++ ones that need the Intel DPC++ compiler. See `dev/bazel/README.md`.

### Full build (Make)

```bash
make -f makefile daal oneapi_c PLAT=lnx32e -j$(nproc)
```

See `INSTALL.md` for other platforms and build variants.

### Where the checks live

| Check | System | Config |
| --- | --- | --- |
| clang-format, editorconfig-checker | Azure DevOps | `.ci/pipeline/ci.yml` (`FormatterChecks`) |
| Make (GNU/MKL, LLVM/OpenBLAS rv64, VC, Intel), Bazel, release compare, sklearnex | Azure DevOps | `.ci/pipeline/ci.yml` |
| Make + DPC++ (icx), ABI check, Make GNU/MKL conda | GitHub Actions | `.github/workflows/ci.yml` |
| Windows (incl. arm64) | GitHub Actions | `.github/workflows/ci-win.yml` |
| aarch64 | GitHub Actions | `.github/workflows/ci-aarch64.yml` |
| License headers | GitHub Actions | `.github/workflows/skywalking-eyes.yml` |
| Bazel Linux/Windows | GitHub Actions (nightly) | `.github/workflows/nightly-test.yml` |

Style is not gated in GitHub Actions: a green Actions run does not mean formatting passes.

## 🔍 Key Files
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines
- **[INSTALL.md](INSTALL.md)** - Build and installation instructions
- **[MODULE.bazel](MODULE.bazel)** - Bazel module configuration

## 📚 Additional Resources
- **API Documentation**: [oneDAL Developer Guide](https://uxlfoundation.github.io/oneDAL/)
- **Coding Guidelines**: [Detailed coding guide](https://uxlfoundation.github.io/oneDAL/contribution/coding_guide.html)
- **CPU Features**: [CPU feature dispatching guide](https://uxlfoundation.github.io/oneDAL/contribution/cpu_features.html)
- **Threading**: [Threading layer guide](https://uxlfoundation.github.io/oneDAL/contribution/threading.html)

---

**Note**: This file serves as the main entry point. For specific implementation details, refer to the relevant sub-AGENTS.md file in the appropriate directory.

