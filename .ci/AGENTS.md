# oneDAL CI Infrastructure Documentation

## Overview

This document describes the CI infrastructure for oneDAL (Intel Data Analytics Library)
## Directory Structure

- `.ci/pipeline/ci.yml`: the Azure DevOps pipeline (build matrix, `FormatterChecks`); `docs.yml` builds the docs.
- `.ci/env/`: dependency installers. `apt.sh` takes a component name (`dev-base`, `mkl`, ...); `tbb`, `openblas` and `bazelisk` each have `.sh` and Windows variants.
- `.ci/scripts/`: `build.sh` / `build.bat` (compiler, optimization, backend and cross-compile options), `test.sh` / `test.bat`, `clang-format.sh`, `abi_check.sh`, and the Windows release checks (`compare_windows_release.ps1`, `test_bazel_release_cmake_example.ps1`).
- `.github/workflows/`: GitHub Actions. `ci.yml`, `ci-win.yml` and `ci-aarch64.yml` build and test; `nightly-build.yml` produces artifacts other repositories download (see `.github/AGENTS.md`).

## CI/CD Architecture

### Multi-Platform Strategy
The oneDAL CI infrastructure supports:
- **Architectures**: x86-64, AArch64, RISC-V
- **Operating Systems**: Ubuntu (multiple versions), Windows Server 2022
- **Compilers**: GNU GCC, Clang, Intel DPC++/ICX
- **Build Systems**: Make, Bazel
- **Hardware**: CPU and GPU (Intel) testing

### CI Platform Integration
- **GitHub Actions**: Primary CI/CD platform for public workflows
- **Azure DevOps**: Extended validation and internal testing
- **Renovate**: Dependency update automation
- **Codefactor**: Code quality analysis

### Build Matrix Configuration
The CI system employs comprehensive build matrices covering:
- **Instruction Sets**: AVX2, AVX-512, and architecture-specific optimizations
- **Backends**: Intel MKL, OpenBLAS reference implementations
- **Threading**: Intel TBB, OpenMP
- **Random Number Generation**: Intel MKL RNG, OpenRNG

### Quality Assurance
- **Code Formatting**: clang-format enforcement
- **License Compliance**: Automated header checking
- **ABI Compatibility**: Binary interface stability validation
- **Security**: OpenSSF Scorecard integration
- **Documentation**: Automated doc generation and validation

## Rules for Changes
- Pipelines call scripts in `.ci/scripts/` and `.ci/env/`. Change the script, not only the pipeline YAML that calls it.
- Style checks (clang-format, editorconfig-checker) run only in Azure `FormatterChecks`. See "Verification Before You Push" in the root `AGENTS.md`.

## Usage Guidelines

### Local Development
Developers can leverage the CI scripts locally:
```bash
# Set up development environment
.ci/env/apt.sh dev-base
.ci/env/apt.sh mkl

# Build the library
.ci/scripts/build.sh --compiler gnu --optimizations avx2 --target daal

# Run tests
.ci/scripts/test.sh
```

### Internal CI
Internal CI integration done in separate repository, though checks are enforced in PRs. 