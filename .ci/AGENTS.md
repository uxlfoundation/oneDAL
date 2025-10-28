# oneDAL CI Infrastructure Documentation

## Overview

This document describes the CI infrastructure for oneDAL (Intel Data Analytics Library)
## Directory Structure

### `.ci` Folder

The `.ci` folder contains the core CI infrastructure scripts and configurations organized into three main subdirectories:

#### `.ci/env/` - Environment Setup Scripts
- **`apt.sh`** - Package installation script for Ubuntu/Debian systems with functions for:
  - Intel OneAPI toolkit components (DPC++, TBB, DPL, MKL)
  - Development tools (clang-format, editorconfig-checker)
  - Base development packages and dependencies
- **`bazelisk.sh`** - Bazel build system setup and installation
- **`editorconfig-checker.sh`** - EditorConfig compliance checker installation
- **`environment.yml`** - Conda environment specification
- **`openblas.sh`** - OpenBLAS library installation and configuration
- **`openrng.sh`** - OpenRNG backend setup for random number generation
- **`tbb.sh`** / **`tbb.bat`** - Intel TBB (Threading Building Blocks) setup for Linux/Windows
- **`riscv64-clang-crosscompile-toolchain.cmake`** - RISC-V cross-compilation toolchain configuration

#### `.ci/pipeline/` - CI Pipeline Definitions
- **`ci.yml`** - Main Azure DevOps pipeline configuration with:
  - Multi-platform build matrices (Linux, Windows)
  - Compiler configurations (GNU, Clang, Intel)
  - Build targets (daal, onedal_c)
  - Testing and validation jobs
  - Artifact publishing
- **`docs.yml`** - Documentation build and deployment pipeline

#### `.ci/scripts/` - Build and Test Scripts
- **`build.sh`** / **`build.bat`** - Cross-platform build orchestration with support for:
  - Multiple compilers (gnu, clang, icx)
  - Architecture optimizations (AVX2, etc.)
  - Backend configurations (MKL, reference implementations)
  - Cross-compilation capabilities
- **`test.sh`** / **`test.bat`** - Comprehensive testing framework execution
- **`clang-format.sh`** - Code formatting verification
- **`describe_system.sh`** - System information collection for debugging
- **`abi_check.sh`** - ABI compatibility verification
- **`install_basekit.bat`** - Intel OneAPI Base Toolkit installation for Windows
- **`collect_opencl_rt.ps1`** - OpenCL runtime collection script

### `.github/workflows/` - GitHub Actions Workflows

#### Core CI Workflows
- **`ci.yml`** - Main CI pipeline for x86 platforms with DPC++ builds
- **`ci-aarch64.yml`** - AArch64 (ARM64) specific CI pipeline
- **`nightly-build.yml`** / **`nightly-test.yml`** - Automated nightly builds and testing

#### Specialized Workflows
- **`docker-validation-ci.yml`** / **`docker-validation-nightly.yml`** - Container-based validation
- **`docs-release.yml`** - Documentation deployment and release management
- **`label-enforcement.yml`** - PR labeling automation
- **`pr-checklist.yml`** - Pull request compliance verification
- **`renovate-validation.yml`** - Dependency update validation
- **`skywalking-eyes.yml`** - License header compliance checking
- **`slack-pr-notification.yml`** - Team notification system
- **`openssf-scorecard.yml`** - Security scorecard assessment

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
- **Mergify**: Automated merge management
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