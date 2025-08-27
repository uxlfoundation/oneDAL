# Copyright contributors to the oneDAL project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# CI/CD Workflows Instructions for GitHub Copilot

## 🎯 **PRIMARY GOAL: PR Review Assistance**

**GitHub Copilot's main purpose is to assist with PR reviews. CI workflows provide validation for PR review scenarios.**

## CI Infrastructure Overview

**oneDAL** uses GitHub Actions for continuous integration and deployment, with multiple validation stages and cross-repository considerations.

### Key CI Characteristics
- **Multi-Platform**: Linux, Windows validation
- **Multi-Build**: Make (🔴 production), Bazel (🟢 development/testing)
- **Cross-Repository**: Integration with scikit-learn-intelex validation
- **Quality Gates**: Code quality, security, and compliance checks

## 🏗️ CI Workflow Structure

### Core Workflows
```
.github/workflows/
├── ci.yml                    # Main CI workflow
├── ci-build.yml             # Build validation workflow
├── ci-test.yml              # Testing workflow
├── ci-performance.yml       # Performance validation
├── ci-security.yml          # Security scanning
├── ci-docs.yml              # Documentation validation
└── ci-release.yml           # Release workflow
```

### Workflow Triggers
- **Push to main**: Full CI validation
- **Pull Request**: Build and test validation
- **Release tag**: Complete release validation
- **Manual dispatch**: On-demand validation

## 📋 CI Configuration Files

### GitHub Actions Configuration
```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        compiler: [gcc, clang, msvc]
        build_system: [make, cmake, bazel]  # Note: make is PRIMARY
    
    steps:
      - uses: actions/checkout@v3
      - name: Setup ${{ matrix.compiler }}
      - name: Build with ${{ matrix.build_system }}
      - name: Run tests
      - name: Performance validation
```

### Build Matrix Strategy
```yaml
# Build matrix for comprehensive validation
strategy:
  matrix:
    os: [ubuntu-20.04, ubuntu-22.04, windows-2019, windows-2022, macos-12, macos-13]
    compiler: [gcc-9, gcc-10, gcc-11, clang-12, clang-13, clang-14, msvc-2019, msvc-2022]
    cxx_std: [14, 17]  # Note: 17 is maximum for compatibility
    build_type: [debug, release, relwithdebinfo]
```

## 🔍 Validation Processes

### Code Quality Validation
- **ClangFormat**: Code formatting consistency
- **CppCheck**: Static analysis and error detection
- **Clang-Tidy**: Modern C++ best practices

### Build Validation
- **Cross-compilation**: Multi-architecture support

### Testing Validation
- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **Performance Tests**: Benchmark validation
- **Regression Tests**: Historical issue prevention

### Security Validation
- **Dependency Scanning**: Vulnerability detection
- **Code Scanning**: Security issue identification
- **License Compliance**: Open source compliance
- **SBOM Generation**: Software bill of materials

## 🚀 Performance Validation

## 🔄 Cross-Repository Integration

### scikit-learn-intelex Validation
```yaml
# Cross-repository integration testing
- name: scikit-learn-intelex Integration
  run: |
    # Clone and test scikit-learn-intelex with oneDAL
    git clone https://github.com/intel/scikit-learn-intelex.git
    cd scikit-learn-intelex
    pip install -e .
    python -m pytest tests/ -v
```

### Integration Validation
- **API Compatibility**: Ensure oneDAL changes don't break scikit-learn-intelex
- **Performance Consistency**: Maintain performance improvements
- **Feature Parity**: Validate feature availability
- **Error Handling**: Consistent error behavior

## 📊 CI Metrics and Reporting

### Quality Metrics
- **Build Success Rate**: Percentage of successful builds
- **Test Coverage**: Code coverage percentage
- **Performance Regression**: Performance change tracking
- **Security Issues**: Security vulnerability count

### Reporting
- **GitHub Status**: Build status integration
- **Slack Notifications**: Team communication
- **Email Alerts**: Critical failure notifications
- **Dashboard**: CI metrics visualization

## 📚 Related Context Files

### For This Area
- **[AGENTS.md](../../AGENTS.md)** - Main repository context
- **[deploy/AGENTS.md](../../deploy/AGENTS.md)** - Deployment context

### For Other Areas
- **[Build Systems](build-systems.md)** - Build configuration guidance
- **[Examples](examples.md)** - Testing examples
- **[Documentation](documentation.md)** - Documentation validation

## 🔄 Cross-Reference Navigation

**For CI Tasks**: Use this file for CI/CD workflow guidance
**For Build Issues**: Refer to build system context files
**For Integration**: Check scikit-learn-intelex compatibility
**For Deployment**: Use deployment context files

**Note**: This file provides CI/CD workflow context for AI agents. For detailed implementation, refer to the specific workflow files in the `.github/workflows/` directory.


