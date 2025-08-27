# CI/CD Workflows Instructions for GitHub Copilot

## CI Infrastructure Overview

**oneDAL** uses GitHub Actions for continuous integration and deployment, with multiple validation stages and cross-repository considerations.

### Key CI Characteristics
- **Multi-Platform**: Linux, Windows, macOS validation
- **Multi-Build**: Make (production), CMake (integration), Bazel (development)
- **Cross-Repository**: Integration with scikit-learn-intelex validation
- **Performance Testing**: Automated performance benchmarks
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

## 🔧 CI Validation Stages

### Stage 1: Code Quality
```yaml
# Code quality checks
- name: Code Quality
  uses: actions/code-quality@v1
  with:
    tools: |
      - clang-format
      - cppcheck
      - clang-tidy
    fail_on: warning
```

### Stage 2: Build Validation
```yaml
# Multi-build system validation
- name: Make Build (Production)
  run: |
    cd dev/make
    make clean all
    
- name: CMake Build (Integration)
  run: |
    mkdir build && cd build
    cmake .. -DCMAKE_CXX_STANDARD=17
    make -j$(nproc)
    
- name: Bazel Build (Development)
  run: |
    cd dev/bazel
    bazel build //...
```

### Stage 3: Testing
```yaml
# Comprehensive testing
- name: Unit Tests
  run: |
    cd dev/bazel
    bazel test //... --test_output=all
    
- name: Integration Tests
  run: |
    cd examples
    make test-integration
    
- name: Performance Tests
  run: |
    cd dev/bazel
    bazel test //test/performance/...
```

### Stage 4: Cross-Repository Validation
```yaml
# scikit-learn-intelex integration validation
- name: Cross-Repository Tests
  run: |
    # Test oneDAL with scikit-learn-intelex
    python -m pytest tests/integration/test_sklearn_integration.py
```

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
        build_system: [make, cmake, bazel]
    
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
    cxx_std: [14, 17]
    build_type: [debug, release, relwithdebinfo]
```

## 🔍 Validation Processes

### Code Quality Validation
- **ClangFormat**: Code formatting consistency
- **CppCheck**: Static analysis and error detection
- **Clang-Tidy**: Modern C++ best practices
- **SonarQube**: Code quality and security analysis

### Build Validation
- **Make**: Production build validation
- **CMake**: End-user integration validation
- **Bazel**: Development workflow validation
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

### Benchmark Workflows
```yaml
# Performance validation workflow
- name: Performance Benchmarks
  run: |
    cd dev/bazel
    bazel run //benchmarks:kmeans_benchmark
    bazel run //benchmarks:svm_benchmark
    bazel run //benchmarks:random_forest_benchmark
```

### Performance Gates
- **Regression Detection**: Performance degradation prevention
- **Baseline Comparison**: Historical performance tracking
- **Platform Optimization**: Architecture-specific validation
- **Memory Profiling**: Memory usage optimization

### Performance Metrics
- **Execution Time**: Algorithm performance measurement
- **Memory Usage**: Memory efficiency validation
- **Scalability**: Multi-threading and distributed performance
- **Accuracy**: Numerical precision validation

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

## 🛠️ Troubleshooting

### Common CI Issues
- **Build Failures**: Compiler compatibility issues
- **Test Failures**: Platform-specific test issues
- **Performance Regressions**: Algorithm performance degradation
- **Integration Failures**: Cross-repository compatibility issues

### Debugging Steps
1. **Local Reproduction**: Reproduce issue locally
2. **Log Analysis**: Analyze CI logs for errors
3. **Environment Check**: Verify CI environment setup
4. **Dependency Validation**: Check dependency versions

### CI Maintenance
- **Regular Updates**: Keep CI tools updated
- **Performance Monitoring**: Monitor CI execution time
- **Resource Optimization**: Optimize CI resource usage
- **Documentation**: Keep CI documentation current

## 📚 Related Context Files

### For This Area
- **[AGENTS.md](../../AGENTS.md)** - Main repository context
- **[deploy/AGENTS.md](../../deploy/AGENTS.md)** - Deployment context

### For Other Areas
- **[Build Systems](../../.github/instructions/build-systems.md)** - Build configuration guidance
- **[Examples](../../.github/instructions/examples.md)** - Testing examples
- **[Documentation](../../.github/instructions/documentation.md)** - Documentation validation

## 🔄 Cross-Reference Navigation

**For CI Tasks**: Use this file for CI/CD workflow guidance
**For Build Issues**: Refer to build system context files
**For Integration**: Check scikit-learn-intelex compatibility
**For Deployment**: Use deployment context files

---

**Note**: This file provides CI/CD workflow context for AI agents. For detailed implementation, refer to the specific workflow files in the `.github/workflows/` directory.
