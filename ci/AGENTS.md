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

# CI Infrastructure - AI Agents Context

> **Purpose**: This file provides comprehensive context for AI agents working with oneDAL's continuous integration infrastructure, validation processes, and workflow management.

## 🎯 CI Infrastructure Overview

**oneDAL** uses a sophisticated CI/CD pipeline built on GitHub Actions with multiple validation stages, cross-repository integration, and comprehensive quality gates.

### Key CI Characteristics
- **Multi-Platform Validation**: Linux, Windows, macOS across multiple versions
- **Multi-Build System**: Make (production), CMake (integration), Bazel (development)
- **Cross-Repository Integration**: Validation with scikit-learn-intelex
- **Performance Monitoring**: Automated performance regression detection
- **Quality Gates**: Code quality, security, and compliance validation

## 🏗️ CI Architecture

### Workflow Organization
```
.github/workflows/
├── ci.yml                    # Main CI orchestration
├── ci-build.yml             # Build system validation
├── ci-test.yml              # Testing and validation
├── ci-performance.yml       # Performance benchmarking
├── ci-security.yml          # Security scanning
├── ci-docs.yml              # Documentation validation
├── ci-release.yml           # Release validation
└── ci-integration.yml       # Cross-repository testing
```

### CI Stages Pipeline
1. **Code Quality**: Formatting, linting, static analysis
2. **Build Validation**: Multi-build system compilation
3. **Testing**: Unit, integration, and performance tests
4. **Integration**: Cross-repository compatibility
5. **Security**: Vulnerability scanning and compliance
6. **Documentation**: Build and validation
7. **Performance**: Benchmark execution and analysis

## 🔧 Build System Validation

### Make Build Validation (Production)
```bash
# Production build validation
cd dev/make
make clean all
make test
make install
```

### CMake Build Validation (Integration)
```bash
# End-user integration validation
mkdir build && cd build
cmake .. -DCMAKE_CXX_STANDARD=17 -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
make test
```

### Bazel Build Validation (Development)
```bash
# Development workflow validation
cd dev/bazel
bazel build //...
bazel test //... --test_output=all
bazel run //benchmarks:all
```

## 🧪 Testing Infrastructure

### Test Categories
- **Unit Tests**: Individual component validation
- **Integration Tests**: Component interaction validation
- **Performance Tests**: Benchmark execution and validation
- **Regression Tests**: Historical issue prevention
- **Cross-Platform Tests**: Platform-specific validation

### Test Execution
```bash
# Bazel test execution
bazel test //... --test_output=all --test_timeout=300

# Performance test execution
bazel run //benchmarks:kmeans_benchmark
bazel run //benchmarks:svm_benchmark
bazel run //benchmarks:random_forest_benchmark

# Integration test execution
cd examples
make test-integration
```

### Test Configuration
```python
# Bazel test configuration
cc_test(
    name = "algorithm_test",
    srcs = ["test_algorithm.cpp"],
    deps = [":algorithm_lib", "//dev/bazel/deps:gtest"],
    size = "medium",
    timeout = "short",
    copts = ["-std=c++17"],
)
```

## 🚀 Performance Validation

### Benchmark Infrastructure
- **Algorithm Benchmarks**: K-means, SVM, Random Forest, etc.
- **Memory Profiling**: Memory usage optimization
- **Scalability Testing**: Multi-threading and distributed performance
- **Platform Optimization**: Architecture-specific validation

### Performance Gates
```yaml
# Performance regression detection
- name: Performance Validation
  run: |
    cd dev/bazel
    bazel run //benchmarks:performance_suite
    python scripts/analyze_performance.py
    # Fail if performance regression > 5%
```

### Performance Metrics
- **Execution Time**: Algorithm performance measurement
- **Memory Usage**: Memory efficiency validation
- **Throughput**: Data processing capacity
- **Accuracy**: Numerical precision validation

## 🔄 Cross-Repository Integration

### scikit-learn-intelex Validation
```bash
# Cross-repository integration testing
git clone https://github.com/intel/scikit-learn-intelex.git
cd scikit-learn-intelex
pip install -e .
python -m pytest tests/ -v

# Integration validation
python -m pytest tests/integration/test_onedal_integration.py
```

### Integration Validation Aspects
- **API Compatibility**: Ensure oneDAL changes don't break scikit-learn-intelex
- **Performance Consistency**: Maintain performance improvements
- **Feature Parity**: Validate feature availability
- **Error Handling**: Consistent error behavior across repositories

## 📊 CI Metrics and Reporting

### Quality Metrics
- **Build Success Rate**: Percentage of successful builds
- **Test Coverage**: Code coverage percentage
- **Performance Regression**: Performance change tracking
- **Security Issues**: Security vulnerability count
- **Integration Success**: Cross-repository compatibility rate

### Reporting Channels
- **GitHub Status**: Build status integration
- **Slack Notifications**: Team communication
- **Email Alerts**: Critical failure notifications
- **Dashboard**: CI metrics visualization
- **Performance Reports**: Automated performance analysis

## 🛠️ Troubleshooting and Maintenance

### Common CI Issues
- **Build Failures**: Compiler compatibility issues
- **Test Failures**: Platform-specific test issues
- **Performance Regressions**: Algorithm performance degradation
- **Integration Failures**: Cross-repository compatibility issues
- **Resource Exhaustion**: CI resource limitations

### Debugging Workflow
1. **Local Reproduction**: Reproduce issue locally
2. **Log Analysis**: Analyze CI logs for errors
3. **Environment Check**: Verify CI environment setup
4. **Dependency Validation**: Check dependency versions
5. **Platform Isolation**: Isolate platform-specific issues

### CI Maintenance
- **Regular Updates**: Keep CI tools updated
- **Performance Monitoring**: Monitor CI execution time
- **Resource Optimization**: Optimize CI resource usage
- **Documentation**: Keep CI documentation current
- **Tool Evaluation**: Evaluate new CI tools and practices

## 🔍 CI Configuration Management

### Environment Configuration
```yaml
# CI environment setup
env:
  CXX_STANDARD: "17"
  BUILD_TYPE: "Release"
  ENABLE_GPU: "false"
  ENABLE_MPI: "false"
  ENABLE_OPENMP: "true"
```

### Matrix Strategy
```yaml
# Build matrix for comprehensive validation
strategy:
  matrix:
    os: [ubuntu-20.04, ubuntu-22.04, windows-2019, windows-2022, macos-12, macos-13]
    compiler: [gcc-9, gcc-10, gcc-11, clang-12, clang-13, clang-14, msvc-2019, msvc-2022]
    cxx_std: [14, 17]
    build_type: [debug, release, relwithdebinfo]
    build_system: [make, cmake, bazel]
```

### Conditional Execution
```yaml
# Conditional CI execution
- name: GPU Tests
  if: matrix.os == 'ubuntu-latest' && matrix.enable_gpu == 'true'
  run: |
    bazel test //test/gpu/...
```

## 📚 Related Context Files

### For This Area
- **[AGENTS.md](../../AGENTS.md)** - Main repository context
- **[deploy/AGENTS.md](../../deploy/AGENTS.md)** - Deployment context

### For Other Areas
- **[Build Systems](../../.github/instructions/build-systems.md)** - Build configuration guidance
- **[Examples](../../.github/instructions/examples.md)** - Testing examples
- **[Documentation](../../.github/instructions/documentation.md)** - Documentation validation

## 🔄 Cross-Reference Navigation

**For CI Tasks**: Use this file for CI infrastructure guidance
**For Build Issues**: Refer to build system context files
**For Integration**: Check scikit-learn-intelex compatibility
**For Deployment**: Use deployment context files
**For Workflows**: Refer to CI workflows documentation

## 🎯 CI Best Practices

### Code Quality
- **Formatting**: Use ClangFormat for consistent code style
- **Static Analysis**: Run CppCheck and Clang-Tidy
- **Coverage**: Maintain high test coverage
- **Documentation**: Keep documentation synchronized

### Performance
- **Benchmarks**: Regular performance benchmarking
- **Regression Detection**: Automated performance regression detection
- **Optimization**: Continuous performance optimization
- **Profiling**: Memory and CPU profiling

### Security
- **Dependency Scanning**: Regular vulnerability scanning
- **Code Scanning**: Security issue identification
- **License Compliance**: Open source compliance
- **SBOM Generation**: Software bill of materials

---

**Note**: This file provides CI infrastructure context for AI agents. For detailed workflow implementation, refer to the `.github/workflows/` directory and the CI workflows documentation.
