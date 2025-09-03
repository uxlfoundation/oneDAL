
# CI Infrastructure - AI Agents Context

> **Purpose**: Context for AI agents working with oneDAL's continuous integration infrastructure and validation processes.

## 🎯 CI Infrastructure Overview

**oneDAL** uses sophisticated CI/CD pipeline built on GitHub Actions with comprehensive quality gates:
- **Multi-Platform Validation**: Linux, Windows, macOS across multiple versions
- **Multi-Build System**: Make (production), CMake (integration), Bazel (development)  
- **Cross-Repository Integration**: Validation with scikit-learn-intelex
- **Performance Monitoring**: Automated performance regression detection

## 🏗️ CI Architecture

### CI Stages Pipeline
1. **Code Quality**: Formatting, linting, static analysis
2. **Build Validation**: Multi-build system compilation
3. **Testing**: Unit, integration, and performance tests
4. **Integration**: Cross-repository compatibility
5. **Security**: Vulnerability scanning and compliance
6. **Documentation**: Build and validation
7. **Performance**: Benchmark execution and analysis

## 🔧 Build System Validation

### Make Build (Production)
```bash
cd dev/make
make clean all
make test
make install
```

### CMake Build (Integration)
```bash
mkdir build && cd build
cmake .. -DCMAKE_CXX_STANDARD=17 -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
make test
```

### Bazel Build (Development)
```bash
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
- **Cross-Platform Tests**: Platform-specific validation

### Test Configuration
```python
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
- name: Performance Validation
  run: |
    cd dev/bazel
    bazel run //benchmarks:performance_suite
    python scripts/analyze_performance.py
    # Fail if performance regression > 5%
```

## 🔄 Cross-Repository Integration

### scikit-learn-intelex Validation
```bash
git clone https://github.com/intel/scikit-learn-intelex.git
cd scikit-learn-intelex
pip install -e .
python -m pytest tests/integration/test_onedal_integration.py
```

**Integration Aspects**:
- **API Compatibility**: Ensure oneDAL changes don't break scikit-learn-intelex
- **Performance Consistency**: Maintain performance improvements
- **Feature Parity**: Validate feature availability

## 🔍 CI Configuration

### Environment Configuration
```yaml
env:
  CXX_STANDARD: "17"
  BUILD_TYPE: "Release"
  ENABLE_GPU: "false"
  ENABLE_MPI: "false"
  ENABLE_OPENMP: "true"
```

### Matrix Strategy
```yaml
strategy:
  matrix:
    os: [ubuntu-20.04, ubuntu-22.04, windows-2019, windows-2022, macos-12]
    compiler: [gcc-9, gcc-10, gcc-11, clang-12, clang-13, msvc-2019]
    cxx_std: [14, 17]
    build_system: [make, cmake, bazel]
```

## 🛠️ Common Issues & Debugging

### Common CI Issues
- **Build Failures**: Compiler compatibility issues
- **Test Failures**: Platform-specific test issues  
- **Performance Regressions**: Algorithm performance degradation
- **Integration Failures**: Cross-repository compatibility issues

### Debugging Workflow
1. **Local Reproduction**: Reproduce issue locally
2. **Log Analysis**: Analyze CI logs for errors
3. **Environment Check**: Verify CI environment setup
4. **Platform Isolation**: Isolate platform-specific issues

## 🎯 CI Best Practices
- **Code Quality**: Use ClangFormat, static analysis, maintain test coverage
- **Performance**: Regular benchmarking, regression detection
- **Security**: Dependency scanning, code scanning, license compliance

## 📖 Further Reading
- **[AGENTS.md](../AGENTS.md)** - Main repository context
- **[deploy/AGENTS.md](../deploy/AGENTS.md)** - Deployment context
