# CI/CD Workflows Instructions for GitHub Copilot

## 🎯 **PRIMARY GOAL: PR Review Assistance**

**GitHub Copilot's main purpose is to assist with PR reviews. CI workflows provide validation for PR review scenarios.**

## CI Infrastructure Overview

oneDAL uses GitHub Actions for CI/CD with multi-platform validation and cross-repository integration with scikit-learn-intelex.

### Key Characteristics
- **Multi-Platform**: Linux, Windows, macOS validation
- **Multi-Build**: Make (🔴 production primary), Bazel (🟢 development), CMake (🟡 integration)
- **Cross-Repository**: scikit-learn-intelex compatibility validation
- **Quality Gates**: Code quality, security, performance validation

## Core CI Workflows

```
.github/workflows/
├── ci.yml                # Main CI workflow
├── ci-build.yml         # Build validation
├── ci-test.yml          # Testing workflow  
├── ci-performance.yml   # Performance validation
├── ci-security.yml      # Security scanning
└── ci-docs.yml          # Documentation validation
```

### Workflow Triggers
- **Push to main**: Full validation pipeline
- **Pull Request**: Build and test validation
- **Release tag**: Complete release validation

## Build Matrix Strategy

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest]
    compiler: [gcc, clang, msvc]
    build_system: [make, cmake, bazel]  # make is PRIMARY
    cxx_std: [14, 17]  # 17 maximum for compatibility
    build_type: [debug, release]
```

## Validation Processes

### Code Quality
- **ClangFormat**: Code formatting consistency
- **CppCheck**: Static analysis and error detection
- **Clang-Tidy**: Modern C++ best practices

### Testing Validation
- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **Performance Tests**: Benchmark validation
- **Regression Tests**: Historical issue prevention

### Security Validation
- **Dependency Scanning**: Vulnerability detection
- **Code Scanning**: Security issue identification
- **License Compliance**: Open source compliance

## Cross-Repository Integration

### scikit-learn-intelex Validation
```yaml
- name: scikit-learn-intelex Integration
  run: |
    git clone https://github.com/intel/scikit-learn-intelex.git
    cd scikit-learn-intelex
    pip install -e .
    python -m pytest tests/ -v
```

**Integration Requirements**:
- API compatibility maintained
- Performance improvements preserved
- Feature parity validated
- Consistent error handling

## CI Metrics

### Quality Metrics
- **Build Success Rate**: Percentage of successful builds
- **Test Coverage**: Code coverage percentage
- **Performance Regression**: Performance change tracking

### Reporting
- **GitHub Status**: Build status integration
- **Team Notifications**: Critical failure alerts
- **Dashboard**: CI metrics visualization

## Cross-Reference
- **[build-systems.md](build-systems.md)** - Build configuration guidance
- **[AGENTS.md](../../AGENTS.md)** - Main repository context
- **[deploy/AGENTS.md](../../deploy/AGENTS.md)** - Deployment context