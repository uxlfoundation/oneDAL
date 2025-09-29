
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
daal/
├── cpp/        # Core C++ implementation
│   ├── daal/   # Traditional DAAL interface
│   └── oneapi/ # Modern oneAPI interface
├── dev/        # Development tools and build configs
├── examples/   # Usage examples and tutorials
├── docs/       # Documentation and API references
└── deploy/     # Deployment and packaging
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

### Documentation, Examples & Infrastructure
- **[docs/AGENTS.md](docs/AGENTS.md)** - Documentation structure and guidelines
- **[examples/AGENTS.md](examples/AGENTS.md)** - Example code patterns and usage
- **[deploy/AGENTS.md](deploy/AGENTS.md)** - Deployment and distribution context
- **[ci/AGENTS.md](ci/AGENTS.md)** - CI/CD infrastructure context

## 📋 Critical Development Rules

### Code Style and Standards
- **ClangFormat**: Use project's `.clang-format` configuration
- **EditorConfig**: Follow `.editorconfig` rules
- **Modern C++**: Use C++14/17 features appropriately
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

## 🔍 Key Files
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines
- **[INSTALL.md](INSTALL.md)** - Build and installation instructions
- **[MODULE.bazel](MODULE.bazel)** - Bazel module configuration
- **[.clang-format](.clang-format)** - Code formatting rules

## 📚 Additional Resources
- **API Documentation**: [oneDAL Developer Guide](https://uxlfoundation.github.io/oneDAL/)
- **Coding Guidelines**: [Detailed coding guide](https://uxlfoundation.github.io/oneDAL/contribution/coding_guide.html)
- **CPU Features**: [CPU feature dispatching guide](https://uxlfoundation.github.io/oneDAL/contribution/cpu_features.html)
- **Threading**: [Threading layer guide](https://uxlfoundation.github.io/oneDAL/contribution/threading.html)

---

**Note**: This file serves as the main entry point. For specific implementation details, refer to the relevant sub-AGENTS.md file in the appropriate directory.

