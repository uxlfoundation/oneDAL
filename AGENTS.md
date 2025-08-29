
# oneDAL Repository - AI Agents Context Guide

> **Purpose**: This file provides comprehensive context for AI agents (including GitHub Copilot) to understand the oneDAL repository structure, coding standards, and development guidelines.

## 🎯 Repository Overview

**oneDAL** (oneAPI Data Analytics Library) is a high-performance C++ library for machine learning algorithms, providing both traditional DAAL interfaces and modern oneAPI interfaces with SYCL support for GPU acceleration.

**Integration Note**: oneDAL works together with the [scikit-learn-intelex](https://github.com/intel/scikit-learn-intelex) project. While they are separate repositories, they share common validation aspects and work together to provide accelerated machine learning capabilities. When working on integration features or validation, consider both repositories.

### Key Characteristics
- **Language**: Modern C++ (C++14/17+)
- **Architecture**: Dual interface system (DAAL + oneAPI)
- **Build Systems**: Make (PRIMARY for production), CMake (end-user integration), Bazel (development/testing)
- **Targets**: CPU (SIMD optimized), GPU (SYCL), Distributed (MPI)
- **License**: Apache License 2.0

## 🏗️ Repository Structure

```
daal/
├── cpp/                    # Core C++ implementation
│   ├── daal/              # Traditional DAAL interface
│   └── oneapi/            # Modern oneAPI interface
├── dev/                    # Development tools and build configs
│   ├── bazel/             # Bazel build system configuration (development/testing)
│   └── make/              # Make-based build system (PRIMARY for production)
├── examples/               # Usage examples and tutorials
├── docs/                   # Documentation and API references
├── samples/                # Advanced usage samples
└── cmake/                  # CMake build system support (end-user integration)
```

## 🔗 Context Files for AI Agents

For detailed context about specific areas, refer to these specialized AGENTS.md files:

### Core Implementation
- **[cpp/AGENTS.md](cpp/AGENTS.md)** - C++ implementation details and patterns
- **[cpp/daal/AGENTS.md](cpp/daal/AGENTS.md)** - Traditional DAAL interface context
- **[cpp/oneapi/AGENTS.md](cpp/oneapi/AGENTS.md)** - Modern oneAPI interface context

### Build Systems
- **[dev/AGENTS.md](dev/AGENTS.md)** - Development tools and build system context
- **[dev/make/AGENTS.md](dev/make/AGENTS.md)** - Make build system specifics (🔴 PRIMARY for production builds)
- **[dev/bazel/AGENTS.md](dev/bazel/AGENTS.md)** - Bazel build system specifics (🟡 development and testing only)

### Documentation and Examples
- **[docs/AGENTS.md](docs/AGENTS.md)** - Documentation structure and guidelines
- **[examples/AGENTS.md](examples/AGENTS.md)** - Example code patterns and usage
- **[samples/AGENTS.md](samples/AGENTS.md)** - Advanced sample implementations

### Deployment and Distribution
- **[deploy/AGENTS.md](deploy/AGENTS.md)** - Deployment, packaging, and distribution context

### CI Infrastructure
- **[ci/AGENTS.md](ci/AGENTS.md)** - CI/CD infrastructure and workflow context

## 🤖 GitHub Copilot Instructions

For GitHub Copilot users, specialized instruction files are available in the `.github/instructions/` directory:

- **[.github/instructions/README.md](.github/instructions/README.md)** - Overview of Copilot instructions
- **[.github/instructions/general.md](.github/instructions/general.md)** - General repository rules and context
- **[.github/instructions/cpp.md](.github/instructions/cpp.md)** - C++ development guidelines
- **[.github/instructions/build-systems.md](.github/instructions/build-systems.md)** - Build system configurations
- **[.github/instructions/documentation.md](.github/instructions/documentation.md)** - Documentation standards
- **[.github/instructions/examples.md](.github/instructions/examples.md)** - Example code patterns

### 🔄 Cross-Reference Navigation

**For Copilot Instructions**: Use the files above for detailed guidance
**For AI Agent Context**: Use the specialized AGENTS.md files below for implementation details
**For PR Review**: Focus on the PR Review Assistance sections in instruction files

## 📋 Critical Development Rules

### 1. Code Style and Formatting
- **ClangFormat**: Use project's `.clang-format` configuration
- **EditorConfig**: Follow `.editorconfig` rules
- **Copyright**: Include proper copyright headers (see CONTRIBUTING.md)

### 2. C++ Standards and Best Practices
- **Modern C++**: Use C++14/17 features appropriately
- **STL**: Leverage standard library containers and algorithms
- **RAII**: Follow Resource Acquisition Is Initialization principles
- **Exception Safety**: Ensure proper exception handling

### 3. Architecture Patterns
- **Interface Design**: Follow existing DAAL/oneAPI patterns
- **Memory Management**: Use smart pointers and RAII
- **Threading**: Use oneDAL threading layer, not direct primitives
- **CPU Features**: Implement CPU feature dispatching for optimizations

### 4. Testing and Validation
- **Bazel Tests**: All changes must pass Bazel validation
- **Examples**: Ensure examples build and run correctly
- **Documentation**: Update relevant documentation

## 🚀 Quick Start for AI Agents

1. **Understand the Context**: Read the relevant sub-AGENTS.md file for your task
2. **Follow Patterns**: Study existing code in similar areas
3. **Respect Standards**: Apply the coding guidelines consistently
4. **Test Thoroughly**: Ensure your changes work with the build system


### 🔄 **Cross-Repository Considerations**:
- **scikit-learn-intelex integration impact**
- **API compatibility preservation**
- **Performance consistency maintenance**

## 🔍 Key Files for Understanding

- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Detailed contribution guidelines
- **[INSTALL.md](INSTALL.md)** - Build and installation instructions
- **[MODULE.bazel](MODULE.bazel)** - Bazel module configuration
- **[.clang-format](.clang-format)** - Code formatting rules
- **[.editorconfig](.editorconfig)** - Editor configuration

## 📚 Additional Resources

- **API Documentation**: [oneDAL Developer Guide](https://uxlfoundation.github.io/oneDAL/)
- **Coding Guidelines**: [Detailed coding guide](https://uxlfoundation.github.io/oneDAL/contribution/coding_guide.html)
- **CPU Features**: [CPU feature dispatching guide](https://uxlfoundation.github.io/oneDAL/contribution/cpu_features.html)
- **Threading**: [Threading layer guide](https://uxlfoundation.github.io/oneDAL/contribution/threading.html)

---

**Note**: This file serves as the main entry point. For specific implementation details, always refer to the relevant sub-AGENTS.md file in the appropriate directory.

