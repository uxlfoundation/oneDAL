# GitHub Copilot Instructions for oneDAL

This directory contains custom instructions for GitHub Copilot to understand the oneDAL repository structure, coding standards, and development patterns.

## 📁 Instruction Files Structure

- **[general.md](general.md)** - General repository context and rules
- **[cpp.md](cpp.md)** - C++ development guidelines and patterns
- **[cpp17-constraints.md](cpp17-constraints.md)** - C++17 constraints and best practices
- **[build-systems.md](build-systems.md)** - Build system configurations and rules
- **[documentation.md](documentation.md)** - Documentation standards and patterns
- **[examples.md](examples.md)** - Example code guidelines and patterns
- **[ci-workflows.md](ci-workflows.md)** - CI/CD workflows and validation

## 🎯 How to Use

These instruction files are automatically loaded by GitHub Copilot when working in the oneDAL repository. They provide context-aware suggestions based on:

- **File location** in the repository
- **File type** and content
- **Development context** (new code vs. legacy maintenance)

## 📋 Key Rules Summary

1. **Interface Selection**: Use oneAPI for new code, DAAL for legacy maintenance
2. **C++ Standards**: C++17 maximum, C++14 minimum (no C++20/23 for compatibility)
3. **Build System**: Make primary (production), CMake integration (end-users), Bazel development (testing)
4. **Code Quality**: Modern C++ patterns, RAII, smart pointers
5. **Context Awareness**: Always check file path for appropriate patterns
6. **C++17 Compliance**: Strict adherence to C++17 maximum standard
7. **Integration**: Works with scikit-learn-intelex project

## 🔗 Related Documentation

- **[AGENTS.md](../../AGENTS.md)** - Comprehensive AI agent context
- **[CONTRIBUTING.md](../../CONTRIBUTING.md)** - Contribution guidelines
- **[cpp/AGENTS.md](../../cpp/AGENTS.md)** - C++ implementation context

## 🔄 Cross-References

### For AI Agents
- **[cpp/daal/AGENTS.md](../../cpp/daal/AGENTS.md)** - Traditional DAAL interface context
- **[cpp/oneapi/AGENTS.md](../../cpp/oneapi/AGENTS.md)** - Modern oneAPI interface context
- **[dev/AGENTS.md](../../dev/AGENTS.md)** - Development tools context
- **[dev/bazel/AGENTS.md](../../dev/bazel/AGENTS.md)** - Bazel build system context
- **[dev/make/AGENTS.md](../../dev/make/AGENTS.md)** - Make build system context
- **[docs/AGENTS.md](../../docs/AGENTS.md)** - Documentation context
- **[examples/AGENTS.md](../../examples/AGENTS.md)** - Examples context
- **[samples/AGENTS.md](../../samples/AGENTS.md)** - Samples context
- **[deploy/AGENTS.md](../../deploy/AGENTS.md)** - Deployment and distribution context
- **[ci/AGENTS.md](../../ci/AGENTS.md)** - CI infrastructure and workflows context

### For Copilot Instructions
- **[Build Systems](../../.github/instructions/build-systems.md)** - Build system guidance
- **[C++ Development](../../.github/instructions/cpp.md)** - C++ coding guidelines
- **[Examples](../../.github/instructions/examples.md)** - Code pattern examples
- **[Documentation](../../.github/instructions/documentation.md)** - Documentation standards

---

**Note**: These instructions are automatically applied by GitHub Copilot. For detailed context, refer to the main AGENTS.md files in the repository.
