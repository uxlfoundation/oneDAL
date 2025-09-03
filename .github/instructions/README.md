# GitHub Copilot Instructions for oneDAL

## 🎯 **PRIMARY GOAL: PR Review Assistance**

**GitHub Copilot's main purpose in this repository is to assist with PR reviews and validation.**

This directory contains custom instructions for GitHub Copilot to understand oneDAL repository structure, coding standards, and development patterns.

## 📁 Instruction Files

- **[general.md](general.md)** - Repository context and core rules
- **[cpp.md](cpp.md)** - C++ development patterns
- **[coding-guidelines.md](coding-guidelines.md)** - Coding standards and best practices
- **[build-systems.md](build-systems.md)** - Build system configurations
- **[documentation.md](documentation.md)** - Documentation standards
- **[examples.md](examples.md)** - Example code guidelines
- **[ci-workflows.md](ci-workflows.md)** - CI/CD workflows

## 🎯 Usage

These instructions provide context-aware suggestions based on:
- **File location** and repository context
- **File type** and development patterns
- **PR Review scenarios** (primary focus)

## 📋 Key Rules

1. **🔴 Build Priority**: Make (production), CMake (integration), Bazel (development)
2. **Interface Selection**: oneAPI for new code, DAAL for legacy
3. **C++ Standards**: C++17 maximum, C++14 minimum
4. **Code Quality**: RAII, smart pointers, exception safety
5. **🎯 PR Focus**: Primary goal is PR review assistance

## 🔍 PR Review Scenarios

- **Algorithm Implementation**: Check interface consistency, Make compatibility
- **Build Changes**: Verify Make compatibility first (🔴 CRITICAL)
- **Performance Changes**: Validate Make build impact
- **Integration**: Check scikit-learn-intelex compatibility

## 🔗 Cross-References

### AI Context Files
- **[AGENTS.md](../../AGENTS.md)** - Main repository context
- **[cpp/AGENTS.md](../../cpp/AGENTS.md)** - C++ implementation context
- **[dev/AGENTS.md](../../dev/AGENTS.md)** - Development tools context

### Copilot Instructions
- **[Build Systems](build-systems.md)** - Build guidance
- **[C++ Development](cpp.md)** - C++ patterns
- **[Examples](examples.md)** - Code patterns