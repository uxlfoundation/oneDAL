# General Repository Instructions for GitHub Copilot

## Repository Context

**oneDAL** (oneAPI Data Analytics Library) is a high-performance C++ library for machine learning algorithms with dual interfaces:

- **Traditional DAAL**: Legacy CPU-focused interface for backward compatibility
- **Modern oneAPI**: GPU-accelerated interface with SYCL support for new development

**Integration Note**: oneDAL works together with [scikit-learn-intelex](https://github.com/intel/scikit-learn-intelex). They share common validation aspects and work together to provide accelerated machine learning capabilities.

## 🎯 **PRIMARY GOAL: PR Review Assistance**

**GitHub Copilot's main purpose in this repository is to assist with PR reviews and validation.**

### 📋 **PR Review Priority Checklist**
- [ ] **🔴 CRITICAL**: C++17 compliance maintained
- [ ] **🟡 IMPORTANT**: Interface consistency preserved
- [ ] **🟡 IMPORTANT**: Cross-repository impact assessed
- [ ] **🟡 IMPORTANT**: Coding standards followed

## Critical Rules

### Interface Selection
- **NEW CODE**: Always use the **oneAPI interface** (`cpp/oneapi/`) for new development
- **LEGACY CODE**: Use **DAAL interface** (`cpp/daal/`) only when modifying existing code
- **NEVER MIX**: Don't mix DAAL and oneAPI interfaces in the same file

### C++ Standards
- **Language**: Use C++17 features when possible, C++14 minimum
- **Maximum Standard**: C++17 ONLY - do not use C++20/23 features for compatibility reasons
- **Headers**: Use `#pragma once` for oneAPI, traditional guards for DAAL
- **Smart Pointers**: Always use `std::unique_ptr` and `std::shared_ptr`
- **RAII**: Follow Resource Acquisition Is Initialization principles

### Coding Standards
- **Comprehensive Guidelines**: Follow [coding-guidelines.md](coding-guidelines.md) for all code
- **Naming Conventions**: Use consistent naming patterns
- **Code Structure**: Follow proper declaration order and organization
- **Documentation**: Include proper comments and documentation

## Context-Aware Behavior

### When Working in `cpp/oneapi/`
- Suggest oneAPI patterns and SYCL integration
- Use modern C++17 features (but not C++20/23)
- Include appropriate oneAPI headers
- Follow oneAPI naming conventions
- Suggest GPU-accelerated patterns when appropriate

### When Working in `cpp/daal/`
- Suggest DAAL patterns and legacy compatibility
- Use C++14/17 features appropriately (but not C++20/23)
- Include appropriate DAAL headers
- Follow DAAL naming conventions
- Maintain backward compatibility

### When Working in `examples/` or `samples/`
- Ensure examples are complete and runnable
- Use appropriate interface based on subdirectory
- Include proper error handling
- Follow example patterns established in the directory

## What NOT to Generate

- Interface mixing between DAAL and oneAPI
- Raw pointers for ownership
- Outdated C++98 patterns
- C++20/23 features (for compatibility reasons)
- Platform-specific hardcoded code
- Incomplete error handling
- Examples that don't compile or run

## What TO Generate

- Modern C++17 patterns with smart pointers
- RAII-compliant resource management
- Exception-safe code
- Context-appropriate interface usage
- Proper dependency management
- Complete, runnable examples
- Proper error handling and validation

## 🔍 **PR Review Assistance (PRIMARY FOCUS)**

### Common PR Review Scenarios

#### **1. New Algorithm Implementation**
- [ ] **Interface Consistency**: Uses appropriate interface (oneAPI for new, DAAL for legacy)
- [ ] **Make Compatibility**: Works with Make build system (🔴 CRITICAL)
- [ ] **Bazel Testing**: Includes proper test configuration
- [ ] **C++17 Compliance**: No C++20/23 features used
- [ ] **Coding Standards**: Follows comprehensive guidelines

#### **2. Build System Changes**
- [ ] **Make Priority**: Make build system takes priority over others
- [ ] **Cross-Platform**: Changes work on Linux, Windows, macOS
- [ ] **Dependency Management**: Proper dependency handling

#### **3. Interface Changes**
- [ ] **Backward Compatibility**: DAAL interface remains unchanged
- [ ] **scikit-learn-intelex Impact**: Consider impact on integration
- [ ] **API Consistency**: New APIs follow established patterns

## Cross-Reference
- **[coding-guidelines.md](coding-guidelines.md)** - Comprehensive coding standards
- **[build-systems.md](build-systems.md)** - Build system guidance
- **[examples.md](examples.md)** - Example patterns