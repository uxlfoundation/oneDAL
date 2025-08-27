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

# General Repository Instructions for GitHub Copilot

## Repository Context

**oneDAL** (oneAPI Data Analytics Library) is a high-performance C++ library for machine learning algorithms with dual interfaces:

- **Traditional DAAL**: Legacy CPU-focused interface for backward compatibility
- **Modern oneAPI**: GPU-accelerated interface with SYCL support for new development

**Integration Note**: oneDAL works together with the [scikit-learn-intelex](https://github.com/intel/scikit-learn-intelex) project. While they are separate repositories, they share common validation aspects and work together to provide accelerated machine learning capabilities.

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
- Follow comprehensive coding guidelines

### When Working in `cpp/daal/`
- Suggest DAAL patterns and legacy compatibility
- Use C++14/17 features appropriately (but not C++20/23)
- Include appropriate DAAL headers
- Follow DAAL naming conventions
- Maintain backward compatibility
- Follow comprehensive coding guidelines


### When Working in `examples/` or `samples/`
- Ensure examples are complete and runnable
- Use appropriate interface based on subdirectory
- Include proper error handling
- Follow example patterns established in the directory
- Follow coding standards for consistency

## What NOT to Generate

- Interface mixing between DAAL and oneAPI
- Raw pointers for ownership
- Outdated C++98 patterns
- C++20/23 features (for compatibility reasons)
- Platform-specific hardcoded code
- Incomplete error handling
- Examples that don't compile or run
- Code that violates coding standards

## What TO Generate

- Modern C++17 patterns with smart pointers
- RAII-compliant resource management
- Exception-safe code
- Context-appropriate interface usage
- Proper dependency management
- Complete, runnable examples
- Proper error handling and validation
- Code that follows comprehensive coding guidelines

## 🔍 **PR Review Assistance (PRIMARY FOCUS)**

### Common PR Review Scenarios

#### **1. New Algorithm Implementation**
- [ ] **Interface Consistency**: Uses appropriate interface (oneAPI for new, DAAL for legacy)
- [ ] **Make Compatibility**: Works with Make build system (🔴 CRITICAL)
- [ ] **Bazel Testing**: Includes proper test configuration
- [ ] **C++17 Compliance**: No C++20/23 features used
- [ ] **Coding Standards**: Follows comprehensive guidelines

#### **2. Build System Changes**
- [ ] **Make Priority**: Changes work with Make FIRST (🔴 CRITICAL)
- [ ] **CMake Support**: End-user integration maintained
- [ ] **Bazel Validation**: Development workflow preserved
- [ ] **Backward Compatibility**: Existing builds not broken

#### **3. Test Additions**
- [ ] **Bazel Configuration**: Proper test setup for development
- [ ] **Make Integration**: Tests work with production builds
- [ ] **Coverage**: Adequate test coverage provided
- [ ] **Performance**: No performance regression introduced

#### **4. Documentation Updates**
- [ ] **Accuracy**: Information is technically correct
- [ ] **Completeness**: All changes documented
- [ ] **Examples**: Code examples compile and run
- [ ] **Cross-References**: Links to related documentation

#### **5. Performance Changes**
- [ ] **Make Build Performance**: Production build performance maintained
- [ ] **Runtime Performance**: Algorithm performance preserved or improved
- [ ] **Memory Usage**: Memory efficiency maintained
- [ ] **Platform Support**: Works across supported platforms

#### **6. Integration Changes**
- [ ] **scikit-learn-intelex Compatibility**: No breaking changes
- [ ] **API Consistency**: Interface consistency maintained
- [ ] **Performance Impact**: Cross-repository performance preserved
- [ ] **Documentation**: Integration changes documented

#### **7. Code Quality Issues**
- [ ] **Coding Standards**: Follows comprehensive guidelines
- [ ] **Naming Conventions**: Consistent naming patterns
- [ ] **Code Structure**: Proper declaration order
- [ ] **Documentation**: Proper comments and documentation
- [ ] **Error Handling**: Robust exception safety


## 🔄 **Cross-Reference Navigation**

### For This Area
- **[AGENTS.md](../../AGENTS.md)** - Main repository context
- **[coding-guidelines.md](coding-guidelines.md)** - Comprehensive coding standards

### For Other Areas
- **[Build Systems](build-systems.md)** - Build system guidance
- **[C++ Development](cpp.md)** - C++ coding guidelines
- **[Examples](examples.md)** - Code pattern examples
- **[Documentation](documentation.md)** - Documentation standards
- **[CI Workflows](ci-workflows.md)** - CI/CD validation guidance

---

**Remember**: 
- When in doubt, refer to the appropriate instruction file for detailed guidance
- Ensure all generated code compiles and follows the established patterns
- **Follow comprehensive coding guidelines** for consistency and quality
