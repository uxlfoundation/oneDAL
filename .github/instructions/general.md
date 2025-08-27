# General Repository Instructions for GitHub Copilot

## Repository Context

**oneDAL** (oneAPI Data Analytics Library) is a high-performance C++ library for machine learning algorithms with dual interfaces:

- **Traditional DAAL**: Legacy CPU-focused interface for backward compatibility
- **Modern oneAPI**: GPU-accelerated interface with SYCL support for new development

**Integration Note**: oneDAL works together with the [scikit-learn-intelex](https://github.com/intel/scikit-learn-intelex) project. While they are separate repositories, they share common validation aspects and work together to provide accelerated machine learning capabilities.

## 🎯 **PRIMARY GOAL: PR Review Assistance**

**GitHub Copilot's main purpose in this repository is to assist with PR reviews and validation.**

### 📋 **PR Review Priority Checklist**
- [ ] **🔴 CRITICAL**: Make build compatibility (production builds)
- [ ] **🟡 IMPORTANT**: CMake integration (end-user support)  
- [ ] **🟢 DEVELOPMENT**: Bazel tests (development workflow)
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

### 🔴 **Build System Priority (CRITICAL FOR PR REVIEWS)**

#### **1. Make Build System (PRIMARY for Production)**
- **🔴 CRITICAL**: All changes MUST work with Make builds
- **Purpose**: Production builds and releases
- **Priority**: HIGHEST - verify Make compatibility FIRST
- **Location**: `dev/make/` directory

#### **2. CMake Build System (End-User Integration)**
- **🟡 IMPORTANT**: Changes should support CMake integration
- **Purpose**: End-user projects using `find_package(oneDAL)`
- **Priority**: HIGH - verify CMake compatibility SECOND
- **Location**: `cmake/` directory

#### **3. Bazel Build System (Development Only)**
- **🟢 DEVELOPMENT**: Changes should work with Bazel for testing
- **Purpose**: Development workflow and CI/CD testing
- **Priority**: MEDIUM - verify Bazel compatibility THIRD
- **Location**: `dev/bazel/` directory

**🚨 WHY MAKE FIRST?** Make is the production build system used for releases. CMake is for end-user integration. Bazel is for development/testing only.

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

### When Working in `dev/make/`
- 🔴 **CRITICAL**: Suggest Make build patterns for PRODUCTION
- Use Make syntax for build files
- Follow Make naming conventions
- Include proper dependencies and targets
- **Remember**: This is the PRIMARY build system

### When Working in `dev/bazel/`
- 🟢 **DEVELOPMENT ONLY**: Suggest Bazel build rules for development/testing
- Use Python syntax for BUILD files
- Follow Bazel naming conventions
- Include proper dependencies
- **Note**: This is for development, NOT production builds

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
- [ ] **CMake Integration**: Supports end-user integration
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

### Review Checklist Template

```markdown
## PR Review Checklist

### 🔴 Build System Validation (CRITICAL)
- [ ] **Make build succeeds** (production validation)
- [ ] **CMake integration works** (end-user support)
- [ ] **Bazel tests pass** (development validation)

### 🟡 Code Quality
- [ ] **C++17 compliance maintained** (no C++20/23)
- [ ] **Interface consistency preserved** (DAAL vs oneAPI)
- [ ] **Error handling implemented** (proper exception safety)
- [ ] **Documentation updated** (accurate and complete)
- [ ] **Coding standards followed** (comprehensive guidelines)

### 🟡 Cross-Repository Impact
- [ ] **scikit-learn-intelex compatibility** assessed
- [ ] **API changes documented** for integration
- [ ] **Performance impact** evaluated
- [ ] **Breaking changes** clearly identified

### 🟢 Development Workflow
- [ ] **Examples build and run** correctly
- [ ] **Tests provide adequate coverage**
- [ ] **Code follows project patterns**
- [ ] **No platform-specific hardcoding**
```

## File Path Context Detection

### oneAPI Context (`cpp/oneapi/`, `examples/oneapi/`)
```cpp
// Use oneAPI patterns
#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/table/homogen.hpp"

auto desc = kmeans::descriptor<float>()
    .set_cluster_count(10);
```

### DAAL Context (`cpp/daal/`, `examples/daal/`)
```cpp
// Use DAAL patterns
#include "algorithms/kmeans/kmeans_batch.h"
#include "data_management/data/homogen_numeric_table.h"

auto training = new kmeans_batch<float>();
```

### Build System Context (`dev/bazel/`, `cmake/`)
```python
# Bazel context (DEVELOPMENT ONLY)
cc_library(
    name = "library_name",
    srcs = glob(["src/**/*.cpp"]),
    deps = [":dependency"],
)
```

## Quick Reference

### For New Development
1. Use **oneAPI interface** (`cpp/oneapi/`)
2. 🔴 **Use Make build system** (`dev/make/`) for production builds
3. 🟢 Use **Bazel build system** (`dev/bazel/`) for development and testing
4. Use **C++17 features** when possible (but not C++20/23)
5. Follow **modern C++ patterns**
6. Follow **comprehensive coding guidelines**

### For Legacy Maintenance
1. Use **DAAL interface** (`cpp/daal/`)
2. Maintain **backward compatibility**
3. Use **C++14/17 features** appropriately (but not C++20/23)
4. Follow **existing patterns**
5. Follow **comprehensive coding guidelines**

### For Examples and Documentation
1. Ensure **completeness** and **runnability**
2. Use **appropriate interface** based on context
3. Include **proper error handling**
4. Follow **established patterns**
5. Follow **coding standards**

## 🚨 Critical Reminders for PR Review

1. **🔴 Make compatibility is CRITICAL** - verify FIRST
2. **🟡 CMake integration is IMPORTANT** - verify SECOND  
3. **🟢 Bazel testing is DEVELOPMENT** - verify THIRD
4. **C++17 maximum standard** - no C++20/23 features
5. **Interface consistency** - don't mix DAAL and oneAPI
6. **Cross-repository impact** - consider scikit-learn-intelex
7. **Coding standards** - follow comprehensive guidelines

## 🔄 **Cross-Reference Navigation**

### For This Area
- **[AGENTS.md](../../AGENTS.md)** - Main repository context
- **[coding-guidelines.md](coding-guidelines.md)** - Comprehensive coding standards

### For Other Areas
- **[Build Systems](../../.github/instructions/build-systems.md)** - Build system guidance
- **[C++ Development](../../.github/instructions/cpp.md)** - C++ coding guidelines
- **[Examples](../../.github/instructions/examples.md)** - Code pattern examples
- **[Documentation](../../.github/instructions/documentation.md)** - Documentation standards
- **[CI Workflows](../../.github/instructions/ci-workflows.md)** - CI/CD validation guidance

---

**Remember**: 
- Always check the file path and context before generating code
- C++17 is the maximum standard allowed for compatibility reasons
- When in doubt, refer to the appropriate instruction file for detailed guidance
- Ensure all generated code compiles and follows the established patterns
- **PR Review is the PRIMARY goal** - focus on validation and quality
- **Follow comprehensive coding guidelines** for consistency and quality
