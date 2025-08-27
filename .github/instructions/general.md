# General Repository Instructions for GitHub Copilot

## Repository Context

**oneDAL** (oneAPI Data Analytics Library) is a high-performance C++ library for machine learning algorithms with dual interfaces:

- **Traditional DAAL**: Legacy CPU-focused interface for backward compatibility
- **Modern oneAPI**: GPU-accelerated interface with SYCL support for new development

**Integration Note**: oneDAL works together with the [scikit-learn-intelex](https://github.com/intel/scikit-learn-intelex) project. While they are separate repositories, they share common validation aspects and work together to provide accelerated machine learning capabilities.

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

### Build System Priority
- **Primary**: Use Make build system (`dev/make/`) for production builds
- **Integration**: Use CMake for end-user integration (find_package support)
- **Development**: Use Bazel build system (`dev/bazel/`) for new tests and development (ongoing migration)

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

### When Working in `dev/make/`
- Suggest Make build patterns and configurations
- Use Make syntax for build files
- Follow Make naming conventions
- Include proper dependencies and targets

### When Working in `dev/bazel/`
- Suggest Bazel build rules and configurations for development/testing
- Use Python syntax for BUILD files
- Follow Bazel naming conventions
- Include proper dependencies
- Note: This is for development, not production builds

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

## PR Review Assistance

### Common PR Review Scenarios
- **New Algorithm Implementation**: Check interface consistency and Make compatibility
- **Build System Changes**: Verify Make compatibility first, then CMake integration
- **Test Additions**: Validate Bazel test configuration for development
- **Documentation Updates**: Ensure accuracy and completeness
- **Performance Changes**: Verify Make build performance impact
- **Integration Changes**: Check scikit-learn-intelex compatibility

### Review Checklist
- [ ] Builds successfully with Make (production)
- [ ] CMake integration works (end-user support)
- [ ] Bazel tests pass (development validation)
- [ ] C++17 compliance maintained
- [ ] Interface consistency preserved
- [ ] Documentation updated
- [ ] Cross-repository considerations addressed

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
# Bazel context
cc_library(
    name = "library_name",
    srcs = glob(["src/**/*.cpp"]),
    deps = [":dependency"],
)
```

## Quick Reference

### For New Development
1. Use **oneAPI interface** (`cpp/oneapi/`)
2. Use **Make build system** (`dev/make/`) for production builds
3. Use **Bazel build system** (`dev/bazel/`) for development and testing
4. Use **C++17 features** when possible (but not C++20/23)
5. Follow **modern C++ patterns**

### For Legacy Maintenance
1. Use **DAAL interface** (`cpp/daal/`)
2. Maintain **backward compatibility**
3. Use **C++14/17 features** appropriately (but not C++20/23)
4. Follow **existing patterns**

### For Examples and Documentation
1. Ensure **completeness** and **runnability**
2. Use **appropriate interface** based on context
3. Include **proper error handling**
4. Follow **established patterns**

---

**Remember**: 
- Always check the file path and context before generating code
- C++17 is the maximum standard allowed for compatibility reasons
- When in doubt, refer to the appropriate instruction file for detailed guidance
- Ensure all generated code compiles and follows the established patterns
