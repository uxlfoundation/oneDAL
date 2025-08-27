# Development Tools and Build Systems - AI Agents Context

> **Purpose**: This file provides context for AI agents working with the development tools, build systems, and configuration patterns used in oneDAL development.

## 🏗️ Build System Overview

oneDAL supports **multiple build systems** to accommodate different development workflows and platform requirements:

### Primary Build Systems
- **Bazel**: Primary build system for development and CI/CD
- **CMake**: Alternative build system for traditional workflows
- **Make**: Legacy build system for specific platforms

### Build System Priority
1. **Bazel** - Recommended for development and testing
2. **CMake** - Alternative for traditional C++ workflows
3. **Make** - Legacy support for specific platforms

## 📁 Directory Structure

```
dev/
├── bazel/                   # Bazel build system configuration
│   ├── BUILD               # Root Bazel configuration
│   ├── cc/                 # C++ build rules and configurations
│   ├── config/             # Build configurations and toolchains
│   ├── deps/               # External dependencies
│   └── toolchains/         # Compiler toolchain configurations
├── make/                    # Make-based build system
│   ├── common.mk           # Common make rules
│   ├── deps.mk             # Dependency management
│   └── compiler_definitions/ # Compiler-specific configurations
└── l0_tools/               # Level Zero development tools
```

## 🔧 Bazel Build System

### Key Characteristics
- **Primary Build System**: Used for development and CI/CD
- **Dependency Management**: Automatic dependency resolution
- **Multi-platform**: Supports Linux, Windows, macOS
- **Incremental Builds**: Fast incremental compilation

### Configuration Files
- **[MODULE.bazel](MODULE.bazel)** - Root module configuration
- **[.bazelrc](.bazelrc)** - Bazel configuration options
- **[dev/bazel/BUILD](bazel/BUILD)** - Root build configuration

### Common Commands
```bash
# Build entire project
bazel build //...

# Build specific target
bazel build //cpp/daal:daal

# Run tests
bazel test //...

# Clean build
bazel clean --expunge
```

### Build Rules
```python
# C++ library target
cc_library(
    name = "daal_core",
    srcs = glob(["src/**/*.cpp"]),
    hdrs = glob(["include/**/*.h"]),
    deps = [
        "//dev/bazel/deps:tbb",
        "//dev/bazel/deps:mkl",
    ],
    visibility = ["//visibility:public"],
)
```

## 🔧 CMake Build System

### Key Characteristics
- **Alternative Build System**: Traditional C++ workflow support
- **IDE Integration**: Good IDE and editor support
- **Package Management**: Integration with package managers
- **Cross-platform**: Native support for multiple platforms

### Configuration Files
- **[cmake/scripts/generate_config.cmake](cmake/scripts/generate_config.cmake)** - Configuration generation
- **[cmake/templates/oneDALConfig.cmake.in](cmake/templates/oneDALConfig.cmake.in)** - CMake package configuration

### Common Commands
```bash
# Configure build
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

# Build project
cmake --build build --parallel

# Install
cmake --install build
```

## 🔧 Make Build System

### Key Characteristics
- **Legacy Support**: Maintained for specific platforms
- **Simple Configuration**: Basic makefile-based builds
- **Platform Specific**: Different configurations per platform
- **Dependency Management**: Manual dependency specification

### Configuration Files
- **[makefile](makefile)** - Root makefile
- **[dev/make/common.mk](make/common.mk)** - Common make rules
- **[dev/make/deps.mk](make/deps.mk)** - Dependency management

### Common Commands
```bash
# Build project
make

# Clean build
make clean

# Install
make install
```

## 🎯 Development Guidelines

### 1. Build System Selection
- **New Development**: Use Bazel for consistency
- **Existing Workflows**: CMake if already established
- **Platform Support**: Make for specific platform requirements

### 2. Dependency Management
- **External Dependencies**: Use Bazel dependency management
- **Version Control**: Pin dependency versions in build files
- **Platform Support**: Ensure dependencies work on target platforms

### 3. Configuration Management
- **Build Configurations**: Use appropriate configuration files
- **Compiler Flags**: Follow project compiler definitions
- **Platform Detection**: Use automatic platform detection

## 🔍 Build System Patterns

### 1. Bazel Pattern
```python
# Library target
cc_library(
    name = "library_name",
    srcs = glob(["src/**/*.cpp"]),
    hdrs = glob(["include/**/*.h"]),
    deps = [
        "//path/to:dependency",
    ],
    visibility = ["//visibility:public"],
)

# Test target
cc_test(
    name = "library_test",
    srcs = glob(["test/**/*.cpp"]),
    deps = [
        ":library_name",
        "//dev/bazel/deps:gtest",
    ],
)
```

### 2. CMake Pattern
```cmake
# Library target
add_library(library_name
    src/file1.cpp
    src/file2.cpp
)

target_include_directories(library_name
    PUBLIC include
)

target_link_libraries(library_name
    dependency1
    dependency2
)
```

### 3. Make Pattern
```makefile
# Library target
LIBRARY_OBJS = $(patsubst %.cpp,%.o,$(wildcard src/*.cpp))

library_name: $(LIBRARY_OBJS)
    $(CXX) $(LDFLAGS) -o $@ $^

%.o: %.cpp
    $(CXX) $(CXXFLAGS) -c $< -o $@
```

## 🚫 Common Pitfalls to Avoid

### 1. Build System Mixing
- **Don't mix** build systems in the same build
- **Don't assume** compatibility between build systems
- **Do use** consistent build system per project

### 2. Dependency Management
- **Don't hardcode** dependency paths
- **Don't ignore** version compatibility
- **Do use** proper dependency management tools

### 3. Configuration
- **Don't assume** default configurations
- **Don't ignore** platform differences
- **Do test** on target platforms

## 🧪 Testing and Validation

### Build Validation
- **All Build Systems**: Ensure all build systems work
- **Dependencies**: Validate dependency resolution
- **Platforms**: Test on supported platforms

### CI/CD Integration
- **Bazel**: Primary CI/CD build system
- **CMake**: Alternative CI/CD option
- **Make**: Platform-specific CI/CD

## 🔧 Development Tools

### Required Tools
- **Bazel**: 5.0+ for Bazel builds
- **CMake**: 3.16+ for CMake builds
- **Make**: GNU Make 3.81+ for Make builds
- **Compilers**: GCC 7+, Clang 6+, MSVC 2017+

### Optional Tools
- **Intel oneAPI**: For SYCL development
- **Intel MKL**: For optimized math operations
- **Intel TBB**: For threading support

## 📖 Further Reading

- **[dev/bazel/AGENTS.md](bazel/AGENTS.md)** - Bazel build system details
- **[dev/make/AGENTS.md](make/AGENTS.md)** - Make build system details
- **[cpp/AGENTS.md](../cpp/AGENTS.md)** - C++ implementation context
- **[docs/AGENTS.md](../docs/AGENTS.md)** - Documentation guidelines

---

**Next Steps**: For specific build system details, refer to the appropriate sub-AGENTS.md file in the `bazel/` or `make/` directories.
