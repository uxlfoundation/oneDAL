# Build Systems Instructions for GitHub Copilot

## Build System Overview

oneDAL supports multiple build systems with different priorities:

- **Make**: Primary build system for production builds
- **CMake**: End-user integration support (find_package)
- **Bazel**: Development and testing (ongoing migration)

## C++ Standard Constraints

**IMPORTANT**: For compatibility reasons, this project is limited to **C++17 maximum**:
- **Do NOT use C++20/23 features** in any build configuration
- **Use C++17 features** when possible for modern development
- **Fall back to C++14** for broader compiler support when needed
- **Ensure all build systems** enforce this constraint

## Bazel Build System (Primary)

### Build Target Patterns
```python
# C++ Library Target
cc_library(
    name = "library_name",
    srcs = glob(["src/**/*.cpp"]),
    hdrs = glob(["include/**/*.h"]),
    deps = [
        "//path/to:dependency",
        "//dev/bazel/deps:external_lib",
    ],
    visibility = ["//visibility:public"],
    copts = [
        "-std=c++17",  # Maximum C++ standard allowed for compatibility
        "-O3",
        "-march=native",
    ],
)

# C++ Test Target
cc_test(
    name = "library_test",
    srcs = glob(["test/**/*.cpp"]),
    deps = [
        ":library_name",
        "//dev/bazel/deps:catch2",
    ],
    copts = [
        "-std=c++17",
        "-g",
    ],
)

# C++ Binary Target
cc_binary(
    name = "example_binary",
    srcs = glob(["examples/**/*.cpp"]),
    deps = [
        ":library_name",
        "//dev/bazel/deps:external_lib",
    ],
    copts = [
        "-std=c++17",
        "-O2",
    ],
)
```

### Dependency Management
```python
# External Dependencies
cc_library(
    name = "tbb",
    srcs = glob(["tbb/src/**/*.cpp"]),
    hdrs = glob(["tbb/include/**/*.h"]),
    visibility = ["//visibility:public"],
)

# Dependency Rules
def ccl_deps():
    native.new_local_repository(
        name = "ccl",
        path = "path/to/ccl",
        build_file = "//dev/bazel/deps:ccl.BUILD",
    )
```

### Conditional Compilation
```python
cc_library(
    name = "platform_specific",
    srcs = select({
        "//dev/bazel/config:linux": ["src/linux.cpp"],
        "//dev/bazel/config:windows": ["src/windows.cpp"],
        "//conditions:default": ["src/default.cpp"],
    }),
    deps = [":common"],
)
```

### Feature Detection
```python
cc_library(
    name = "feature_detection",
    srcs = ["src/feature_detection.cpp"],
    deps = [":common"],
    copts = select({
        "//dev/bazel/config:avx512": ["-mavx512f", "-mavx512dq"],
        "//dev/bazel/config:avx2": ["-mavx2"],
        "//conditions:default": [],
    }),
)
```

## CMake Build System (Alternative)

### Library Targets
```cmake
# Library target
add_library(library_name
    src/file1.cpp
    src/file2.cpp
)

# Include directories
target_include_directories(library_name
    PUBLIC include
)

# Link libraries
target_link_libraries(library_name
    dependency1
    dependency2
)

# Compiler options
target_compile_features(library_name PRIVATE cxx_std_17)  # Maximum C++ standard allowed for compatibility
target_compile_options(library_name PRIVATE -O3)
```

### Find Packages
```cmake
# Find oneDAL
find_package(oneDAL REQUIRED)

# Find other dependencies
find_package(Threads REQUIRED)
find_package(OpenMP)

# Link with found packages
target_link_libraries(example
    oneDAL::oneDAL
    Threads::Threads
    $<$<BOOL:OpenMP_CXX_FOUND>:OpenMP::OpenMP_CXX>
)
```

### Platform Detection
```cmake
# Platform-specific settings
if(WIN32)
    target_compile_definitions(library_name PRIVATE WIN32_LEAN_AND_MEAN)
elseif(UNIX)
    target_compile_options(library_name PRIVATE -fPIC)
endif()

# Compiler-specific settings
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(library_name PRIVATE -march=native)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(library_name PRIVATE -march=native)
endif()
```

## Make Build System (Legacy)

### Basic Patterns
```makefile
# Variables
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra  # Maximum C++ standard allowed for compatibility
LDFLAGS = -L/usr/local/lib
LIBS = -lonedal -lpthread

# Source files
SOURCES = $(wildcard src/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = program

# Main target
$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

# Object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean
clean:
	rm -f $(OBJECTS) $(TARGET)
```

### Dependency Management
```makefile
# Include directories
INCLUDES = -Iinclude -I/usr/local/include

# Library paths
LIBPATHS = -L/usr/local/lib -L/opt/intel/oneapi/lib

# Dependencies
DEPS = -lonedal -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

# Compilation
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Linking
$(TARGET): $(OBJECTS)
	$(CXX) $(LIBPATHS) -o $@ $^ $(DEPS)
```

## Build System Selection Guidelines

### When to Use Make
- **Production Builds**: Primary build system for product releases
- **Platform Support**: All supported platforms and configurations
- **Performance**: Optimized production builds
- **Stability**: Proven, reliable build system

### When to Use CMake
- **End-User Integration**: Projects using oneDAL via find_package
- **IDE Integration**: Better IDE and editor support
- **Package Management**: Integration with package managers
- **Cross-Platform**: Consistent build experience across platforms

### When to Use Bazel
- **Development**: New feature development and testing
- **Testing**: Comprehensive test suites and validation
- **CI/CD**: Automated testing and development workflows
- **Dependencies**: Complex dependency management for development

## Common Build Patterns

### Testing Configuration
```python
# Bazel testing
cc_test(
    name = "comprehensive_test",
    srcs = glob(["test/**/*.cpp"]),
    deps = [
        ":library_name",
        "//dev/bazel/deps:catch2",
        "//dev/bazel/deps:gtest",
    ],
    size = "large",
    timeout = "long",
)
```

### Performance Optimization
```python
# Bazel performance flags
cc_library(
    name = "optimized_lib",
    copts = [
        "-O3",
        "-march=native",
        "-ffast-math",
        "-funroll-loops",
    ],
)
```

### Debug Configuration
```python
# Bazel debug configuration
cc_library(
    name = "debug_lib",
    copts = [
        "-g",
        "-O0",
        "-DDEBUG",
        "-Wall",
        "-Wextra",
    ],
)
```

## Build System Integration

### Cross-Platform Support
```python
# Bazel platform detection
cc_library(
    name = "cross_platform",
    srcs = select({
        "//dev/bazel/config:linux": ["src/linux.cpp"],
        "//dev/bazel/config:windows": ["src/windows.cpp"],
        "//dev/bazel/config:macos": ["src/macos.cpp"],
        "//conditions:default": ["src/default.cpp"],
    }),
)
```

### Dependency Versioning
```python
# Bazel dependency versions
bazel_dep(name = "rules_cc", version = "0.0.9")
bazel_dep(name = "catch2", version = "3.4.0")
bazel_dep(name = "gtest", version = "1.14.0")
```

## Best Practices

1. **Consistency**: Use the same build system throughout a project
2. **Dependencies**: Manage dependencies properly with version pinning
3. **Platform Support**: Ensure builds work on all target platforms
4. **Performance**: Use appropriate optimization flags for release builds
5. **Testing**: Include comprehensive testing in build configurations

## Common Pitfalls

1. **Build System Mixing**: Don't mix build systems in the same project
2. **Hardcoded Paths**: Avoid hardcoded platform-specific paths
3. **Dependency Conflicts**: Resolve dependency version conflicts
4. **Platform Assumptions**: Don't assume tool availability
5. **Build Performance**: Monitor and optimize build performance

---

**Note**: Always use the build system that best fits your project's needs. Bazel is recommended for new development, but existing projects may benefit from their current build system.
