# Copyright contributors to the oneDAL project
#
<!--
  ~ Licensed under the Apache License, Version 2.0 (the "License");
  ~ you may not use this file except in compliance with the License.
  ~ You may obtain a copy of the License at
  ~
  ~     http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS,
  ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ~ See the License for the specific language governing permissions and
  ~ limitations under the License.
-->

# Build Systems Instructions for GitHub Copilot

## Build System Overview

oneDAL supports multiple build systems with different priorities:

- **🔴 Make**: Primary build system for production builds
- **🟡 CMake**: End-user integration support (find_package)
- **🟢 Bazel**: Development and testing (ongoing migration)

## C++ Standard Constraints

**IMPORTANT**: For compatibility reasons, this project is limited to **C++17 maximum**:
- **Do NOT use C++20/23 features** in any build configuration
- **Use C++17 features** when possible for modern development
- **Ensure all build systems** enforce this constraint

### 🔴 **CRITICAL IMPORTANCE**
- **Production Builds**: All releases use Make builds
- **Platform Support**: All supported platforms and configurations
- **Performance**: Optimized production builds
- **Stability**: Proven, reliable build system

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

### 🔴 **Make Build Validation (CRITICAL FOR PR REVIEWS)**
```makefile
# Production build validation
.PHONY: validate-production
validate-production:
	@echo "🔴 Validating Make production build..."
	$(MAKE) clean
	$(MAKE) all
	@echo "✅ Make production build successful"
```

## 🟡 CMake support (End-User Integration)

### 🟡 **IMPORTANT FOR END-USERS**
- **End-User Integration**: Projects using oneDAL via find_package
- **IDE Integration**: Better IDE and editor support
- **Package Management**: Integration with package managers
- **Cross-Platform**: Consistent build experience across platforms

### Find Packages
```cmake
# Find oneDAL
find_package(oneDAL REQUIRED)


## 🟢 Bazel Build System (Development and Testing)

### 🟢 **DEVELOPMENT ONLY - NOT FOR PRODUCTION**
- **Development**: New feature development and testing
- **Testing**: Comprehensive test suites and validation
- **CI/CD**: Automated testing and development workflows
- **Dependencies**: Complex dependency management for development

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

### When to Use Make (🔴 CRITICAL)
- **🔴 Production Builds**: Primary build system for product releases
- **🔴 Platform Support**: All supported platforms and configurations
- **🔴 Performance**: Optimized production builds
- **🔴 Stability**: Proven, reliable build system
- **🔴 PR Review Priority**: Verify Make compatibility FIRST

### When to Use Bazel (🟢 DEVELOPMENT)
- **🟢 Development**: New feature development and testing
- **🟢 Testing**: Comprehensive test suites and validation
- **🟢 CI/CD**: Automated testing and development workflows
- **🟢 Dependencies**: Complex dependency management for development
- **🟢 PR Review Priority**: Verify Bazel compatibility THIRD

## Common Build Patterns


## Best Practices

1. **Consistency**: Use the same build system throughout a project
2. **Dependencies**: Manage dependencies properly with version pinning
3. **Platform Support**: Ensure builds work on all target platforms
4. **Performance**: Use appropriate optimization flags for release builds
5. **Testing**: Include comprehensive testing in build configurations
