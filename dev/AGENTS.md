
# Development Tools and Build Systems - AI Agents Context

> **Purpose**: Context for AI agents working with oneDAL's sophisticated build systems, Bazel rules, and development tools.

## 🏗️ Build System Architecture

oneDAL uses **dual build systems** optimized for different workflows:

### Build Systems
- **Bazel**: Modern development build system with sophisticated C++ template instantiation
- **Make**: Production build system with platform-specific optimizations

### Development Tools
- **Level Zero (L0) Tools**: GPU development utilities (`dev/l0_tools/`)
- **Docker**: Containerized development environment (`dev/docker/`)
- **Dependency Management**: Automated TBB, MKL, SYCL integration

## 📁 Structure
```
dev/
├── bazel/                  # Bazel build system with custom rules
│   ├── dal.bzl            # oneAPI interface build rules
│   ├── daal.bzl           # DAAL interface build rules  
│   ├── cc/                # C++ compilation rules
│   ├── deps/              # External dependency management
│   └── config/            # Build configuration
├── make/                   # Make-based build system
│   ├── common.mk          # Common make patterns
│   ├── deps.mk            # Dependency resolution
│   └── compiler_definitions/ # Compiler-specific settings
├── l0_tools/              # Level Zero GPU development tools
└── docker/                # Development containers
```

## 🔧 Bazel Build System

### Key Characteristics
- **Development Build System**: Used for development and CI/CD
- **Dependency Management**: Automatic dependency resolution
- **Multi-platform**: Supports Linux
- **Incremental Builds**: Fast incremental compilation

### Configuration Files
- **[MODULE.bazel](MODULE.bazel)** - Root module configuration
- **[.bazelrc](.bazelrc)** - Bazel configuration options
- **[dev/bazel/BUILD](bazel/BUILD)** - Root build configuration

### Common Commands
```bash
# Build entire project
bazel build //...

# Run tests
bazel test //...

# Clean build
bazel clean --expunge
```


## 🔧 Make Build System

### Key Characteristics
- **Production Build System**: Main build system for production builds
- **Platform Specific**: Different configurations per platform
- **Dependency Management**: Manual dependency specification

### Configuration Files
- **[makefile](makefile)** - Root makefile
- **[dev/make/common.mk](make/common.mk)** - Common make rules
- **[dev/make/deps.mk](make/deps.mk)** - Dependency management



## 🔍 Build System Patterns

### Bazel Pattern
```python
cc_library(
    name = "library_name",
    srcs = glob(["src/**/*.cpp"]),
    hdrs = glob(["include/**/*.h"]),
    deps = ["//path/to:dependency"],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "library_test",
    srcs = glob(["test/**/*.cpp"]),
    deps = [":library_name", "//dev/bazel/deps:gtest"],
)
```

### Make Pattern
```makefile
LIBRARY_OBJS = $(patsubst %.cpp,%.o,$(wildcard src/*.cpp))

library_name: $(LIBRARY_OBJS)
    $(CXX) $(LDFLAGS) -o $@ $^

%.o: %.cpp
    $(CXX) $(CXXFLAGS) -c $< -o $@
```

## 🚫 Common Pitfalls
- **Build System Mixing**: Don't mix build systems, use consistent approach per project
- **Dependency Management**: Don't hardcode paths, use proper dependency tools
- **Configuration**: Don't assume defaults, test on target platforms

## 🧪 Testing and Validation
- **Build Validation**: Ensure all build systems work
- **Dependencies**: Validate dependency resolution
- **Platforms**: Test on supported platforms

## 🔧 Required Tools
- **Bazel**: 5.0+ for Bazel builds
- **Make**: GNU Make 3.81+ for Make builds  
- **Compilers**: GCC 7+, Clang 6+, MSVC 2017+
- **Intel oneAPI**: For SYCL development
- **Intel MKL**: For optimized math operations
- **Intel TBB**: For threading support

## 📖 Further Reading
- **[dev/bazel/AGENTS.md](bazel/AGENTS.md)** - Bazel build system details
- **[cpp/AGENTS.md](../cpp/AGENTS.md)** - C++ implementation context
- **[docs/AGENTS.md](../docs/AGENTS.md)** - Documentation guidelines
