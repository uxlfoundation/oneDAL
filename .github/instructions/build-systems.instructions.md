---
applyTo: ["**/makefile", "**/Makefile", "**/BUILD", "**/BUILD.bazel", "**/*.tpl.BUILD", "**/*.bazel", "**/*.bzl", ".bazelrc", "**/.bazelrc", "**/*.mk", "**/CMakeLists.txt"]
---

# Build Systems Instructions for GitHub Copilot

## Build System Priority

1. **Make** - Primary production builds, core library
2. **CMake** - End-user integration, examples
3. **Bazel** - Development, testing, new features

## 🚀 Essential Commands

### Make (Production)
```bash
# Build everything
make -f makefile daal oneapi PLAT=lnx32e

# Platform-specific builds
make -f makefile daal oneapi_c PLAT=lnx32e COMPILER=gnu
make -f makefile oneapi_c PLAT=win32e COMPILER=vc

# CPU targets
make -f makefile daal PLAT=lnx32e REQCPU="avx2 avx512"

# Backend selection
make -f makefile daal PLAT=lnx32e BACKEND_CONFIG=mkl  # Intel MKL (default)
make -f makefile daal PLAT=lnx32e BACKEND_CONFIG=ref  # Reference/OpenBLAS
```

### Bazel (Development)
```bash
# Build targets
bazel build //cpp/oneapi/dal:core
bazel build //examples/daal/cpp:cholesky

# Test targets
bazel test --config=host //cpp/oneapi/dal:tests               # CPU only
bazel test --config=dpc --device=gpu //cpp/oneapi/dal:tests   # DPC++ on GPU
```

### CMake (Integration)
There is no root `CMakeLists.txt`. CMake builds the examples against an installed release, found with `find_package(oneDAL)`:
```bash
source __release_lnx/daal/latest/env/vars.sh   # Make release tree
cd examples/oneapi/cpp
cmake -B build -S . -DONEDAL_LINK=dynamic
cmake --build build --parallel
```

## 🏗️ CPU Architecture Support

- **x86-64**: sse2, avx2, avx512 (Intel/AMD)
- **ARM64**: sve (ARM Scalable Vector Extension)
- **RISC-V**: rv64 (RISC-V 64-bit)

### CPU Dispatch Pattern
```cpp
// Runtime CPU detection and optimal code selection
template <typename algorithmFPType, CpuType cpu>
services::Status compute(/* parameters */) {
    // CPU-specific optimized implementation
}
```

## 🔧 Platform Configuration

### Linux (lnx32e)
```makefile
PLAT := lnx32e
COMPILER := icx     # Intel oneAPI C++/DPC++
COMPILER := gnu     # GCC
COMPILER := clang   # Clang
```

### Windows (win32e)
```makefile
PLAT := win32e
COMPILER := vc      # Microsoft Visual C++
COMPILER := icx     # Intel oneAPI
```

### Required Environment
```bash
# Intel oneAPI (for DPC++)
export ONEAPI_ROOT=/path/to/oneapi
export PATH=$ONEAPI_ROOT/compiler/latest/linux/bin:$PATH

# TBB (required)
export TBBROOT=/path/to/tbb

# MKL (if using MKL backend)
export MKLROOT=/path/to/mkl
```

## 🎯 Bazel Build Rules

### Algorithm Module Pattern
```python
# DAAL module
daal_module(
    name = "kmeans",
    features = [ "c++17" ],
    cpu_defines = {
        "sse2":   [ "DAAL_CPU=sse2" ],
        "avx2":   [ "DAAL_CPU=avx2" ],
        "avx512": [ "DAAL_CPU=avx512" ],
    },
)

# oneAPI module
dal_module(
    name = "kmeans",
    compile_as = ["c++", "dpc++"],  # CPU and GPU
)
```

### Test Configuration
```python
dal_test_module(
    name = "core_test",
    compile_as = ["c++", "dpc++"],
    dal_deps = [":core"],
)
```

## 🔍 Common Issues

### Build Failures
- **C++17 compliance**: Ensure GCC 7+, Clang 6+, MSVC 2017+
- **Missing dependencies**: Check TBB, MKL installation
- **CPU targets**: Verify CPU flags match target architecture

### Platform Differences
- **Windows**: Use `vc` compiler for native Windows builds
- **Linux**: Prefer `icx` for Intel optimizations, `gnu` for compatibility
- **ARM**: Use `clang` compiler, set `PLAT=lnxarm`

## 🎯 Critical Rules

- **Make**: Primary for production builds and CI/CD
- **Bazel**: Development workflow and new feature development
- **CMake**: User integration, not primary build system
- **CPU Dispatch**: Always implement multi-architecture support
- **Dependencies**: TBB required, MKL preferred for performance

## 🔗 References

- **[general.instructions.md](general.instructions.md)** - Repository overview
- **[cpp-coding-guidelines.instructions.md](cpp-coding-guidelines.instructions.md)** - C++ standards
