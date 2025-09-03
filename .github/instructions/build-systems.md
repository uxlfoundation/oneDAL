# Build Systems Instructions for GitHub Copilot

## Build System Priority and Usage

### **Priority Order**
1. **🔴 Make** - Primary production builds, core library compilation
2. **🟡 CMake** - End-user integration (`find_package`), examples  
3. **🟢 Bazel** - New tests, development, ongoing migration (NOT production)

### **When to Use Each System**
- **Make**: Production builds, releases, CI/CD validation
- **CMake**: User projects integrating oneDAL, example builds
- **Bazel**: Development workflow, new test development, experimental features

## 🏗️ Make Build System (Primary)

### **Key Make Variables**

#### **Platform Selection**
```bash
PLAT := lnx32e    # Linux x86-64
PLAT := win32e    # Windows x86-64  
PLAT := mac32e    # macOS x86-64
```

#### **Compiler Selection**
```bash
COMPILER := icx      # Intel oneAPI DPC++ (default)
COMPILER := gnu      # GCC/Clang
COMPILER := vc       # Microsoft Visual C++
```

#### **Backend Configuration**
```bash
BACKEND_CONFIG := mkl     # Intel MKL (default)
BACKEND_CONFIG := ref     # Reference implementation
```

### **Common Make Targets**
```bash
# Build everything
make

# Build specific components  
make daal_core
make onedal_core

# Platform-specific builds
make PLAT=lnx32e COMPILER=icx
make PLAT=win32e COMPILER=vc
```

### **CPU Dispatch System**
```makefile
# CPU-specific compilation flags
p4_OPT.icx = -march=nocona      # SSE2
avx2_OPT.icx = -march=haswell   # AVX2
skx_OPT.icx = -march=skx        # AVX-512

# CPU dispatch macros
ONEAPI.dispatcher_tag.nrh := -D__CPU_TAG__=__CPU_TAG_SSE2__
ONEAPI.dispatcher_tag.hsw := -D__CPU_TAG__=__CPU_TAG_AVX2__
ONEAPI.dispatcher_tag.skx := -D__CPU_TAG__=__CPU_TAG_AVX512__
```

## 🟢 Bazel Build System (Development)

### **Key Bazel Commands**
```bash
# Build targets
bazel build //cpp/oneapi/dal:core
bazel build //examples/daal/cpp:association_rules

# Test targets
bazel test //cpp/oneapi/dal:tests
bazel test //examples/daal/cpp:association_rules

# Configuration options
bazel build -c opt      # Optimized (default)
bazel build -c dbg      # Debug with assertions
bazel test --config=host        # C++ only (no DPC++)
bazel test --config=dpc         # DPC++ with SYCL
bazel test --config=dpc --device=gpu     # GPU only
```

### **Bazel Build Rules**
```python
# Basic module definition
dal_module(
    name = "core",
    auto = True,                    # Auto-discover sources
    dal_deps = [":dependency"],     # oneDAL dependencies
    compile_as = ["c++", "dpc++"],  # Compilation modes
)

# Test module
dal_test_module(
    name = "core_test",
    dal_deps = [":core"],
    dal_test_deps = ["@catch2//:catch2"],
)
```

## 🔧 Build System Integration

### **Required Environment Variables**
```bash
# TBB installation
export TBBROOT=/path/to/tbb

# MKL installation (if using MKL backend)
export MKLROOT=/path/to/mkl

# Intel oneAPI (for DPC++ compilation)
export ONEAPI_ROOT=/path/to/oneapi
export PATH=$ONEAPI_ROOT/compiler/latest/linux/bin:$PATH
```

### **Cross-System Dependencies**
```makefile
# MKL backend configuration
ifeq ($(BACKEND_CONFIG), mkl)
    MKL_LIBS := -lmkl_intel_lp64 -lmkl_core -lmkl_sequential
    MKL_INCLUDES := -I$(MKLROOT)/include
else ifeq ($(BACKEND_CONFIG), ref)
    REF_LIBS := -lopenblas -llapack
endif
```

## 🔄 Cross-Reference
- **[dev/AGENTS.md](../../dev/AGENTS.md)** - Development tools context
- **[dev/bazel/AGENTS.md](../../dev/bazel/AGENTS.md)** - Bazel-specific context
- **[C++ Development](cpp.md)** - C++ standards and patterns
