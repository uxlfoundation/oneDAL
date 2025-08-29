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

### **High-Level Structure**

```
dev/make/
├── common.mk              # Core build macros and functions
├── deps.mk                # Dependency management and incremental builds
├── deps.mkl.mk            # MKL-specific dependencies
├── deps.ref.mk            # Reference backend dependencies
├── compiler_definitions/  # Compiler-specific configurations
│   ├── gnu.mk            # GCC/Clang configurations
│   ├── icc.mk            # Intel C++ Compiler
│   ├── icx.mk            # Intel oneAPI DPC++ Compiler
│   ├── dpcpp.mk          # DPC++ specific settings
│   └── vc.mk             # Microsoft Visual C++
└── function_definitions/  # Platform and architecture configurations
    ├── 32e.mk            # x86-64 architecture
    ├── lnx32e.mk         # Linux x86-64
    ├── win32e.mk         # Windows x86-64
    ├── mac32e.mk         # macOS x86-64
    ├── arm.mk            # ARM architecture
    └── riscv64.mk        # RISC-V architecture
```

### **Key Make Variables and Options**

#### **Platform Selection**
```bash
# Platform detection (auto) or manual specification
PLAT := lnx32e    # Linux x86-64
PLAT := win32e    # Windows x86-64  
PLAT := mac32e    # macOS x86-64
PLAT := lnxarm    # Linux ARM
PLAT := lnxriscv64 # Linux RISC-V
```

#### **Compiler Selection**
```bash
# Supported compilers
COMPILER := icx      # Intel oneAPI DPC++ (default)
COMPILER := icc      # Intel C++ Compiler
COMPILER := gnu      # GCC/Clang
COMPILER := clang    # Clang
COMPILER := vc       # Microsoft Visual C++
COMPILER := dpcpp    # DPC++ specific
```

#### **CPU Architecture Support**
```bash
# Specific CPU selection
REQCPU := avx2 avx512    # Build only for specific ISAs
```

#### **Backend Configuration**
```bash
# Math backend
BACKEND_CONFIG := mkl     # Intel MKL (default)
BACKEND_CONFIG := ref     # Reference implementation

# RNG backend
RNG_BACKEND := mkl       # MKL RNG
RNG_BACKEND := openrng   # Open source RNG
```

### **Common Make Targets**

#### **Production Builds**
```bash
# Build everything
make

# Build specific components
make daal_core
make onedal_core
```

#### **Platform-Specific Builds**
```bash
# Linux x86-64 with Intel compiler
make PLAT=lnx32e COMPILER=icx

# Windows x86-64 with MSVC
make PLAT=win32e COMPILER=vc MSVC_RUNTIME_VERSION=release
```
### **Make System Patterns**

#### **1. Incremental Building Support**
- **Dependency Generation**: Automatic source dependency tracking
- **Command Line Monitoring**: Rebuilds only when build commands change

#### **2. CPU Dispatch System**
```makefile
# CPU-specific compilation flags
p4_OPT.icx = -march=nocona      # SSE2
mc3_OPT.icx = -march=nehalem    # SSE4.2  
avx2_OPT.icx = -march=haswell   # AVX2
skx_OPT.icx = -march=skx        # AVX-512

# CPU dispatch macros
ONEAPI.dispatcher_tag.nrh := -D__CPU_TAG__=__CPU_TAG_SSE2__
ONEAPI.dispatcher_tag.neh := -D__CPU_TAG__=__CPU_TAG_SSE42__
ONEAPI.dispatcher_tag.hsw := -D__CPU_TAG__=__CPU_TAG_AVX2__
ONEAPI.dispatcher_tag.skx := -D__CPU_TAG__=__CPU_TAG_AVX512__
```

#### **3. Security and Compliance**
```makefile
# Security flags (Intel SDL compliance)
secure.opts.win = -GS
secure.opts.lnx = -Wformat -Wformat-security -fstack-protector-strong
secure.opts.mac = -Wformat -Wformat-security -O2 -D_FORTIFY_SOURCE=2

# Linker security
secure.opts.link.win = -DYNAMICBASE -NXCOMPAT
secure.opts.link.lnx = -z relro -z now -z noexecstack
```

## 🟢 Bazel Build System (Development)

### **High-Level Structure**

```
dev/bazel/
├── dal.bzl              # Main oneDAL build rules
├── cc.bzl               # C++ compilation rules
├── utils.bzl            # Utility functions
├── flags.bzl            # Compiler flag definitions
├── release.bzl          # Release configuration
├── config/              # Build configuration
│   └── config.bzl       # CPU and feature flags
├── deps/                # External dependencies
├── cc/                  # C++ toolchain configuration
└── toolchains/          # Platform toolchains
```

### **Key Bazel Commands**

#### **Basic Commands**
```bash
# Build targets
bazel build //cpp/oneapi/dal:core
bazel build //examples/daal/cpp:association_rules

# Run targets
bazel run //cpp/oneapi/dal:core_test
bazel run //examples/daal/cpp:association_rules

# Test targets
bazel test //cpp/oneapi/dal:tests
bazel test //examples/daal/cpp:association_rules
```

#### **Configuration Options**
```bash
# Compilation mode
bazel build -c opt      # Optimized (default)
bazel build -c dbg      # Debug with assertions
bazel build -c fastbuild # Fast build, no optimization

# Interface configuration
bazel test --config=host        # C++ only (no DPC++)
bazel test --config=dpc         # DPC++ with SYCL
bazel test --config=public      # Public interfaces only
bazel test --config=private     # Private interfaces only

# Device selection (for DPC++)
bazel test --config=dpc --device=auto    # Auto-detect
bazel test --config=dpc --device=cpu     # CPU only
bazel test --config=dpc --device=gpu     # GPU only

# CPU instruction sets
bazel test --cpu=auto           # Auto-detect (default)
bazel test --cpu=modern         # SSE2, AVX2, AVX-512
bazel test --cpu=avx2,avx512   # Specific ISAs
```

### **Bazel Build Rules**

#### **1. dal_module Rule**
```python
# Basic module definition
dal_module(
    name = "core",
    auto = True,                    # Auto-discover sources
    hdrs = ["header.hpp"],          # Additional headers
    srcs = ["source.cpp"],          # Additional sources
    dal_deps = [":dependency"],     # oneDAL dependencies
    extra_deps = ["@external//:lib"], # External dependencies
    compile_as = ["c++", "dpc++"],  # Compilation modes
)

# Test module
dal_test_module(
    name = "core_test",
    dal_deps = [":core"],
    dal_test_deps = ["@catch2//:catch2"],
)
```

#### **2. dal_example_suite Rule**
```python
# Example suite with data dependencies
dal_example_suite(
    name = "kmeans_example",
    compile_as = ["c++"],
    srcs = glob(["source/*.cpp"]),
    dal_deps = [
        "@onedal//cpp/daal/src/algorithms/kmeans:kernel",
    ],
    data = ["@onedal//examples/daal:data"],
    extra_deps = [":example_util", "@tbb//:tbb"],
)
```

#### **3. CPU Dispatcher Generation**
```python
# Generate CPU dispatch headers
dal_generate_cpu_dispatcher(
    name = "cpu_dispatcher",
    out = "_dal_cpu_dispatcher_gen.hpp",
)
```

### **Bazel Configuration Patterns**

#### **1. CPU Feature Detection**
```python
# CPU extension support
_ISA_EXTENSIONS = ["sse2", "sse42", "avx2", "avx512"]
_ISA_EXTENSIONS_MODERN = ["sse2", "avx2", "avx512"]

# Auto-detection with fallback
def _cpu_info_impl(ctx):
    if ctx.build_setting_value == "auto":
        isa_extensions = [ctx.attr.auto_cpu]  # Auto-detect
    elif ctx.build_setting_value == "modern":
        isa_extensions = _ISA_EXTENSIONS_MODERN
    else:
        isa_extensions = ctx.build_setting_value.split(" ")
    
    # Always include SSE2
    isa_extensions = utils.unique(["sse2"] + isa_extensions)
```

#### **2. Compiler Flag Management**
```python
# Platform-specific flags
lnx_cc_common_flags = [
    "-fwrapv",
    "-fstack-protector-strong", 
    "-fno-delete-null-pointer-checks",
    "-Werror",
    "-Wformat",
    "-Wformat-security",
    "-Wreturn-type",
]

# Compiler-specific additions
def get_default_flags(arch_id, os_id, compiler_id, category = "common"):
    if compiler_id == "icx":
        flags = flags + [
            "-qopenmp-simd",
            "-no-intel-lib=libirc",
            "-no-canonical-prefixes",
        ]
    elif compiler_id == "icpx":
        flags = flags + ["-fsycl", "-fno-canonical-system-headers"]
```

#### **3. SYCL/DPC++ Integration**
```python
# DPC++ specific compilation
COMPILER.lnx.dpcpp = icpx -fsycl -m64 -stdlib=libstdc++ \
                     -fgnu-runtime -fwrapv -Werror -Wreturn-type \
                     -fsycl-device-code-split=per_kernel

# SYCL linking
link.dynamic.lnx.dpcpp = icpx -fsycl -m64 \
                         -fsycl-device-code-split=per_kernel \
                         -fsycl-max-parallel-link-jobs=$(SYCL_LINK_PRL)
```

## 🔧 Build System Integration

### **Cross-System Dependencies**

#### **TBB Integration**
```makefile
# Make system TBB detection
TBBDIR := $(if $(wildcard $(DIR)/__deps/tbb/$(_OS)/*),$(DIR)/__deps/tbb/$(_OS))
TBBDIR.include := $(TBBDIR)/include/tbb $(TBBDIR)/include
TBBDIR.libia := $(TBBDIR)/lib$(if $(OS_is_mac),,$(if $(OS_is_win),/vc_mt,/$(TBBDIR.libia.lnx.gcc)))
```

#### **MKL Integration**
```makefile
# MKL backend configuration
ifeq ($(BACKEND_CONFIG), mkl)
    MKL_LIBS := -lmkl_intel_lp64 -lmkl_core -lmkl_sequential
    MKL_INCLUDES := -I$(MKLROOT)/include
else ifeq ($(BACKEND_CONFIG), ref)
    REF_LIBS := -lopenblas -llapack
endif
```

### **Build Environment Variables**

#### **Required Environment Variables**
```bash
# TBB installation
export TBBROOT=/path/to/tbb

# MKL installation (if using MKL backend)
export MKLROOT=/path/to/mkl

# Intel oneAPI (for DPC++ compilation)
export ONEAPI_ROOT=/path/to/oneapi
export PATH=$ONEAPI_ROOT/compiler/latest/linux/bin:$PATH
```

## 📚 Related Context Files

### **For This Area**
- **[dev/AGENTS.md](../../dev/AGENTS.md)** - Development tools context
- **[dev/bazel/AGENTS.md](../../dev/bazel/AGENTS.md)** - Bazel-specific context

### **For Other Areas**
- **[C++ Development](cpp.md)** - C++ standards and patterns
- **[Examples](examples.md)** - Build system usage examples
- **[CI Workflows](ci-workflows.md)** - Build validation in CI

## 🔄 Cross-Reference Navigation

**For Build Issues**: Use this file for build system guidance
**For Production Builds**: Refer to Make system patterns
**For Development**: Use Bazel system for new features
**For Integration**: Check CMake examples for user projects

---

**Note**: This file provides comprehensive build system guidance for AI agents. For detailed implementation, refer to the specific build files in `dev/make/` and `dev/bazel/` directories.
