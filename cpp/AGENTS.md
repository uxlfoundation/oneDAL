
# oneDAL C++ Implementation - AI Agents Context

> **Purpose**: Context for AI agents working with oneDAL's dual C++ interface architecture.

## 🏗️ C++ Architecture Overview

oneDAL provides **two distinct C++ interfaces**:

### 1. Traditional DAAL Interface (`cpp/daal/`)
- **Target**: CPU-focused with SIMD optimizations, backward compatible
- **Style**: Traditional C++ with `daal::services::SharedPtr<T>`, `services::Status` return codes
- **Headers**: `.h` files with `#ifndef` guards, `daal::algorithms` namespaces

### 2. Modern oneAPI Interface (`cpp/oneapi/`)  
- **Target**: CPU + GPU (SYCL) + distributed computing (primary development focus)
- **Style**: Modern C++17 with STL smart pointers, exceptions, RAII
- **Headers**: `.hpp` files with `#pragma once`, `oneapi::dal` namespaces

## 🔧 Development Standards

- **C++ Standard**: C++17 (no C++20/23 features for compatibility)
- **Architecture**: x86_64, ARM64 (SVE), RISC-V 64-bit with CPU-specific optimizations
- **Build**: Bazel with `dal.bzl`/`daal.bzl` rules, MKL/OpenBLAS backend selection

## 🎭 Key Template Patterns

### Template Specialization & CPU Dispatch
```cpp
// DAAL - Multi-dimensional specialization for CPU optimization
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch> {
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

// oneAPI - Type-safe dispatching with perfect forwarding  
template <typename Context, typename Float, typename Method, typename Task>
struct train_ops_dispatcher {
    train_result<Task> operator()(const Context&, const descriptor_base<Task>&,
                                  const train_parameters<Task>&, const train_input<Task>&) const;
};
```

## 🏛️ Core Design Patterns

### Memory Management
```cpp
// DAAL - Custom smart pointers
daal::services::SharedPtr<NumericTable> data_;

// oneAPI - STL smart pointers with RAII
std::unique_ptr<object> ptr_ = std::make_unique<object>();
std::shared_ptr<table> table_ = std::make_shared<table>();
```

### Error Handling  
- **DAAL**: `services::Status` return codes with `throwIfPossible()` conversion
- **oneAPI**: STL exceptions (`std::invalid_argument`, `std::domain_error`)

## ⚡ Platform Optimizations

### Multi-Architecture CPU Support
```cpp
// Compile-time CPU optimization selection
#if defined(TARGET_X86_64)
    enum CpuType { sse2 = 0, sse42 = 2, avx2 = 4, avx512 = 6 };
#elif defined(TARGET_ARM)
    enum CpuType { sve = 0 };  // ARM SVE
#elif defined(TARGET_RISCV64)
    enum CpuType { rv64 = 0 }; // RISC-V 64-bit
#endif

// SIMD optimization
#define PRAGMA_FORCE_SIMD _Pragma("ivdep")  // Intel compiler vectorization
```

### Runtime CPU Feature Detection
```cpp
enum cpu_extension : uint64_t {
    sse2 = 1U << 0, sse42 = 1U << 2, avx2 = 1U << 4, avx512 = 1U << 5
};
```

## 🌐 Dependencies & Namespaces

### Key Dependencies
- **Math**: Intel MKL (primary), OpenBLAS (reference)
- **Threading**: Intel TBB for task-based parallelism
- **GPU**: Intel SYCL for heterogeneous computing
- **Distributed**: MPI via `oneapi::dal::preview::spmd`

### Namespace Structure
```cpp
// oneAPI: oneapi::dal::{v1, preview, detail, spmd}
// DAAL: daal::{algorithms, data_management, services}::{internal}
```

## 📚 Algorithm Interface Patterns

### DAAL Pattern
```cpp
// Traditional algorithm lifecycle with explicit memory management
auto training = new algorithm_training_batch<float>();
training->input.set(data_input::data, data_table);
training->compute();
auto result = training->getResult();
auto model = result->get(training_result::model);
```

### oneAPI Pattern  
```cpp
// Modern fluent interface with automatic resource management
auto desc = kmeans::descriptor<float>()
    .set_cluster_count(10)
    .set_max_iteration_count(100);
auto train_result = train(desc, train_data);
auto infer_result = infer(desc, train_result.get_model(), test_data);
```

## 🎯 Critical Rules

### Interface Separation
- **NEVER mix DAAL and oneAPI patterns** in same file
- **DAAL**: `.h` headers, `#ifndef` guards, `daal::services::SharedPtr<T>`
- **oneAPI**: `.hpp` headers, `#pragma once`, `std::unique_ptr/shared_ptr`

### Memory & Error Handling
- **DAAL**: Custom smart pointers, `services::Status` codes
- **oneAPI**: STL RAII, C++ exceptions

### Performance
- **CPU Dispatch**: Templates specialized by `CpuType` for optimal SIMD
- **Threading**: TBB integration for task-based parallelism
- **GPU**: SYCL for heterogeneous computing

## 📖 Reference Links
- **[cpp/daal/AGENTS.md](daal/AGENTS.md)** - DAAL interface specifics
- **[cpp/oneapi/AGENTS.md](oneapi/AGENTS.md)** - oneAPI interface specifics
