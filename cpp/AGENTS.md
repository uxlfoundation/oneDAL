
# oneDAL C++ Implementation - AI Agents Context

> **Purpose**: Context for AI agents working with oneDAL's dual C++ interface architecture.

## 🏗️ C++ Architecture Overview

oneDAL provides **two distinct C++ interfaces**:

### 1. Traditional DAAL Interface (`cpp/daal/`)
- **Target**: CPU-focused with SIMD optimizations, backward compatible
- **Style**: Traditional C++ with `daal::services::SharedPtr<T>`, `services::Status` return codes, highly-nested namespaces
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
// DAAL - Custom objects collections
daal::services::Collection<int> collection(5);

// oneAPI - STL smart pointers with RAII
std::unique_ptr<object> ptr_ = std::make_unique<object>();
std::shared_ptr<table> table_ = std::make_shared<table>();
// oneAPI - STL containers
std::vector<int> vec(5);
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
enum class cpu_feature : uint64_t {
    unknown = 0ULL,
    sstep = 1ULL << 0,          // Intel(R) SpeedStep
    tb = 1ULL << 1,             // Intel(R) Turbo Boost
    avx512_bf16 = 1ULL << 2,    // AVX512 bfloat16
    avx512_vnni = 1ULL << 3,    // AVX512 VNNI
    tb3 = 1ULL << 4             // Intel(R) Turbo Boost Max 3.0
};
```

## 🌐 Dependencies & Namespaces

### Key Dependencies
- **Math**: Intel MKL (primary), OpenBLAS (reference)
- **Threading**: Intel TBB for task-based parallelism
- **GPU**: Intel SYCL for heterogeneous computing
- **Distributed**: MPI via `oneapi::dal::preview::spmd`

### Namespace Structure

- **oneAPI**:
  - `oneapi::dal`: Top level oneDAL namespace.
    - `oneapi::dal::{...}::backend`: APIs for internal oneDAL use, not visible to the users.
    - `oneapi::dal::{...}::detail`: APIs that are visible to the users, but might be a subject to change. Those APIs do not follow ABI compatibility requirements.
    - `oneapi::dal::{...}::preview`: Functionality added into a product for users to try it out. Also might be a subject to change or removal, and does not follow the ABI compatibility requirements.
    - `oneapi::dal::<algorithm>`, for example `oneapi::dal::kmeans`: Namespace of a respective algorithm.
- **DAAL**:
  - `daal`: Top level DAAL namespace.
    - `daal::algorithms`: Algorithms and related classes like `Parameter`, `Input`, `Result`.
    - `daal::data_management`: Numeric tables and data sources.
    - `daal::services`: Error handling, `SharedPtr`, `Collection`.
    - `daal::{}::internal`: APIs for internal DAAL use, not visible to the users.

## 📚 Algorithm Interface Patterns

### DAAL Pattern
```cpp
// Traditional algorithm lifecycle with explicit memory management
using rr_train = daal::ridge_regression::training;
rr_train::Batch<float> training(2.0 /* ridge coefficient */);
training.input.set(rr_train::data, data_table);
training.input.set(rr_train::dependentVariables, dependents);
training.compute();
auto result = training->getResult();
auto model = result->get(rr_train::model);
```

### oneAPI Pattern
```cpp
// Modern fluent interface with automatic resource management
auto desc = dal::kmeans::descriptor<float>()
    .set_cluster_count(10)
    .set_max_iteration_count(100);
auto train_result = dal::train(desc, data_table);
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

## 📖 Further Reading
- **[AGENTS.md](/AGENTS.md)** - Repository overview and context
- **[cpp/daal/AGENTS.md](/cpp/daal/AGENTS.md)** - DAAL interface specifics
- **[cpp/oneapi/AGENTS.md](/cpp/oneapi/AGENTS.md)** - oneAPI interface specifics
- **[dev/AGENTS.md](/dev/AGENTS.md)** - Build system architecture
- **[dev/bazel/AGENTS.md](/dev/bazel/AGENTS.md)** - Bazel-specific patterns
- **[docs/AGENTS.md](/docs/AGENTS.md)** - Documentation generation
- **[examples/AGENTS.md](/examples/AGENTS.md)** - Example integration patterns
