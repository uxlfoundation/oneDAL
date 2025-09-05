# Traditional DAAL Interface - AI Agents Context

> **Purpose**: Context for AI agents working with the traditional DAAL interface patterns and CPU optimizations.

## 🏗️ DAAL Interface Architecture

The traditional DAAL interface provides **CPU-focused, object-oriented** machine learning with explicit memory management.

### Key Characteristics
- **Headers**: `.h` files with traditional `#ifndef __FILE_H__` guards
- **Memory**: `daal::services::SharedPtr<T>` custom smart pointer system
- **Error Handling**: `services::Status` return codes with `throwIfPossible()` conversion
- **Threading**: TBB-based with CPU-specific kernels
- **Optimization**: Multi-architecture dispatch (SSE2, AVX2, AVX-512, ARM SVE, RISC-V)

### CPU Architecture Support
```cpp
// Multi-architecture template specialization
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch> {
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

// CPU type enumeration
enum CpuType {
#if defined(TARGET_X86_64)
    sse2 = 0, sse42 = 2, avx2 = 4, avx512 = 6
#elif defined(TARGET_ARM)
    sve = 0     // ARM Scalable Vector Extension
#elif defined(TARGET_RISCV64) 
    rv64 = 0    // RISC-V 64-bit
#endif
};
```

## 🎭 Advanced DAAL Template Patterns

### 1. Container-Kernel Pattern (CRTP)
```cpp
// Algorithm container manages lifecycle and dispatch
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public AnalysisContainerIface<batch> {
    typedef KMeansBatchKernel<algorithmFPType, method, cpu> KernelType;
    
    services::Status compute() override {
        return kernel.compute(input, parameter, *result);
    }
};

// Separate CPU-specific kernel implementations
template <typename algorithmFPType, Method method, CpuType cpu>
class KMeansBatchKernel : public Kernel {
    services::Status compute(const Input* input, const Parameter* par, Result* result);
};
```

### 2. DAAL Smart Pointer System
```cpp
// Custom reference-counted smart pointer with RAII
template<typename T>
class SharedPtr {
    T* _ptr;
    RefCounter* _refs;
public:
    SharedPtr(T* ptr = nullptr);
    SharedPtr(const SharedPtr& other) { _refs->inc(); }
    ~SharedPtr() { if (_refs->dec() == 0) { delete _ptr; delete _refs; } }
    T* get() const { return _ptr; }
    T& operator*() const { return *_ptr; }
};

// Usage pattern
daal::services::SharedPtr<HomogenNumericTable<float>> data_;
services::SharedPtr<services::KernelErrorCollection> errors_;
```

### 3. Status-Based Error Handling
```cpp
// Hierarchical error status system
class Status {
    services::ErrorID _id;
    SharedPtr<ErrorCollection> _collection;
public:
    static const Status OK;
    bool operator==(const Status& s) const;
};

// Exception conversion for error propagation
void throwIfPossible(const Status& s) {
    if (s != Status::OK) {
        throw std::runtime_error(s.getDescription());
    }
}
```

## ⚡ DAAL Performance Optimization

### 1. CPU-Specific Kernel Dispatch
```cpp
// Runtime CPU detection and optimal code path selection  
#define DAAL_KERNEL_CPU_PATH(cpuType, ...)                    \
    if (daal::services::Environment::getInstance()->getCpuId() == cpuType) { \
        return KernelType<DAAL_FPTYPE, cpuType>::compute(__VA_ARGS__); \
    }

// Automatic kernel selection
services::Status dispatchCompute(/* parameters */) {
#if defined(TARGET_X86_64)
    DAAL_KERNEL_CPU_PATH(avx512, /*args*/);
    DAAL_KERNEL_CPU_PATH(avx2, /*args*/);
    DAAL_KERNEL_CPU_PATH(sse42, /*args*/);
    DAAL_KERNEL_CPU_PATH(sse2, /*args*/);  // Fallback
#endif
}
```

### 2. SIMD Integration
```cpp
// Compiler-specific vectorization pragmas
#if defined(__INTEL_COMPILER)
    #define PRAGMA_FORCE_SIMD _Pragma("ivdep")
#endif

// Vectorized algorithm kernels
template <typename algorithmFPType, CpuType cpu>
void vectorizedCompute(const algorithmFPType* data, size_t n) {
    PRAGMA_FORCE_SIMD
    for (size_t i = 0; i < n; ++i) {
        result[i] = complexComputation(data[i]);
    }
}
```

### 3. BLAS/LAPACK Template Wrappers
```cpp
// CPU and data-type specific BLAS optimization
template <typename fpType, CpuType cpu>
struct MklBlas<double, cpu> {
    static void xgemm(/* parameters */) {
        #define __DAAL_MKLFN_CALL_BLAS(f_name, f_args) f_name f_args;
        __DAAL_MKLFN_CALL_BLAS(dgemm, (/* optimized parameters */));
    }
};
```

## 🔧 Core Algorithm Interface

### Algorithm Workflow Pattern
```cpp
// Traditional algorithm class with typed template parameters
template <typename algorithmFPType, Method method>
class algorithm_batch {
public:
    typedef algorithm_input<algorithmFPType> InputType;
    typedef algorithm_result ResultType;
    
    void setInput(algorithmInputId id, const NumericTablePtr& ptr);
    void compute();
    services::SharedPtr<ResultType> getResult();
    
private:
    services::SharedPtr<InputType> _input;
    services::SharedPtr<ResultType> _result;
};

// Usage example
auto training = new algorithm_training_batch<float>();
training->input.set(algorithm_input::data, data);
training->compute();
auto result = training->getResult();
auto model = result->get(training_result::model);
```

### Data Management Pattern
```cpp
// NumericTable as primary data structure
auto table = new homogen_numeric_table<float>(nRows, nCols);
float* data = table->getArray();

// CSV data source
auto dataSource = new csv_data_source<CSVFeatureManager>(filename);
auto table = dataSource->loadDataBlock();
```

## 🎯 Critical DAAL Rules

- **Memory**: Use `daal::services::SharedPtr<T>`, never raw pointers for ownership
- **Error Handling**: Check `services::Status` return codes, use `throwIfPossible()`
- **Headers**: Traditional `#ifndef` guards, `daal::algorithms` namespaces
- **Templates**: CPU-specific specialization for performance optimization
- **Threading**: TBB integration with `threader_for()` patterns

## 📖 Further Reading
- **[cpp/AGENTS.md](../AGENTS.md)** - C++ implementation overview
- **[cpp/oneapi/AGENTS.md](../oneapi/AGENTS.md)** - Modern oneAPI interface

---
**Note**: This interface is maintained for backward compatibility. For new development, consider the modern oneAPI interface.