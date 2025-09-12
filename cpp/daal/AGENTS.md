# Traditional DAAL Interface - AI Agents Context

> **Purpose**: AI guide for traditional DAAL interface (CPU-focused, backward compatibility)

## 🎯 Quick Rules

- **Headers**: `.h` files with `#ifndef __FILE_H__` guards
- **Memory**: `daal::services::SharedPtr<T>` custom smart pointers
- **Errors**: `services::Status` return codes with `throwIfPossible()`
- **Threading**: TBB-based with CPU-specific kernels
- **Optimization**: Multi-architecture dispatch (SSE2, AVX2, AVX-512, ARM SVE, RISC-V)

## 🚀 Essential Commands

```bash
# Build DAAL interface
`make daal_core`

# Platform-specific builds
`make PLAT=lnx32e COMPILER=icx`

# CPU target selection
`make REQCPU="sse42 avx2 avx512"`
```

## 🛠️ Core Patterns

### Algorithm Pattern
```cpp
#include "algorithms/kmeans/kmeans_batch.h"
#include "data_management/data/homogen_numeric_table.h"

// Traditional algorithm lifecycle
auto training = new kmeans::Batch<float>();
auto parameter = training->getParameter();
parameter->nClusters = 10;

training->input.set(kmeans::data, data_table);
services::Status status = training->compute();
if (status != services::Status::OK) {
    throwIfPossible(status);
}

auto result = training->getResult();
auto model = result->get(kmeans::model);
```

### Memory Management
```cpp
// Custom smart pointer system
daal::services::SharedPtr<NumericTable> data_;
services::SharedPtr<services::KernelErrorCollection> errors_;

// Usage pattern
auto table = new HomogenNumericTable<float>(rows, cols);
services::SharedPtr<NumericTable> shared_table(table);
```

### Status-Based Error Handling
```cpp
// Hierarchical error status system
services::Status compute() {
    // Implementation
    return services::Status::OK;
}

// Usage pattern
services::Status status = algorithm->compute();
if (status != services::Status::OK) {
    daal::services::throwIfPossible(status);
    return nullptr;
}
```

### CPU-Specific Kernel Dispatch
```cpp
// Multi-architecture template specialization
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch> {
public:
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

### Data Management
```cpp
// NumericTable as primary data structure
auto table = new HomogenNumericTable<float>(rows, cols);
float* data = table->getArray();

// CSV data source
auto dataSource = new FileDataSource<CSVFeatureManager>(filename);
auto table = dataSource->loadDataBlock();
```

## 🎯 Critical DAAL Rules

- **Memory**: Use `daal::services::SharedPtr<T>`, never raw pointers for ownership
- **Error Handling**: Check `services::Status` return codes, use `throwIfPossible()`
- **Headers**: Traditional `#ifndef` guards, `daal::algorithms` namespaces
- **Templates**: CPU-specific specialization for performance optimization
- **Interface**: Never mix DAAL and oneAPI patterns in same file

## 🔗 References

- **[AGENTS.md](../../AGENTS.md)** - Repository overview
- **[cpp/oneapi/AGENTS.md](../oneapi/AGENTS.md)** - Modern oneAPI interface
- **[.github/instructions/cpp-coding-guidelines.instructions.md](../../.github/instructions/cpp-coding-guidelines.instructions.md)** - Detailed C++ standards

**Note**: This interface is maintained for backward compatibility. For new development, consider the modern oneAPI interface.
