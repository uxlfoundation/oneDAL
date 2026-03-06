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
`make daal_c`

# Platform-specific builds
`make daal_c PLAT=lnx32e COMPILER=icx`

# CPU target selection
`make daal_c REQCPU="sse42 avx2 avx512"`
```

## 🛠️ Core Patterns

### Algorithm Usage Pattern
```cpp
#include "algorithms/kmeans/kmeans_batch.h"
#include "data_management/data/homogen_numeric_table.h"

// Traditional algorithm lifecycle
kmeans::Batch<float> training(clusters, iterations);

training.input.set(kmeans::data, data_table);
services::Status status = training.compute();
if (!status) {
    throwIfPossible(status);
}

auto result = training.getResult();
auto model = result->get(kmeans::model);
```

### Memory Management
```cpp
// Custom smart pointer system
daal::services::SharedPtr<NumericTable> data_;
services::SharedPtr<services::KernelErrorCollection> errors_;

// Usage pattern
using dm = daal::data_management
services::Status status;
services::SharedPtr<dm::HomogenNumericTable<float>> table(dm::HomogenNumericTable<float>::create(data_array, rows, cols, &status));
// Shorter variant
dm::NumericTablePtr table(dm::HomogenNumericTable<float>::create(data_array, rows, cols, &status));
```

### Error Handling

DAAL has a mixed approach to error handling:
- Status-based approach is used inside the CPU-specific implementations of the algorithms,
- Exception-based approach is used at the interface layer; The error codes returned from the  CPU-specific implementations are converted into C++ exceptions using `throwIfPossible()`.

If `DAAL_NOTHROW_EXCEPTIONS` macro is defined during DAAL build `throwIfPossible()` doesn't perform status code to exception conversion.

```cpp
// Hierarchical error status system
services::Status compute() {
    services::Status status;

    // Implementation
    if (/* Error condition */) {
        // Add error code ID to the status
        status.add(services::SomeErrorCodeID);
        return status;
    }
    // Implementation

    // Computations are successful if status holds no errors on exit
    return status;
}
```

### CPU-Specific Kernel Dispatch
```cpp
// Multi-architecture template specialization
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch> {
public:
    virtual services::Status compute() override;
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

DAAL numeric tables do not own the data they work with, they can be viewed as wrappers over the user-provided data.

```cpp
// NumericTable as primary data structure
float * data_array = new float[rows * cols];
services::Status status;
auto table = HomogenNumericTable<float>::create(data_array, rows, cols, &status);
// User is responsible for allocation and deletion of the data assocoated with numeric tables
delete[] data_array;

// CSV data source allows to produce a numeric table from CSV file
FileDataSource<CSVFeatureManager> fileDataSource(datafile,
                                                 DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);
dataSource.loadDataBlock();
// dataSource.loadDataBlock(10); loads next 10 rows from the file
auto table = dataSource.getNumericTable();
```

## 🎯 Critical DAAL Rules

- **Memory**: Use `daal::services::SharedPtr<T>`, never raw pointers for ownership
- **Error Handling**: Check `services::Status` return codes or catch the exceptions
- **Headers**: Traditional `#ifndef` guards
- **Templates**: CPU-specific specialization for performance optimization
- **Interface**: Never mix DAAL and oneAPI patterns in same file

## 🔗 References

- **[AGENTS.md](../../AGENTS.md)** - Repository overview
- **[cpp/oneapi/AGENTS.md](../oneapi/AGENTS.md)** - Modern oneAPI interface
- **[.github/instructions/cpp-coding-guidelines.instructions.md](../../.github/instructions/cpp-coding-guidelines.instructions.md)** - Detailed C++ coding guidelines

**Note**: This interface is maintained for backward compatibility. For new development, consider the modern oneAPI interface.
