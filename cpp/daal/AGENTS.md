# Traditional DAAL Interface - AI Agents Context

> **Purpose**: AI guide for traditional DAAL interface (CPU-focused, backward compatibility)

## 🎯 Quick Rules

- **Headers**: `.h` files with `#ifndef __FILE_H__` guards
- **Memory**: `daal::services::SharedPtr<T>` custom smart pointers
- **Errors**: `services::Status` return codes with `throwIfPossible()`
- **Threading**: the DAAL threading layer (`src/threading/threading.h`), with CPU-specific kernels
- **Optimization**: Multi-architecture dispatch (SSE2, AVX2, AVX-512, ARM SVE, RISC-V)

## 🚀 Essential Commands

```bash
# Build DAAL interface
make daal_c

# Platform-specific builds
make daal_c PLAT=lnx32e COMPILER=icx

# CPU target selection
make daal_c REQCPU="sse2 avx2 avx512"
```

## 🛠️ Core Patterns

### Error Handling

DAAL has a mixed approach to error handling:
- Status-based approach is used inside the CPU-specific implementations of the algorithms,
- Exception-based approach is used at the interface layer; The error codes returned from the  CPU-specific implementations are converted into C++ exceptions using `throwIfPossible()`.

If `DAAL_NOTHROW_EXCEPTIONS` macro is defined during DAAL build `throwIfPossible()` doesn't perform status code to exception conversion.

```cpp
services::Status compute() {
    services::Status status;
    if (/* allocation failed */) {
        status.add(services::ErrorMemoryAllocationFailed);
        return status;
    }
    // Computations are successful if status holds no errors on exit
    return status;
}
```

### CPU-Specific Kernel Dispatch

Containers and kernels are templated on `CpuType cpu`; the values per architecture are in `src/services/cpu_type.h`.

```cpp
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch> {
public:
    virtual services::Status compute() override;
};
```

### Data Management

DAAL numeric tables do not own the data they work with, they can be viewed as wrappers over the user-provided data. The user allocates and frees that data.

```cpp
// CSV data source allows to produce a numeric table from CSV file
FileDataSource<CSVFeatureManager> dataSource(datafile,
                                             DataSource::doAllocateNumericTable,
                                             DataSource::doDictionaryFromContext);
dataSource.loadDataBlock();
// dataSource.loadDataBlock(10); loads next 10 rows from the file
auto table = dataSource.getNumericTable();
```

## 📝 Rules for Changes

- Kernel bodies live in `.i` files, which `rg --type cpp` skips; search them with `rg -g '*.i'`. `*_fpt_cpu.cpp`, `*_fpt_dispatcher.cpp` and `*_fpt.cpp` only instantiate templates.
- Kernel templates take `daal::internal::CpuType cpu` and `algorithmFPType` as template parameters. Code compiled without the `cpu` parameter is not dispatched and can execute instructions the running CPU lacks.
- Allocate scratch memory with `TArray`, `TArrayCalloc`, `TArrayScalable` or `TArrayScalableCalloc` (`src/services/service_arrays.h`), not raw `new`. Parallelize through the threading layer (`src/threading/threading.h`, e.g. `daal::threader_for`), never TBB directly.
- For accuracy, accumulate each row into a zero-initialized local and add it back into the target, rather than accumulating in place. Clip results that must be non-negative.
- Guard every division by a row, observation or rank count against zero; in distributed runs a rank can have no rows.
- Don't wrap a temporary in a non-owning array.
- Don't use `reduction(- : x)` with OpenMP SIMD; reduce with `+` over negated terms.
- No magic numbers. Use a named constant and say where its value comes from.

## 🔗 References

- **[AGENTS.md](../../AGENTS.md)** - Repository overview
- **[cpp/oneapi/AGENTS.md](../oneapi/AGENTS.md)** - Modern oneAPI interface
- **[.github/instructions/cpp-coding-guidelines.instructions.md](../../.github/instructions/cpp-coding-guidelines.instructions.md)** - Detailed C++ coding guidelines

**Note**: This interface is maintained for backward compatibility. For new development, consider the modern oneAPI interface.
