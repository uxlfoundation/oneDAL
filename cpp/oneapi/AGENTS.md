# Modern oneAPI Interface - AI Agents Context

> **Purpose**: AI guide for oneAPI interface (modern C++17, SYCL, distributed computing)

## 🎯 Quick Rules

- **Headers**: `.hpp` files with `#pragma once`
- **Memory**: STL RAII (`std::unique_ptr`, `std::shared_ptr`)
- **Errors**: C++ exceptions (`std::invalid_argument`, `std::domain_error`)
- **GPU**: Intel SYCL with USM for CPU/GPU operations
- **Namespace**: `oneapi::dal::v1` (stable), `preview` (experimental)

## 🚀 Essential Commands

### Bazel build and test
```bash
# Build oneAPI interface
`bazel build //cpp/oneapi/dal:core`

# Run CPU tests
`bazel test //cpp/oneapi/dal:tests`

# Run GPU tests
`bazel test --config=dpc //cpp/oneapi/dal:tests`
```

### Make and CMake build

```bash
# Build oneAPI interface with CPU-only support
make onedal_c

# Build oneAPI interface with CPU and GPU support
make onedal_dpc

# Build dynamic link version of examples
export CC=icx
export CXX=icpx
cmake -G "Unix Makefiles" -DONEDAL_LINK=dynamic
make
```

## 🛠️ Core Patterns

### Algorithm Usage
```cpp
#include "oneapi/dal/algo/kmeans.hpp"

// Configure algorithm
auto desc = kmeans::descriptor<float>()
    .set_cluster_count(10)
    .set_max_iteration_count(100);

// CPU execution
auto result = train(desc, data);

// GPU execution
sycl::queue gpu_q(sycl::gpu_selector_v);
auto gpu_result = train(gpu_q, desc, data);

// Distributed execution
auto comm = spmd::make_communicator();
auto dist_result = train(comm, desc, data);
```

### Data Tables

Data tables in oneDAL are always immutable. There is intentionally no way to overwrite the data in the numeric table.

Data ownership variant depends on the way the table is constructed.

```cpp
#include "oneapi/dal/table/homogen.hpp"

// Create table with type safety
auto table = homogen_table::wrap(data, rows, cols);

// Access data
auto accessor = row_accessor<const float>(table);
auto subset = accessor.pull({0, 10}); // Rows 0-9
const float * data_block subset.get_data();

// Pull memory with device access
auto subset_gpu = accessor.pull({0, 10}, sycl::usm::alloc::device);
// SYCL USM pointer
const float * gpu_data_block subset_gpu.get_data();
```

### Exception Handling
```cpp
try {
    auto result = train(desc, data);
} catch (const dal::invalid_argument& e) {
    // Handle parameter error
} catch (const dal::unimplemented& e) {
    // Handle unsupported operation
}
```

### Memory Management (RAII)
```cpp
class DataProcessor {
private:
    std::unique_ptr<float[]> buffer_;
    std::shared_ptr<homogen_table> table_;

public:
    DataProcessor(size_t size)
        : buffer_(std::make_unique<float[]>(size))
        , table_(std::make_shared<homogen_table>(buffer_.get(), rows, cols)) {}
};
```

### SYCL GPU Kernels
```cpp
template <typename Float>
sycl::event gpu_compute(sycl::queue& q,
                       const Float * data,  // SYCL USM data pointer
                       std::int64_t n,
                       const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(n, 256), [=](sycl::nd_item<1> item) {
            // Dependencies handling
            cgh.depends_on(deps);
            const auto idx = item.get_global_id(0);
            // GPU computation
        });
    });
}
```

## 🎯 Critical Rules

- **Memory**: Always use STL smart pointers, never raw pointers for ownership
- **Headers**: Use `.hpp` with `#pragma once`, `oneapi::dal` namespace
- **GPU**: SYCL integration with USM for zero-copy operations
- **Type Safety**: Template metaprogramming with compile-time dispatch
- **Interface**: Never mix DAAL and oneAPI patterns in same file

## 🔗 References

- **[AGENTS.md](../../AGENTS.md)** - Repository overview
- **[cpp/daal/AGENTS.md](../daal/AGENTS.md)** - Traditional DAAL interface
- **[.github/instructions/cpp-coding-guidelines.instructions.md](../../.github/instructions/cpp-coding-guidelines.instructions.md)** - Detailed C++ standards
