---
applyTo: ["**/*.cpp", "**/*.hpp", "**/*.h", "**/cpp/**", "**/include/**"]
---

# C++ Development and Coding Guidelines for GitHub Copilot

## 📋 **C++ Standards and Language Features**

### Language Compliance
- **C++ Standard**: C++17
- **Compiler Support**: GCC 7+, Clang 6+, MSVC 2017+
- **Extensions**: Avoid compiler-specific extensions

### Code Organization
- **Header Files**: Keep headers clean with minimal dependencies
- **Implementation**: Implement in .cpp files, not headers
- **DAAL .i Files**: Use `.i` files for template implementations requiring compile-time inclusion
- **Forward Declarations**: Use to minimize header dependencies
- **Template Specializations**: Place in appropriate headers

## 🏗️ **Interface-Specific Development Patterns**

### oneAPI Interface - `cpp/oneapi/`
```cpp
#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/table/homogen.hpp"

auto desc = kmeans::descriptor<float>()
    .set_cluster_count(10)
    .set_max_iteration_count(100);

auto train_result = train(desc, train_data);
auto infer_result = infer(desc, train_result.get_model(), test_data);
```

**oneAPI Patterns**:
- **Headers**: Use `.hpp` extension with `#pragma once`
- **Namespaces**: `oneapi::dal` structure
- **Memory Management**: `std::unique_ptr`, `std::shared_ptr`, `std::make_unique`
- **Error Handling**: C++ exceptions (`throw`, `try/catch`)

### DAAL Interface - `cpp/daal/`
```cpp
#include "algorithms/kmeans/kmeans_batch.h"
#include "data_management/data/homogen_numeric_table.h"

auto training = new kmeans_batch<float>();
auto parameter = training->getParameter();
parameter->nClusters = 10;

training->input.set(kmeans_batch_input::data, data);
services::Status status = training->compute();
if (status != services::Status::OK) {
    // Handle error
}
```

**DAAL Patterns**:
- **Headers**: Use `.h` extension with traditional include guards (`#ifndef __FILE_NAME_H__`)
- **Implementation Files**: Use `.i` extension for template implementations
- **Namespaces**: `daal::algorithms`, `daal::data_management`, `daal::services`
- **Memory Management**: `daal::services::SharedPtr<T>` (validated in codebase)
- **Error Handling**: `services::Status` return codes (validated in codebase)

### DAAL Template Implementation Files (.i files)
```cpp
// Example: kmeans_init_impl.i
#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"

namespace daal::algorithms::kmeans::init::internal {

template <Method method, typename algorithmFPType, CpuType cpu>
Status init(size_t p, size_t n, size_t nRowsTotal, size_t nClusters, 
           algorithmFPType * clusters, NumericTable * ntData, unsigned int seed) {
    // Template implementation here
}

} // namespace
```

**DAAL .i File Patterns**:
- **Usage**: Include in `.cpp` files: `#include "algorithm_method_impl.i"`
- **Purpose**: Template implementations requiring compile-time instantiation
- **Location**: Exclusively in `cpp/daal/src/` directory structure
- **CPU Specialization**: Support different CPU architectures (SSE, AVX, AVX512)
- **Naming**: Follow pattern: `{algorithm}_{method}_impl.i`
- **Build System**: Listed as headers in Bazel BUILD files

## 💾 **Memory Management Patterns**

### DAAL Interface
```cpp
// ✅ CORRECT - DAAL patterns found in codebase
daal::services::SharedPtr<daal::data_management::NumericTable> data_;
services::SharedPtr<Error> error_ptr;

// Pattern found in error handling
typedef SharedPtr<Error> ErrorPtr;
```

### oneAPI Interface
```cpp
// ✅ CORRECT - oneAPI patterns found in codebase
std::unique_ptr<uniform_voting<ClassType>> voting_;
std::shared_ptr<object_store> store_ = std::make_shared<object_store>();
explicit array(const std::shared_ptr<T>& data, std::int64_t count);

// RAII Pattern
class DataProcessor {
private:
    std::unique_ptr<float[]> buffer_;
    std::shared_ptr<homogen_table> table_;
    
public:
    DataProcessor(size_t size) 
        : buffer_(std::make_unique<float[]>(size))
        , table_(std::make_shared<homogen_table>(buffer_.get(), rows, cols))
    { }
};
```

## 🚨 **Error Handling**

### DAAL Interface - Status Codes
```cpp
// Pattern validated in codebase
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

### oneAPI Interface - Exceptions
```cpp
// Pattern validated in codebase
try {
    auto result = train(desc, data);
    return result.get_model();
} catch (const std::exception& e) {
    std::cerr << "Training failed: " << e.what() << std::endl;
    throw;
}

// Custom exceptions found in codebase
throw std::invalid_argument{ "Data types do not match" };
throw std::runtime_error{ "We reached the end of input stream" };
throw unimplemented{ dal::detail::error_messages::unsupported_data_layout() };
```

## 🏷️ **Naming Conventions**

Based on actual codebase analysis:

- **File Names**: 
  - DAAL: Lowercase with underscores: `kmeans_batch.h`, `homogen_numeric_table.h`
  - oneAPI: Lowercase with underscores: `train.hpp`, `compute_kernel.hpp`
- **Classes/Structs**: 
  - DAAL: `BatchContainer`, `HomogenNumericTable`, `KMeansBatch`
  - oneAPI: `train_ops`, `compute_kernel`, `uniform_voting`
- **Functions**: Descriptive verb-noun: `compute()`, `get_data()`, `wait_and_throw()`
- **Variables**: Meaningful names: `cluster_count`, `error_count`, `host_copy`
- **Class Members**: With underscore suffix: `entries_`, `store_`, `comm_`
- **Constants**: UPPER_CASE: `MAX_ITERATIONS`, `DEFAULT_CLUSTER_COUNT`

## 🏗️ **Template Patterns**

### Function Templates
```cpp
// Pattern from codebase
template <typename Descriptor>
struct train_ops<Descriptor, dal::kmeans::detail::descriptor_tag>
        : dal::kmeans::detail::train_ops<Descriptor> {};

template<typename T>
auto process_data(const std::vector<T>& input) -> std::vector<T> {
    std::vector<T> output;
    output.reserve(input.size());
    
    for (const auto& value : input) {
        output.push_back(process_value(value));
    }
    return output;
}
```

### Class Templates (DAAL Pattern)
```cpp
// Pattern validated in DAAL codebase
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

template <typename DataType = DAAL_DATA_TYPE>
class DAAL_EXPORT HomogenNumericTable : public NumericTable
{
    typedef DataType baseDataType;
};
```

## 📝 **Coding Style Standards**

### General Rules
- **Consistency**: Maintain consistent coding style across the codebase
- **Readability**: Code should be self-documenting and easy to understand
- **Indentation**: Use 4 spaces (not tabs)
- **File Endings**: Always put an empty line at the end of files
- **Include Guards**: 
  - oneAPI: Use `#pragma once`
  - DAAL: Traditional guards (`#ifndef __FILENAME_H__`)

### Class Organization (VALIDATED)
```cpp
template <typename DataType>
class Algorithm {
public:
    // Public interface first
    services::Status compute();
    Result getResult() const;
    
protected:
    // Protected members
    virtual void initialize();
    
private:
    // Private implementation and data
    DataTable data_;
    Parameters params_;
};
```

### Move Semantics
```cpp
// ✅ CORRECT - Modern C++ patterns
class DataManager {
public:
    DataManager(std::vector<float> data) : data_(std::move(data)) {}
    
    auto get_data() && -> std::vector<float> {
        return std::move(data_);
    }
    
private:
    std::vector<float> data_;
};
```

## 🔍 **PR Review Checklist**

### Code Quality (VALIDATED CRITERIA)
- [ ] **Interface consistency** - No mixing DAAL/oneAPI patterns
  - DAAL: `services::Status`, `SharedPtr`, `.h` headers, traditional guards
  - oneAPI: exceptions, `std::unique_ptr`/`std::shared_ptr`, `.hpp` headers, `#pragma once`
- [ ] **Memory management** follows interface patterns
  - DAAL: `daal::services::SharedPtr<T>`
  - oneAPI: `std::unique_ptr`, `std::shared_ptr`
- [ ] **Error handling** appropriate for interface
  - DAAL: `services::Status` return codes
  - oneAPI: C++ exceptions
- [ ] **Naming conventions** followed consistently
- [ ] **Header guards** correct for interface (`#pragma once` vs traditional)

### Style and Security
- [ ] **C++17 maximum standard** - no C++20/23 features
- [ ] **Indentation** uses 4 spaces (no tabs)
- [ ] **File headers** include copyright and description (validated format)
- [ ] **Type safety** with proper validation
- [ ] **Const correctness** applied appropriately
- [ ] **Bounds checking** implemented where needed

## 🚨 **Critical Reminders for PR Review**

1. **Interface Separation** - NEVER mix DAAL and oneAPI patterns in the same file
2. **Memory Management** - Use correct smart pointer types for each interface
3. **Error Handling** - Use status codes for DAAL, exceptions for oneAPI
4. **Header Extensions** - `.h` for DAAL, `.hpp` for oneAPI
5. **Include Guards** - Traditional guards for DAAL, `#pragma once` for oneAPI
6. **C++17 Compliance** - No C++20/23 features for maximum compatibility

## 🔄 **Cross-Reference**
- **[general.md](/.github/instructions/general.md)** - General repository context
- **[build-systems.md](/.github/instructions/build-systems.md)** - Build system instructions
- **[examples.md](/.github/instructions/examples.md)** - Example code patterns