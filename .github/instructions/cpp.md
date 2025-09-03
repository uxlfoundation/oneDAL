# C++ Development Instructions for GitHub Copilot

## C++ Standards and Patterns

### Language Features
- **C++ Standard**: C++17 preferred, C++14 minimum
- **Maximum Standard**: C++17 ONLY - do not use C++20/23 features for compatibility reasons
- **Compiler Support**: GCC 7+, Clang 6+, MSVC 2017+
- **Extensions**: Avoid compiler-specific extensions

### Code Organization
- **Header Files**: Keep headers clean with minimal dependencies
- **Implementation**: Implement in .cpp files, not headers
- **Forward Declarations**: Use to minimize header dependencies
- **Template Specializations**: Place in appropriate headers

## Interface-Specific Patterns

### oneAPI Interface (Modern)
```cpp
#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/table/homogen.hpp"

auto desc = kmeans::descriptor<float>()
    .set_cluster_count(10)
    .set_max_iteration_count(100);

auto train_result = train(desc, train_data);
auto infer_result = infer(desc, train_result.get_model(), test_data);
```

### DAAL Interface (Legacy)
```cpp
#include "algorithms/kmeans/kmeans_batch.h"
#include "data_management/data/homogen_numeric_table.h"

auto training = new kmeans_batch<float>();
auto parameter = training->getParameter();
parameter->nClusters = 10;

training->input.set(kmeans_batch_input::data, data);
training->compute();
```

## Memory Management

### Smart Pointers
```cpp
// ✅ CORRECT - Use smart pointers
auto data = std::make_unique<float[]>(size);
auto table = std::make_shared<homogen_table>(data.get(), rows, cols);

// ❌ AVOID - Raw pointers for ownership
float* raw_data = new float[size];  // Don't do this
```

### RAII Pattern
```cpp
// ✅ CORRECT - RAII with smart pointers
class DataProcessor {
private:
    std::unique_ptr<float[]> buffer_;
    std::shared_ptr<homogen_table> table_;
    
public:
    DataProcessor(size_t size) 
        : buffer_(std::make_unique<float[]>(size))
        , table_(std::make_shared<homogen_table>(buffer_.get(), rows, cols))
    { }
    // Automatic cleanup on destruction
};
```

## Error Handling

### oneAPI Interface (Exceptions)
```cpp
try {
    auto result = train(desc, data);
    return result.get_model();
} catch (const std::exception& e) {
    std::cerr << "Training failed: " << e.what() << std::endl;
    throw;
}
```

### DAAL Interface (Status Codes)
```cpp
daal::services::Status status = algorithm->compute();
if (status != daal::services::Status::OK) {
    daal::services::throwIfPossible(status);
    return nullptr;  // Handle error appropriately
}
```

## Template Patterns

### Function Templates
```cpp
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

### Class Templates
```cpp
template<typename Float, Method M>
class algorithm_descriptor {
private:
    std::int64_t cluster_count_ = 2;
    std::int64_t max_iterations_ = 100;
    
public:
    auto set_cluster_count(std::int64_t count) -> algorithm_descriptor& {
        cluster_count_ = count;
        return *this;
    }
};
```

## Performance Guidelines

### Const Correctness
```cpp
// ✅ CORRECT - Proper const usage
class DataTable {
public:
    std::int64_t get_row_count() const { return row_count_; }  // const method
    const float* get_data() const { return data_.get(); }     // const return
    
private:
    const std::int64_t row_count_;  // const member
    std::unique_ptr<float[]> data_;
};
```

### Move Semantics
```cpp
// ✅ CORRECT - Use move semantics
class DataManager {
public:
    DataManager(std::vector<float> data) : data_(std::move(data)) {}
    
    auto get_data() && -> std::vector<float> {  // Move-qualified
        return std::move(data_);
    }
};
```

## Cross-Reference
- **[coding-guidelines.md](coding-guidelines.md)** - Comprehensive coding standards
- **[general.md](general.md)** - General repository context
- **[AGENTS.md](../../cpp/AGENTS.md)** - C++ implementation context