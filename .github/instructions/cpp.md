# Copyright contributors to the oneDAL project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
auto table = std::make_shared<homogen_table>(rows, cols);

// ❌ WRONG - Don't use raw pointers for ownership
float* data = new float[size];  // Memory leak risk
```

### RAII Principles
```cpp
class DataManager {
private:
    std::unique_ptr<float[]> data_;
    size_t size_;
    
public:
    DataManager(size_t size) 
        : data_(std::make_unique<float[]>(size))
        , size_(size) {}
    
    // Destructor automatically cleans up
};
```

## Error Handling

### Exception Safety
```cpp
void safe_operation() {
    auto resource = std::make_unique<Resource>();
    
    try {
        resource->do_work();
    } catch (const std::exception& e) {
        // Handle exception appropriately
        throw;
    }
}
```

### Status Checking (DAAL)
```cpp
#include "services/error_handling.h"

if (status != services::Status::OK) {
    services::throwIfPossible(status);
}
```

## Modern C++ Patterns

### Range-based For Loops
```cpp
// ✅ CORRECT - Use range-based for
std::vector<float> data;
for (const auto& value : data) {
    // Process value
}

// ❌ WRONG - Don't use index-based loops
for (size_t i = 0; i < data.size(); ++i) {
    // Process data[i]
}
```

### Auto Keyword
```cpp
// ✅ CORRECT - Use auto for type deduction
auto result = train(desc, data);
auto table = homogen_table::wrap(data, rows, cols);

// ❌ WRONG - Don't repeat obvious types
homogen_table<float> table = homogen_table<float>::wrap(data, rows, cols);
```

### Lambda Expressions
```cpp
// Use lambdas for simple operations
auto processor = [](const auto& value) {
    return value * 2.0f;
};

std::transform(data.begin(), data.end(), result.begin(), processor);
```

## Threading and Concurrency

### oneDAL Threading Layer
```cpp
// ✅ CORRECT - Use oneDAL threading abstractions
#include "oneapi/dal/threading.hpp"

// ❌ WRONG - Don't use standard threading directly
#include <thread>
std::thread worker([]{ /* work */ });  // Avoid this
```

### SYCL Integration
```cpp
#include <sycl/sycl.hpp>
#include "oneapi/dal/algo/kmeans.hpp"

// Create SYCL queue
sycl::queue q(sycl::gpu_selector_v);

// Execute on GPU
auto result = train(q, desc, data);
```

## Data Management

### Table Creation
```cpp
// oneAPI - Modern table creation
auto table = homogen_table::wrap(data, row_count, column_count);
auto csr_table = csr_table::wrap(data, row_count, column_count, row_offsets);

// DAAL - Legacy table creation
auto table = new homogen_numeric_table<float>(nRows, nCols);
auto csr_table = new csr_numeric_table<float>(data, nRows, nCols, rowOffsets);
```

### Data Access
```cpp
// oneAPI - Safe data access
auto accessor = row_accessor<const float>(table);
auto row_data = accessor.pull({0, 10}); // Rows 0-9

// DAAL - Direct data access
float* data = table->getArray();
size_t nRows = table->getNumberOfRows();
```

## Common Pitfalls to Avoid

1. **Interface Mixing**: Never mix DAAL and oneAPI in same file
2. **Memory Leaks**: Always use smart pointers for ownership
3. **Exception Safety**: Provide strong exception guarantees
4. **Thread Safety**: Use oneDAL threading layer, not standard primitives
5. **Platform Dependencies**: Avoid hardcoded platform-specific code
6. **C++20/23 Features**: Do not use C++20/23 features - stick to C++17 maximum for compatibility

## Best Practices

1. **Consistency**: Follow existing patterns in the file
2. **Documentation**: Include clear comments for complex logic
3. **Testing**: Ensure code is testable and maintainable
4. **Performance**: Consider performance implications of design choices
5. **Maintainability**: Write code that others can understand and modify

---

**Note**: Always check the file path to determine which interface to use. Follow the patterns already established in the file you're working with.
