
# Modern oneAPI Interface - AI Agents Context

> **Purpose**: Context for AI agents working with the modern oneAPI interface, SYCL integration, and modern C++ practices.

## 🏗️ oneAPI Interface Architecture

The modern oneAPI interface provides **GPU-accelerated, modern C++** approach to machine learning:
- **Memory Model**: Modern C++ with RAII and smart pointers
- **GPU Support**: SYCL integration for GPU acceleration  
- **Distributed Computing**: Built-in support for multi-node setups
- **Modern C++**: C++17 features and contemporary patterns

## 📁 Structure
```
cpp/oneapi/
├── dal/            # Core oneAPI implementation
│   ├── algo/       # Algorithm implementations
│   ├── table/      # Data table abstractions  
│   ├── backend/    # Backend implementations
│   └── spmd/       # Distributed computing
├── test/           # Unit tests
└── examples/       # Usage examples
```

## 🔧 Core Design Patterns

### Modern Algorithm Interface
```cpp
#include "oneapi/dal/algo/kmeans.hpp"

// Algorithm descriptor
auto desc = kmeans::descriptor<float>()
    .set_cluster_count(10)
    .set_max_iteration_count(100)
    .set_accuracy_threshold(1e-6);

// Training
auto train_result = train(desc, train_data);

// Inference  
auto infer_result = infer(desc, train_result.get_model(), test_data);
```

### Data Table Pattern
```cpp
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

// Create table from data
auto table = homogen_table::wrap(data, row_count, column_count);

// Access data safely
auto accessor = row_accessor<const float>(table);
auto row_data = accessor.pull({0, 10}); // Rows 0-9
```

### SYCL Integration Pattern
```cpp
#include <sycl/sycl.hpp>
#include "oneapi/dal/algo/kmeans.hpp"

// Create SYCL queue for GPU
sycl::queue q(sycl::gpu_selector_v);

// Execute on GPU
auto result = train(q, desc, data);
```

## 📚 Algorithm Categories

### Clustering  
- **K-Means**: `kmeans::descriptor`, `kmeans::train`, `kmeans::infer`
- **DBSCAN**: `dbscan::descriptor`, `dbscan::train`, `dbscan::infer`

### Classification
- **Decision Forest**: `decision_forest::descriptor`
- **Support Vector Machine**: `svm::descriptor`
- **K-Nearest Neighbors**: `knn::descriptor`

### Regression & Dimensionality Reduction
- **Linear Regression**: `linear_regression::descriptor`
- **PCA**: `pca::descriptor`
- **SVD**: `svd::descriptor`

### Other Algorithms
- **Covariance**: `covariance::descriptor`
- **Basic Statistics**: `basic_statistics::descriptor`

## 🔍 Common Patterns

### Algorithm Workflow
```cpp
// 1. Create descriptor with parameters
auto desc = algorithm::descriptor<float>()
    .set_parameter1(value1)
    .set_parameter2(value2);

// 2. Train model
auto train_result = train(desc, train_data);

// 3. Inference
auto infer_result = infer(desc, train_result.get_model(), test_data);
```

### Data Management
```cpp
// Create tables from various sources
auto table1 = homogen_table::wrap(array_data, rows, cols);
auto table2 = read<homogen_table>(csv_file);

// Convert between table types
auto csr_table = convert_to<csr_table>(dense_table);
```

### Distributed Computing
```cpp
#include "oneapi/dal/spmd/communicator.hpp"

// Create communicator and distributed training
auto comm = spmd::make_communicator();
auto result = train(comm, desc, local_data);
```

## 🚫 Common Pitfalls
- **SYCL Integration**: Don't assume GPU availability, handle CPU fallback gracefully
- **Memory Management**: Don't use raw pointers, use RAII and smart pointers
- **Distributed Computing**: Don't assume all nodes have same data, test different configurations

## 🧪 Testing Requirements
- **Unit Testing**: Test all algorithm paths, SYCL devices, edge cases
- **Integration Testing**: Ensure examples build and run, validate GPU acceleration
- **Performance Testing**: Compare CPU vs GPU gains, test scaling, check memory patterns

## 🔧 Required Tools
- **Intel oneAPI DPC++**: Primary SYCL implementation
- **Intel VTune**: Performance profiling

## 📖 Further Reading
- **[cpp/AGENTS.md](../AGENTS.md)** - C++ implementation context
- **[cpp/daal/AGENTS.md](../daal/AGENTS.md)** - Traditional DAAL interface
- **[dev/AGENTS.md](../../dev/AGENTS.md)** - Build system context
