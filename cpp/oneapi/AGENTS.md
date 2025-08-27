# Copyright contributors to the oneDAL project
#
<!--
  ~ Licensed under the Apache License, Version 2.0 (the "License");
  ~ you may not use this file except in compliance with the License.
  ~ You may obtain a copy of the License at
  ~
  ~     http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS,
  ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ~ See the License for the specific language governing permissions and
  ~ limitations under the License.
-->

# Modern oneAPI Interface - AI Agents Context

> **Purpose**: This file provides context for AI agents working with the modern oneAPI interface, explaining the new patterns, SYCL integration, and modern C++ practices.

## 🏗️ oneAPI Interface Architecture

The modern oneAPI interface provides a **GPU-accelerated, modern C++** approach to machine learning algorithms with SYCL support, distributed computing, and contemporary C++ patterns.

### Key Characteristics
- **Memory Model**: Modern C++ with RAII and smart pointers
- **GPU Support**: SYCL integration for GPU acceleration
- **Distributed Computing**: Built-in support for multi-node setups
- **Modern C++**: C++17 features and contemporary patterns

## 📁 Directory Structure

```
cpp/oneapi/
├── dal/                     # Core oneAPI implementation
│   ├── algo/               # Algorithm implementations
│   │   ├── kmeans/         # K-means clustering
│   │   ├── pca/            # Principal Component Analysis
│   │   ├── svm/            # Support Vector Machine
│   │   └── ...             # Other algorithms
│   ├── table/              # Data table abstractions
│   │   ├── common.hpp      # Common table interfaces
│   │   ├── homogen.hpp     # Homogeneous tables
│   │   ├── csr.hpp         # Compressed Sparse Row
│   │   └── ...             # Other table types
│   ├── backend/            # Backend implementations
│   │   ├── cpu/            # CPU backend
│   │   ├── gpu/            # GPU backend
│   │   └── ...             # Other backends
│   └── spmd/               # Single Program Multiple Data
│       ├── communicator.hpp # Communication primitives
│       └── ...             # Distributed computing
├── test/                    # Unit tests
└── examples/                # Usage examples
```

## 🔧 Core Design Patterns

### 1. Modern Algorithm Interface Pattern
oneAPI algorithms use a modern, template-based approach:

```cpp
#include "oneapi/dal/algo/kmeans.hpp"

// Algorithm descriptor
auto desc = kmeans::descriptor<float, kmeans::method::lloyd_dense>()
    .set_cluster_count(10)
    .set_max_iteration_count(100)
    .set_accuracy_threshold(1e-6);

// Training
auto train_result = train(desc, train_data);

// Inference
auto infer_result = infer(desc, train_result.get_model(), test_data);
```

### 2. Data Table Pattern
Modern table abstractions with type safety:

```cpp
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

// Create table from data
auto table = homogen_table::wrap(data, row_count, column_count);

// Access data safely
auto accessor = row_accessor<const float>(table);
auto row_data = accessor.pull({0, 10}); // Rows 0-9
```

### 3. SYCL Integration Pattern
GPU acceleration with SYCL:

```cpp
#include <sycl/sycl.hpp>
#include "oneapi/dal/algo/kmeans.hpp"

// Create SYCL queue
sycl::queue q(sycl::gpu_selector_v);

// Execute on GPU
auto result = oneapi::dal::train(q, desc, data);
```

## 📚 Algorithm Categories

### Clustering
- **K-Means**: `kmeans::descriptor`, `kmeans::train`, `kmeans::infer`
- **DBSCAN**: `dbscan::descriptor`, `dbscan::train`, `dbscan::infer`

### Classification
- **Decision Forest**: `decision_forest::descriptor`
- **Support Vector Machine**: `svm::descriptor`
- **K-Nearest Neighbors**: `knn::descriptor`

### Regression
- **Linear Regression**: `linear_regression::descriptor`
- **Ridge Regression**: `ridge_regression::descriptor`

### Dimensionality Reduction
- **Principal Component Analysis**: `pca::descriptor`
- **Singular Value Decomposition**: `svd::descriptor`

### Other Algorithms
- **Covariance**: `covariance::descriptor`
- **Basic Statistics**: `basic_statistics::descriptor`
- **Graph Algorithms**: `connected_components::descriptor`

## 🎯 Implementation Guidelines

### Header Files
- **Include Guards**: Use `#pragma once`
- **Modern C++**: Leverage C++17 features
- **Template Parameters**: Use concepts when possible

### Source Files
- **Implementation**: Keep headers clean
- **Exception Safety**: Provide strong guarantees
- **Performance**: Optimize for target platform

### SYCL Integration
```cpp
// Check SYCL device capabilities
if (q.get_device().has(sycl::aspect::fp64)) {
    // Use double precision
    auto desc = algorithm::descriptor<double>();
} else {
    // Fall back to single precision
    auto desc = algorithm::descriptor<float>();
}
```

## 🔍 Common Patterns and Best Practices

### 1. Algorithm Workflow
```cpp
// 1. Create descriptor with parameters
auto desc = algorithm::descriptor<float>()
    .set_parameter1(value1)
    .set_parameter2(value2);

// 2. Train model
auto train_result = train(desc, train_data);

// 3. Get model
auto model = train_result.get_model();

// 4. Inference
auto infer_result = infer(desc, model, test_data);
```

### 2. Data Management
```cpp
// Create table from various sources
auto table1 = homogen_table::wrap(array_data, rows, cols);
auto table2 = homogen_table::wrap(vector_data, rows, cols);

// CSV input
auto table3 = read<homogen_table>(csv_file);

// Convert between table types
auto csr_table = convert_to<csr_table>(dense_table);
```

### 3. Distributed Computing
```cpp
#include "oneapi/dal/spmd/communicator.hpp"

// Create communicator
auto comm = spmd::make_communicator();

// Distributed training
auto result = train(comm, desc, local_data);
```

## 🚫 Common Pitfalls to Avoid

### 1. SYCL Integration
- **Don't assume** GPU availability
- **Don't forget** to check device capabilities
- **Do handle** fallback to CPU gracefully

### 2. Memory Management
- **Don't use** raw pointers for ownership
- **Don't forget** SYCL memory management
- **Do use** RAII and smart pointers

### 3. Distributed Computing
- **Don't assume** all nodes have same data
- **Don't forget** communication overhead
- **Do test** with different node configurations

## 🧪 Testing and Validation

### Unit Testing
- **Coverage**: Test all algorithm paths
- **SYCL**: Test on different devices
- **Edge Cases**: Test boundary conditions

### Integration Testing
- **Examples**: Ensure examples build and run
- **Performance**: Validate GPU acceleration
- **Distributed**: Test multi-node scenarios

### Performance Testing
- **CPU vs GPU**: Compare performance gains
- **Scaling**: Test weak and strong scaling
- **Memory**: Check memory usage patterns

## 🔧 Development Tools

### SYCL Development
- **Intel oneAPI DPC++**: Primary SYCL implementation
- **Intel VTune**: Performance profiling


## 📖 Further Reading

- **[cpp/AGENTS.md](../AGENTS.md)** - Overall C++ implementation context
- **[cpp/daal/AGENTS.md](../daal/AGENTS.md)** - Traditional DAAL interface
- **[dev/AGENTS.md](../../dev/AGENTS.md)** - Build system and development tools
- **[docs/AGENTS.md](../../docs/AGENTS.md)** - Documentation guidelines

---

**Note**: This is the primary interface for new development. It provides modern C++ patterns, GPU acceleration, and distributed computing capabilities.
