# Examples and Tutorials - AI Agents Context

> **Purpose**: This file provides context for AI agents working with the examples and tutorials in oneDAL, explaining the structure, patterns, and guidelines for example code.

## 🏗️ Examples Architecture

oneDAL provides **comprehensive examples** demonstrating both DAAL and oneAPI interfaces across different platforms and use cases:

### Example Categories
- **C++ Examples**: Traditional C++ implementations
- **DPC++ Examples**: SYCL-based GPU acceleration
- **Distributed Examples**: MPI-based distributed computing
- **Integration Examples**: Database and framework integration

## 📁 Directory Structure

```
examples/
├── cmake/                     # CMake configuration for examples
│   └── setup_examples.cmake  # Example setup configuration
├── daal/                      # Traditional DAAL interface examples
│   ├── BUILD                 # Bazel build configuration
│   ├── cpp/                  # C++ examples
│   │   ├── BUILD            # C++ examples build config
│   │   ├── CMakeLists.txt   # CMake configuration
│   │   └── source/          # Example source code
│   │       ├── algorithms/  # Algorithm-specific examples
│   │       ├── data_management/ # Data management examples
│   │       └── ...          # Other example categories
│   └── data/                 # Example data files
│       ├── batch/            # Batch processing data
│       ├── distributed/      # Distributed processing data
│       └── online/           # Online processing data
└── oneapi/                    # Modern oneAPI interface examples
    ├── BUILD                 # Bazel build configuration
    ├── cpp/                  # C++ examples
    │   ├── BUILD            # C++ examples build config
    │   ├── CMakeLists.txt   # CMake configuration
    │   └── source/          # Example source code
    │       ├── algorithms/  # Algorithm-specific examples
    │       ├── data_management/ # Data management examples
    │       └── ...          # Other example categories
    ├── data/                 # Example data files
    └── dpc/                  # DPC++ (SYCL) examples
        ├── BUILD            # DPC++ examples build config
        ├── CMakeLists.txt   # CMake configuration
        └── source/          # Example source code
```

## 🔧 Example Code Patterns

### 1. Traditional DAAL Interface Examples

#### Basic Algorithm Example
```cpp
#include "algorithms/kmeans/kmeans_batch.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_shared_ptr.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

int main() {
    // Create input data
    auto data = new homogen_numeric_table<float>(nRows, nCols);
    
    // Create algorithm
    auto kmeans = new kmeans::Batch<float>();
    
    // Set parameters
    auto parameter = kmeans->getParameter();
    parameter->nClusters = 10;
    parameter->maxIterations = 100;
    
    // Set input and compute
    kmeans->input.set(kmeans::BatchInput::data, data);
    kmeans->compute();
    
    // Get results
    auto result = kmeans->getResult();
    auto centroids = result->get(kmeans::ResultId::centroids);
    
    return 0;
}
```

#### Data Management Example
```cpp
#include "data_management/data_source/csv_feature_manager.h"
#include "data_management/data_source/data_source.h"

// Load data from CSV
auto dataSource = new CSVDataSource<CSVFeatureManager>(filename);
auto table = dataSource->loadDataBlock();

// Create numeric table
auto numericTable = new homogen_numeric_table<float>(table);
```

### 2. Modern oneAPI Interface Examples

#### Basic Algorithm Example
```cpp
#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/io/csv.hpp"

using namespace oneapi::dal;

int main() {
    // Load data from CSV
    auto data = read<homogen_table>(csv_file);
    
    // Create algorithm descriptor
    auto desc = kmeans::descriptor<float, kmeans::method::lloyd_dense>()
        .set_cluster_count(10)
        .set_max_iteration_count(100)
        .set_accuracy_threshold(1e-6);
    
    // Training
    auto train_result = train(desc, data);
    
    // Inference
    auto test_data = read<homogen_table>(test_csv_file);
    auto infer_result = infer(desc, train_result.get_model(), test_data);
    
    return 0;
}
```

#### SYCL GPU Example
```cpp
#include <sycl/sycl.hpp>
#include "oneapi/dal/algo/kmeans.hpp"

int main() {
    // Create SYCL queue for GPU
    sycl::queue q(sycl::gpu_selector_v);
    
    // Load data
    auto data = read<homogen_table>(csv_file);
    
    // Create algorithm descriptor
    auto desc = kmeans::descriptor<float>()
        .set_cluster_count(10);
    
    // Execute on GPU
    auto result = train(q, desc, data);
    
    return 0;
}
```

### 3. Distributed Computing Examples

#### MPI-based Example
```cpp
#include <mpi.h>
#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/spmd/communicator.hpp"

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    // Create communicator
    auto comm = spmd::make_communicator();
    
    // Load local data
    auto local_data = read<homogen_table>(local_csv_file);
    
    // Distributed training
    auto desc = kmeans::descriptor<float>().set_cluster_count(10);
    auto result = train(comm, desc, local_data);
    
    MPI_Finalize();
    return 0;
}
```

## 🎯 Example Guidelines

### 1. Code Quality Standards
- **Readability**: Clear, well-commented code
- **Completeness**: Self-contained, runnable examples
- **Best Practices**: Follow oneDAL coding guidelines
- **Error Handling**: Proper error checking and handling

### 2. Documentation Standards
- **Header Comments**: Clear description of example purpose
- **Inline Comments**: Explain complex logic
- **README Files**: Setup and usage instructions
- **Build Instructions**: Clear build and run steps

### 3. Data Management
- **Sample Data**: Include realistic sample data
- **Data Formats**: Support common formats (CSV, binary)
- **Data Validation**: Check data integrity
- **Memory Management**: Proper resource cleanup

## 🔍 Example Categories

### 1. Algorithm Examples
- **Classification**: Decision trees, SVM, Naive Bayes
- **Clustering**: K-means, DBSCAN, EM
- **Regression**: Linear regression, Ridge regression
- **Dimensionality Reduction**: PCA, SVD

### 2. Data Management Examples
- **Data Loading**: CSV, binary, streaming
- **Data Transformation**: Feature scaling, normalization
- **Data Validation**: Type checking, range validation
- **Memory Management**: Efficient data handling

### 3. Performance Examples
- **CPU Optimization**: SIMD, threading
- **GPU Acceleration**: SYCL, memory management
- **Distributed Computing**: MPI, load balancing
- **Benchmarking**: Performance measurement

### 4. Integration Examples
- **Database Integration**: MySQL, PostgreSQL
- **Framework Integration**: Apache Spark, TensorFlow
- **API Integration**: REST APIs, gRPC
- **Deployment**: Docker, Kubernetes

## 🚫 Common Pitfalls to Avoid

### 1. Incomplete Examples
- **Don't provide** incomplete code snippets
- **Don't assume** user knowledge
- **Do include** complete, runnable examples

### 2. Poor Error Handling
- **Don't ignore** error conditions
- **Don't assume** data validity
- **Do include** proper error checking

### 3. Platform Dependencies
- **Don't hardcode** platform-specific paths
- **Don't assume** tool availability
- **Do provide** cross-platform solutions

## 🧪 Example Testing and Validation

### Build Testing
- **Compilation**: Ensure examples compile without errors
- **Linking**: Verify proper library linking
- **Dependencies**: Check dependency resolution
- **Platforms**: Test on supported platforms

### Runtime Testing
- **Execution**: Verify examples run successfully
- **Output Validation**: Check expected results
- **Performance**: Validate performance characteristics
- **Memory**: Check for memory leaks

### Integration Testing
- **Build Systems**: Test with different build systems
- **Dependencies**: Test with different dependency versions
- **Environments**: Test in different environments
- **CI/CD**: Integrate with continuous integration

## 🔧 Example Development Tools

### Required Tools
- **Compilers**: GCC 7+, Clang 6+, MSVC 2017+
- **Build Systems**: Bazel, CMake, Make
- **Development Environment**: IDE or text editor
- **Version Control**: Git for source management

### Optional Tools
- **Intel oneAPI**: For SYCL development
- **MPI**: For distributed computing examples
- **Database Tools**: For integration examples
- **Performance Tools**: For benchmarking examples

## 📖 Further Reading

- **[AGENTS.md](../AGENTS.md)** - Overall repository context
- **[cpp/AGENTS.md](../cpp/AGENTS.md)** - C++ implementation context
- **[docs/AGENTS.md](../docs/AGENTS.md)** - Documentation guidelines
- **[CONTRIBUTING.md](../CONTRIBUTING.md)** - Contribution guidelines

---

**Note**: Examples are crucial for user adoption and understanding. Maintain high quality, completeness, and usability for all examples.
