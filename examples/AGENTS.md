
# Examples and Tutorials - AI Agents Context

> **Purpose**: Context for AI agents working with oneDAL examples, demonstrating both DAAL and oneAPI interfaces.

## 🏗️ Examples Architecture

oneDAL provides **comprehensive examples** across different platforms:
- **C++ Examples**: Traditional C++ implementations  
- **DPC++ Examples**: SYCL-based GPU acceleration
- **Distributed Examples**: MPI-based distributed computing
- **Integration Examples**: Database and framework integration

## 📁 Structure
```
examples/
├── daal/              # Traditional DAAL interface examples
│   ├── cpp/source/   # Algorithm-specific examples
│   └── data/         # Example data files
└── oneapi/           # Modern oneAPI interface examples
    ├── cpp/source/   # Algorithm-specific examples  
    ├── dpc/source/   # SYCL GPU examples
    └── data/         # Example data files
```

## 🔧 Example Patterns

### Traditional DAAL Interface
```cpp
#include "algorithms/kmeans/kmeans_batch.h"
#include "data_management/data/homogen_numeric_table.h"

using namespace daal::algorithms;

int main() {
    // Create data and algorithm
    auto data = new homogen_numeric_table<float>(nRows, nCols);
    auto kmeans = new kmeans::Batch<float>();
    
    // Set parameters and compute
    auto parameter = kmeans->getParameter();
    parameter->nClusters = 10;
    kmeans->input.set(kmeans::BatchInput::data, data);
    kmeans->compute();
    
    // Get results
    auto result = kmeans->getResult();
    return 0;
}
```

### Modern oneAPI Interface
```cpp
#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/table/homogen.hpp"

using namespace oneapi::dal;

int main() {
    // Load data and create descriptor
    auto data = read<homogen_table>(csv_file);
    auto desc = kmeans::descriptor<float>()
        .set_cluster_count(10)
        .set_max_iteration_count(100);
    
    // Train and infer
    auto train_result = train(desc, data);
    auto infer_result = infer(desc, train_result.get_model(), test_data);
    return 0;
}
```

### SYCL GPU Example
```cpp
#include <sycl/sycl.hpp>
#include "oneapi/dal/algo/kmeans.hpp"

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    auto data = read<homogen_table>(csv_file);
    auto desc = kmeans::descriptor<float>().set_cluster_count(10);
    auto result = train(q, desc, data);  // GPU execution
    return 0;
}
```

### Distributed Computing (MPI)
```cpp
#include <mpi.h>
#include "oneapi/dal/spmd/communicator.hpp"

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    auto comm = spmd::make_communicator();
    auto local_data = read<homogen_table>(local_csv_file);
    auto result = train(comm, desc, local_data);  // Distributed training
    MPI_Finalize();
    return 0;
}
```

## 🎯 Example Guidelines

### Code Quality Standards
- **Readability**: Clear, well-commented code
- **Completeness**: Self-contained, runnable examples  
- **Best Practices**: Follow oneDAL coding guidelines
- **Error Handling**: Proper error checking and handling

### Algorithm Categories
- **Classification**: Decision trees, SVM, Naive Bayes
- **Clustering**: K-means, DBSCAN, EM
- **Regression**: Linear regression, Ridge regression  
- **Dimensionality Reduction**: PCA, SVD

## 🚫 Common Pitfalls
- **Incomplete examples**: Always provide complete, runnable code
- **Poor error handling**: Include proper error checking
- **Platform dependencies**: Avoid hardcoded paths, provide cross-platform solutions

## 🧪 Testing Requirements
- **Build Testing**: Ensure examples compile without errors
- **Runtime Testing**: Verify examples run successfully
- **Performance**: Validate performance characteristics
- **Memory**: Check for memory leaks

## 🔧 Required Tools
- **Compilers**: GCC 7+, Clang 6+, MSVC 2017+
- **Build Systems**: CMake
- **Intel oneAPI**: For SYCL development  
- **MPI**: For distributed computing examples

## 📖 Further Reading
- **[AGENTS.md](../AGENTS.md)** - Repository context
- **[cpp/AGENTS.md](../cpp/AGENTS.md)** - C++ implementation
- **[docs/AGENTS.md](../docs/AGENTS.md)** - Documentation guidelines
