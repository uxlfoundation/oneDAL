
# Examples - AI Agents Context

> **Purpose**: Context for AI agents working with oneDAL example patterns demonstrating dual C++ interface usage.

## 🏗️ Examples Architecture

oneDAL examples demonstrate **three distinct interface patterns** corresponding to the dual C++ architecture:

### Interface Categories
- **DAAL Interface** (`examples/daal/cpp/source/`) - Traditional CPU-focused patterns
- **oneAPI CPU** (`examples/oneapi/cpp/source/`) - Modern C++ with fluent interfaces  
- **oneAPI GPU** (`examples/oneapi/dpc/source/`) - Heterogeneous computing with SYCL

## 📁 Structure
```
examples/
├── daal/cpp/source/           # Traditional DAAL interface examples
│   ├── covariance/           # Algorithm-specific examples
│   ├── kmeans/               # K-means clustering examples
│   └── [algorithms]/         # Other algorithm examples
├── oneapi/cpp/source/         # Modern oneAPI CPU examples
│   ├── covariance/           # Same algorithms, modern interface
│   ├── kmeans/               # Modern K-means patterns
│   └── example_util/         # Shared utilities
└── oneapi/dpc/source/         # GPU-accelerated examples
    ├── kmeans/               # SYCL-enabled K-means
    └── [algorithms]/         # GPU algorithm examples
```

## 🎭 Interface Pattern Comparison

### 1. DAAL Traditional Pattern
```cpp
// Explicit lifecycle management with status codes
#include "daal.h"
using namespace daal::algorithms;

covariance::Batch<> algorithm;
algorithm.input.set(covariance::data, dataSource.getNumericTable());
algorithm.compute();
covariance::ResultPtr res = algorithm.getResult();
printNumericTable(res->get(covariance::covariance), "Covariance matrix:");
```

**Characteristics:**
- **Headers**: `#include "daal.h"` with `using namespace daal::algorithms`
- **Memory**: Custom `SharedPtr<T>` and `NumericTable` management
- **Workflow**: Instantiate → set input → compute() → getResult()
- **Error Handling**: Implicit status checking

### 2. oneAPI CPU Pattern
```cpp
// Modern fluent interface with RAII
#include "oneapi/dal/algo/kmeans.hpp"
namespace dal = oneapi::dal;

const auto kmeans_desc = dal::kmeans::descriptor<>()
                             .set_cluster_count(20)
                             .set_max_iteration_count(5)
                             .set_accuracy_threshold(0.001);
const auto result_train = dal::train(kmeans_desc, x_train, initial_centroids);
std::cout << "Centroids:\n" << result_train.get_model().get_centroids() << std::endl;
```

**Characteristics:**
- **Headers**: `#include "oneapi/dal/algo/[algorithm].hpp"`
- **Memory**: STL RAII with automatic resource management
- **Workflow**: Descriptor configuration → train/compute → result access
- **Data**: Modern `dal::table` with CSV data sources

### 3. oneAPI GPU Pattern
```cpp
// SYCL queue integration for heterogeneous computing
#include <sycl/sycl.hpp>
#include "oneapi/dal/algo/kmeans.hpp"

void run(sycl::queue &q) {
    const auto x_train = dal::read<dal::table>(q, dal::csv::data_source{...});
    const auto result_train = dal::train(q, kmeans_desc, x_train, initial_centroids);
}
```

**Characteristics:**
- **Headers**: SYCL integration with `#include <sycl/sycl.hpp>`
- **Queue Parameter**: All operations accept `sycl::queue& q` as first parameter
- **Data Loading**: Queue-aware `dal::read<dal::table>(q, data_source)`
- **Execution**: GPU-accelerated with same API as CPU version

## 🔧 Build System Integration

### Bazel Configuration
```python
# examples/oneapi/cpp/BUILD
dal_example_suite(
    name = "kmeans",
    compile_as = ["c++"],
    srcs = glob(["source/kmeans/*.cpp"]),
    dal_deps = ["@onedal//cpp/oneapi/dal/algo:kmeans"],
    data = ["@onedal//examples/oneapi:data"],
    extra_deps = [":example_util"],
)
```

### Common Patterns
- **Algorithm Suites**: Each algorithm gets `dal_example_suite` target
- **Shared Utilities**: `example_util` module for common helpers
- **Data Dependencies**: Centralized test data management
- **OpenCL Integration**: GPU examples require OpenCL binary dependencies

## 🎯 Example Usage Patterns

### Data Loading Evolution
```cpp
// DAAL: Explicit data source management
FileDataSource<CSVFeatureManager> dataSource(fileName, 
                                             DataSource::doAllocateNumericTable,
                                             DataSource::doDictionaryFromContext);
dataSource.loadDataBlock();

// oneAPI CPU: Modern data loading
const auto input = dal::read<dal::table>(dal::csv::data_source{fileName});

// oneAPI GPU: Queue-aware data loading  
const auto input = dal::read<dal::table>(queue, dal::csv::data_source{fileName});
```

### Algorithm Configuration Evolution
```cpp
// DAAL: Parameter-based configuration
const size_t nClusters = 20;
const size_t nIterations = 5;
// Configuration through algorithm parameters

// oneAPI: Fluent descriptor pattern
const auto desc = dal::kmeans::descriptor<>()
                      .set_cluster_count(20)
                      .set_max_iteration_count(5)
                      .set_accuracy_threshold(0.001);
```

## 🏛️ Design Philosophy

### Progressive Modernization
1. **DAAL Examples**: Demonstrate traditional patterns for backward compatibility
2. **oneAPI CPU**: Show modern C++ best practices with same algorithms  
3. **oneAPI GPU**: Extend CPU patterns to heterogeneous computing

### Interface Consistency
- **Same Algorithm Logic**: Core computation remains identical across interfaces
- **Consistent Results**: All three patterns produce equivalent outputs
- **Performance Scaling**: GPU examples demonstrate acceleration without API complexity

## 🎯 Critical Example Rules

### Interface Separation
- **NEVER mix interfaces** within single example
- **DAAL**: Traditional headers, explicit lifecycle, custom smart pointers
- **oneAPI**: Modern headers, fluent API, STL RAII

### GPU Programming  
- **Queue Management**: Always pass `sycl::queue` as first parameter
- **Data Locality**: Use queue-aware data loading for optimal GPU performance
- **Memory Management**: Leverage USM for CPU/GPU data sharing

### Build Dependencies
- **Algorithm Dependencies**: Match example to correct `dal_deps` in BUILD files
- **Utility Sharing**: Use `example_util` for common patterns across examples
- **Data Management**: Reference centralized data dependencies

## 📖 Further Reading
- **[cpp/AGENTS.md](../cpp/AGENTS.md)** - C++ implementation overview
- **[cpp/daal/AGENTS.md](../cpp/daal/AGENTS.md)** - DAAL interface patterns
- **[cpp/oneapi/AGENTS.md](../cpp/oneapi/AGENTS.md)** - oneAPI interface patterns
- **[dev/AGENTS.md](../dev/AGENTS.md)** - Build system and development tools
