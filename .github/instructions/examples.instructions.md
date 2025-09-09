---
applyTo: ["**/examples/**", "**/samples/**"]
---

# Examples Instructions for GitHub Copilot

## 🎯 **PRIMARY GOAL: PR Review Assistance**

**GitHub Copilot's main purpose is to assist with PR reviews. Examples should be concise and review-friendly.**

## Quick Reference Patterns

### 🔴 **Critical Patterns (Must Include)**
- **Error Handling**: Always include proper error checking
- **Resource Management**: Use RAII and smart pointers
- **Interface Consistency**: Follow DAAL vs oneAPI patterns
- **Build Compatibility**: Ensure Make compatibility

### 🟡 **Important Patterns (Should Include)**
- **Documentation**: Clear comments and examples
- **Testing**: Include test cases where appropriate
- **Performance**: Consider performance implications
- **Platform Support**: Cross-platform compatibility

## 🚀 Quick Start Examples

### 1. Basic Algorithm Usage (Concise)
```cpp
// Quick K-means example
auto desc = kmeans::descriptor<float>()
    .set_cluster_count(10)
    .set_max_iteration_count(100);
auto result = train(desc, data);
```

### 2. Error Handling (Essential)
```cpp
try {
    auto result = train(desc, data);
    return result;
} catch (const std::exception& e) {
    std::cerr << "Training failed: " << e.what() << std::endl;
    throw;
}
```

### 3. Resource Management (Critical)
```cpp
// Use smart pointers for ownership
auto data = std::make_unique<homogen_table>(rows, cols);
auto result = std::make_unique<training_result>();
```

## 📚 Example Categories

### Algorithm Examples
- **Classification**: Decision trees, SVM, Naive Bayes
- **Clustering**: K-means, DBSCAN, EM
- **Regression**: Linear regression, Ridge regression
- **Dimensionality Reduction**: PCA, SVD

### Data Management Examples
- **Data Loading**: CSV, binary, streaming
- **Data Transformation**: Feature scaling, normalization
- **Memory Management**: Efficient data handling

### Performance Examples
- **CPU Optimization**: SIMD, threading
- **GPU Acceleration**: SYCL, memory management
- **Distributed Computing**: MPI, load balancing

## 🔍 PR Review Example Checklist

### For New Examples
- [ ] **Concise and Clear**: Easy to understand quickly
- [ ] **Complete**: Self-contained and runnable
- [ ] **Error Handling**: Proper exception safety
- [ ] **Build Compatibility**: Works with Make builds
- [ ] **Interface Consistency**: Uses appropriate interface
- [ ] **Documentation**: Clear comments and usage

### For Example Updates
- [ ] **Backward Compatibility**: Existing examples still work
- [ ] **Performance Impact**: No performance regression
- [ ] **Documentation**: Examples are updated
- [ ] **Testing**: Examples are tested

## 🎯 Context-Aware Examples

### oneAPI Context (`cpp/oneapi/`, `examples/oneapi/`)
```cpp
// Modern oneAPI pattern (concise)
#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/table/homogen.hpp"

auto desc = kmeans::descriptor<float>().set_cluster_count(10);
auto data = read<homogen_table>(csv_file);
auto result = train(desc, data);
```

### DAAL Context (`cpp/daal/`, `examples/daal/`)
```cpp
// Traditional DAAL pattern (concise)
#include "algorithms/kmeans/kmeans_batch.h"
#include "data_management/data/homogen_numeric_table.h"

auto algorithm = new kmeans::Batch<float>();
algorithm->input.set(kmeans::data, data);
algorithm->compute();
auto result = algorithm->getResult();
```

### SYCL GPU Example
```cpp
#include <sycl/sycl.hpp>

sycl::queue q(sycl::gpu_selector_v);
auto desc = kmeans::descriptor<float>().set_cluster_count(10);
auto result = train(q, desc, data);  // GPU execution
```

## Cross-Reference
- **[AGENTS.md](/examples/AGENTS.md)** - Example patterns context
- **[coding-guidelines.md](/.github/instructions/cpp-coding-guidelines.md)** - Coding standards