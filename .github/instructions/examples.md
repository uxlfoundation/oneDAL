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

### 🟢 **Development Patterns (Nice to Have)**
- **Modern C++**: Use C++17 features appropriately
- **Clean Code**: Readable and maintainable
- **Best Practices**: Follow project conventions

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
- **Data Validation**: Type checking, range validation
- **Memory Management**: Efficient data handling

### Performance Examples
- **CPU Optimization**: SIMD, threading
- **GPU Acceleration**: SYCL, memory management
- **Distributed Computing**: MPI, load balancing
- **Benchmarking**: Performance measurement

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

auto desc = kmeans::descriptor<float>()
    .set_cluster_count(10);
auto result = train(desc, data);
```

### DAAL Context (`cpp/daal/`, `examples/daal/`)
```cpp
// Traditional DAAL pattern (concise)
#include "algorithms/kmeans/kmeans_batch.h"
#include "data_management/data/homogen_numeric_table.h"

auto training = new kmeans_batch<float>();
training->parameter.nClusters = 10;
training->compute();
```

### Build System Context
```makefile
# Make example (production builds)
CXXFLAGS = -std=c++17 -O3
LIBS = -lonedal
TARGET = example
SOURCES = example.cpp

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)
```

## 🚫 What NOT to Include in Examples

- **Verbose Code**: Keep examples concise for quick review
- **Unnecessary Complexity**: Focus on essential patterns
- **Platform-Specific Code**: Ensure cross-platform compatibility
- **Incomplete Error Handling**: Always include proper error checking
- **Hardcoded Values**: Use configuration parameters

## ✅ What TO Include in Examples

- **Essential Patterns**: Core functionality demonstration
- **Error Handling**: Proper exception safety
- **Resource Management**: RAII and smart pointers
- **Documentation**: Clear comments and usage
- **Testing**: Include test cases where appropriate

## 🔧 Example Development Guidelines

### 1. Keep Examples Concise
- **Focus on essentials**: Core functionality only
- **Remove verbosity**: Eliminate unnecessary code
- **Clear structure**: Easy to follow and understand

### 2. Ensure Completeness
- **Self-contained**: Examples should run independently
- **Dependencies**: Include necessary includes and setup
- **Error handling**: Proper exception safety

### 3. Follow Project Patterns
- **Interface consistency**: Use appropriate interface
- **Coding standards**: Follow project conventions
- **Build compatibility**: Ensure Make compatibility

### 4. Include Documentation
- **Clear comments**: Explain what the example does
- **Usage instructions**: How to run and use
- **Expected output**: What to expect from the example

## 🧪 Example Testing and Validation

### Build Testing
- **Compilation**: Examples compile without errors
- **Linking**: Proper library linking
- **Dependencies**: Dependency resolution works
- **Platforms**: Cross-platform compatibility

### Runtime Testing
- **Execution**: Examples run successfully
- **Output validation**: Expected results
- **Performance**: No performance regression
- **Memory**: No memory leaks

### Integration Testing
- **Build systems**: Work with different build systems
- **Dependencies**: Different dependency versions
- **Environments**: Different environments
- **CI/CD**: Integration with continuous integration

## 📖 Quick Reference Templates

### Basic Algorithm Template
```cpp
// Quick algorithm template
auto desc = algorithm::descriptor<Float>()
    .set_parameter1(value1)
    .set_parameter2(value2);

auto result = train(desc, data);
// Process result...
```

### Error Handling Template
```cpp
try {
    // Algorithm execution
    auto result = execute_algorithm(desc, data);
    return result;
} catch (const std::exception& e) {
    // Log error and rethrow
    std::cerr << "Algorithm failed: " << e.what() << std::endl;
    throw;
}
```

### Resource Management Template
```cpp
// Smart pointer resource management
auto resource = std::make_unique<ResourceType>();
auto result = process_resource(*resource);
// Automatic cleanup when scope ends
```

## 🔄 Cross-Reference Navigation

### For This Area
- **[AGENTS.md](../../AGENTS.md)** - Main repository context
- **[cpp/AGENTS.md](../../cpp/AGENTS.md)** - C++ implementation context

### For Other Areas
- **[Build Systems](build-systems.md)** - Build configuration guidance
- **[C++ Development](cpp.md)** - C++ coding guidelines
- **[Documentation](documentation.md)** - Documentation standards

## 🚨 Critical Reminders for PR Review

1. **Examples must be concise** for quick review
2. **Include proper error handling** for robustness
3. **Ensure Make compatibility** for production builds
4. **Follow interface consistency** (DAAL vs oneAPI)
5. **Examples should be complete** and runnable
6. **Focus on essential patterns** for clarity

---

**Note**: Examples are crucial for user adoption and understanding. Keep them concise, complete, and review-friendly for effective PR review assistance.

**🎯 PRIMARY GOAL**: Assist with PR reviews by providing clear, concise examples that demonstrate proper usage patterns.
