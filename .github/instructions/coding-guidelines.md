# Coding Guidelines for GitHub Copilot

## 🎯 **PRIMARY GOAL: PR Review Assistance**

**GitHub Copilot's main purpose is to assist with PR reviews. These coding guidelines ensure consistent, high-quality code that follows oneDAL standards.**

## 📋 **Coding Style Standards**

### General Rules
- **Consistency**: Maintain consistent coding style across the codebase
- **Readability**: Code should be self-documenting and easy to understand
- **Maintainability**: Write code that others can easily maintain and modify

### Whitespace and Formatting
- **Indentation**: Use 4 spaces (not tabs)
- **Trailing Spaces**: Do not leave trailing spaces at end of lines
- **File Endings**: Always put an empty line at the end of files
- **Operator Spacing**: Surround operators with spaces (`x += 1` not `x+=1`)
- **Line Length**: Keep lines reasonably short for readability

### C++ File Structure
Each C++ file should begin with:
1. **File Header Comment**: Name and purpose of the file
2. **Copyright Notice**: Proper copyright header
3. **Brief Description**: What the file contains

```cpp
/*
 * kmeans_batch.h
 *
 * Copyright contributors to the oneDAL project
 *
 * K-means clustering algorithm batch processing interface
 */
```

### Header Files
- **Minimal Includes**: Include only necessary headers in header files
- **Forward Declarations**: Use forward declarations when possible
- **Implementation Includes**: Put includes in .cpp files when possible
- **Include Guards**: Use `#pragma once` for oneAPI, traditional guards for DAAL

## 🏷️ **Naming Conventions**

### File Names
- **Lowercase**: Use all lowercase letters
- **Underscores**: May include underscores for separation
- **Examples**: `kmeans_batch.h`, `data_management.cpp`

### Type Names
- **Classes/Structs**: Start with capital letter, use descriptive names
- **Examples**: `KMeansBatch`, `DataTable`, `AlgorithmDescriptor`

### Function Names
- **Descriptive**: Use clear, descriptive names
- **Verb-Noun**: Start with a verb describing the action
- **Examples**: `computeClustering()`, `validateInput()`, `initializeAlgorithm()`

### Variable Names
- **Meaningful**: Avoid abbreviations and single letters
- **Examples**: `clusterCount` (not `n`), `errorCount` (not `nerr`)

### Class Member Names
- **Private Members**: Use descriptive names, avoid abbreviations
- **Examples**: `clusterCenters_`, `iterationCount_`, `convergenceThreshold_`

### Constant Names
- **Uppercase**: Use UPPER_CASE for constants
- **Examples**: `MAX_ITERATIONS`, `DEFAULT_CLUSTER_COUNT`, `EPSILON_THRESHOLD`

## 🏗️ **Declaration Order**

### Class Member Organization
```cpp
class AlgorithmClass {
public:
    // Public interface first
    AlgorithmClass();
    ~AlgorithmClass();
    
    // Public methods
    void compute();
    Result getResult() const;
    
protected:
    // Protected members
    virtual void initialize();
    
private:
    // Private implementation
    void validateInput();
    void processData();
    
    // Private data members
    DataTable data_;
    Parameters params_;
};
```

### Function Parameter Order
- **Input Parameters**: First (const references when possible)
- **Output Parameters**: Last (non-const references or pointers)
- **Example**: `void process(const InputData& input, Result& output, Options options);`

## 🔧 **Programming Guidelines**

### Local Variables
- **Scope**: Declare variables as close to their use as possible
- **Initialization**: Always initialize variables at declaration
- **Const**: Use `const` when variables won't change after initialization

```cpp
// Good
const auto clusterCount = params.getClusterCount();
const auto maxIterations = params.getMaxIterations();

// Avoid
int clusterCount;
clusterCount = params.getClusterCount();
```

### Constants
- **Const Correctness**: Use `const` wherever possible
- **Const References**: Pass large objects by const reference
- **Const Methods**: Mark methods that don't modify object state as `const`

```cpp
class Algorithm {
public:
    // Const method - doesn't modify object state
    Result getResult() const { return result_; }
    
    // Non-const method - modifies object state
    void setParameters(const Parameters& params) { params_ = params; }
};
```

### Smart Pointers and RAII
- **Ownership**: Use `std::unique_ptr` for exclusive ownership
- **Shared Ownership**: Use `std::shared_ptr` when ownership is shared
- **RAII**: Follow Resource Acquisition Is Initialization principles

```cpp
// Good - RAII with smart pointers
class DataProcessor {
private:
    std::unique_ptr<DataTable> data_;
    std::shared_ptr<Algorithm> algorithm_;
    
public:
    DataProcessor(std::unique_ptr<DataTable> data, 
                  std::shared_ptr<Algorithm> algo)
        : data_(std::move(data)), algorithm_(algo) {}
    
    // Destructor automatically cleans up resources
};
```

### Function Design
- **Short Functions**: Keep functions focused and short
- **Single Responsibility**: Each function should do one thing well
- **Clear Names**: Function names should clearly describe what they do

```cpp
// Good - focused, single responsibility
void validateInputData(const DataTable& data) {
    if (data.getRowCount() == 0) {
        throw std::invalid_argument("Data table is empty");
    }
    if (data.getColumnCount() == 0) {
        throw std::invalid_argument("Data table has no columns");
    }
}

// Avoid - doing too many things
void processData(DataTable& data) {
    // Validation, processing, and output all mixed together
    // ... many lines of mixed functionality
}
```

## 🚨 **SDL Requirements (Security and Performance)**

### Type Safety
- **Safe Conversions**: Verify type conversions are safe
- **Bounds Checking**: Always check array bounds and container sizes
- **Null Checks**: Validate pointers before dereferencing

```cpp
// Good - safe type conversion with bounds checking
void processArray(const std::vector<int>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data array is empty");
    }
    
    // Safe conversion with bounds checking
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] < 0) {
            throw std::invalid_argument("Negative values not allowed");
        }
        processElement(data[i]);
    }
}
```

### Performance Considerations
- **Efficient Algorithms**: Choose appropriate algorithms for performance requirements
- **Memory Management**: Minimize memory allocations and copies
- **SIMD Optimization**: Use SIMD instructions when appropriate

```cpp
// Good - efficient memory usage
void processData(const std::vector<float>& input, std::vector<float>& output) {
    output.reserve(input.size());  // Pre-allocate to avoid reallocation
    
    for (const auto& value : input) {
        output.push_back(processValue(value));
    }
}
```

## 📝 **Comments and Documentation**

### Function Comments
- **Purpose**: Explain WHAT the function does, not HOW
- **Parameters**: Document input/output parameters
- **Exceptions**: Note any exceptions that may be thrown
- **Thread Safety**: Document thread safety requirements

```cpp
/**
 * @brief Performs K-means clustering on input data
 * 
 * @param[in] data Input data table for clustering
 * @param[in] params Clustering parameters (cluster count, iterations, etc.)
 * @return Clustering result with cluster centers and assignments
 * 
 * @throws std::invalid_argument if data is empty or invalid
 * @throws std::runtime_error if clustering fails to converge
 * 
 * @note This function is not thread-safe
 */
Result performClustering(const DataTable& data, const Parameters& params);
```

### Variable Comments
- **Purpose**: Explain WHY a variable exists, not WHAT it is
- **Constraints**: Document any constraints or assumptions
- **Units**: Specify units for numerical values

```cpp
// Maximum number of iterations for convergence
// Must be positive and reasonable (< 10000)
const int maxIterations = 1000;

// Convergence threshold for algorithm termination
// Smaller values = more accurate but slower convergence
const double epsilon = 1e-6;
```

## 🔍 **PR Review Coding Checklist**

### Code Quality
- [ ] **Naming conventions** followed consistently
- [ ] **Function design** follows single responsibility principle
- [ ] **Error handling** implemented properly
- [ ] **Memory management** uses RAII and smart pointers
- [ ] **Const correctness** applied appropriately

### Style and Formatting
- [ ] **Indentation** uses 4 spaces (no tabs)
- [ ] **Line endings** have proper spacing
- [ ] **File headers** include copyright and description
- [ ] **Comments** explain purpose, not implementation
- [ ] **Code structure** follows declaration order guidelines

### Performance and Security
- [ ] **Type safety** maintained with proper validation
- [ ] **Bounds checking** implemented where needed
- [ ] **Exception safety** provided for all operations
- [ ] **Memory efficiency** optimized for performance
- [ ] **SDL requirements** met for security


## 🚨 **Critical Reminders for PR Review**

1. **C++17 maximum standard** - no C++20/23 features
2. **Naming conventions** must be consistent
3. **RAII and smart pointers** for memory management
4. **Const correctness** for type safety
5. **Exception safety** for robustness

## 🔄 **Cross-Reference Navigation**

### For This Area
- **[AGENTS.md](../../AGENTS.md)** - Main repository context
- **[cpp/AGENTS.md](../../cpp/AGENTS.md)** - C++ implementation context

### For Other Areas
- **[Build Systems](build-systems.md)** - Build configuration guidance
- **[Examples](examples.md)** - Code pattern examples
- **[Documentation](documentation.md)** - Documentation standards

---

**Note**: These coding guidelines ensure consistent, high-quality code that follows oneDAL standards.

**🎯 PRIMARY GOAL**: Assist with PR reviews by ensuring code follows established coding standards and best practices.
