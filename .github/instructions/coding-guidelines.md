# Coding Guidelines for GitHub Copilot

## 🎯 **PRIMARY GOAL: PR Review Assistance**

**GitHub Copilot's main purpose is to assist with PR reviews. These coding guidelines ensure consistent, high-quality code that follows oneDAL standards.**

## 📋 **Coding Style Standards**

### General Rules
- **Consistency**: Maintain consistent coding style across the codebase
- **Readability**: Code should be self-documenting and easy to understand
- **Indentation**: Use 4 spaces (not tabs)
- **File Endings**: Always put an empty line at the end of files
- **Include Guards**: Use `#pragma once` for oneAPI, traditional guards for DAAL

### C++ File Structure
```cpp
/*
 * kmeans_batch.h
 *
 * Copyright contributors to the oneDAL project
 *
 * K-means clustering algorithm batch processing interface
 */
```

## 🏷️ **Naming Conventions**

- **File Names**: Use lowercase with underscores: `kmeans_batch.h`, `data_management.cpp`
- **Classes/Structs**: Start with capital: `KMeansBatch`, `DataTable`, `AlgorithmDescriptor`
- **Functions**: Descriptive verb-noun: `computeClustering()`, `validateInput()`
- **Variables**: Meaningful names: `clusterCount` (not `n`), `errorCount` (not `nerr`)
- **Class Members**: Descriptive with underscore: `clusterCenters_`, `iterationCount_`
- **Constants**: UPPER_CASE: `MAX_ITERATIONS`, `DEFAULT_CLUSTER_COUNT`

## 🔧 **Programming Guidelines**

### Memory Management

#### DAAL Interface (`cpp/daal/`)
- **Use**: `daal::services::SharedPtr<T>` for memory management
- **Pattern**: Follow existing DAAL memory management patterns

```cpp
daal::services::SharedPtr<daal::data_management::NumericTable> data_;
```

#### oneAPI Interface (`cpp/oneapi/`)
- **Use**: `std::unique_ptr` for exclusive ownership
- **Pattern**: Follow modern C++ RAII principles

```cpp
std::unique_ptr<oneapi::dal::table> data_;
```

### Error Handling

#### DAAL Interface
- **Status Codes**: `if (status != services::Status::OK) { /* handle */ }`

#### oneAPI Interface
- **Exceptions**: `try { /* operation */ } catch (const std::exception& e) { /* handle */ }`

### Class Organization
```cpp
class AlgorithmClass {
public:
    // Public interface first
    void compute();
    Result getResult() const;
    
protected:
    // Protected members
    virtual void initialize();
    
private:
    // Private implementation and data
    DataTable data_;
    Parameters params_;
};
```

## 🚨 **Critical Requirements**

### Type Safety
- **Bounds Checking**: Always check array bounds and container sizes
- **Null Checks**: Validate pointers before dereferencing
- **Safe Conversions**: Verify type conversions are safe

### Performance
- **Const Correctness**: Use `const` wherever possible
- **RAII**: Automatic resource management
- **Efficient Memory**: Pre-allocate containers when possible

## 🔍 **PR Review Checklist**

### Code Quality
- [ ] **Naming conventions** followed consistently
- [ ] **Error handling** proper for interface (DAAL: status codes, oneAPI: exceptions)
- [ ] **Memory management** appropriate patterns (DAAL: SharedPtr, oneAPI: STL)
- [ ] **Const correctness** applied appropriately
- [ ] **Interface consistency** maintained (no mixing DAAL/oneAPI patterns)

### Style and Security
- [ ] **Indentation** uses 4 spaces (no tabs)
- [ ] **File headers** include copyright and description
- [ ] **Type safety** with proper validation
- [ ] **Bounds checking** implemented where needed

## 🚨 **Critical Reminders for PR Review**

1. **C++17 maximum standard** - no C++20/23 features
2. **Memory management** - Use DAAL patterns for DAAL code, STL for oneAPI code
3. **Error handling** - Use status codes for DAAL, exceptions for oneAPI
4. **Interface separation** - Don't mix DAAL and oneAPI patterns

## 🔄 **Cross-Reference**
- **[AGENTS.md](../../AGENTS.md)** - Main repository context
- **[cpp/daal/AGENTS.md](../../cpp/daal/AGENTS.md)** - Traditional DAAL interface
- **[cpp/oneapi/AGENTS.md](../../cpp/oneapi/AGENTS.md)** - Modern oneAPI interface
