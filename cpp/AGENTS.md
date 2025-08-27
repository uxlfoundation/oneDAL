# oneDAL C++ Implementation - AI Agents Context

> **Purpose**: This file provides context for AI agents working with the C++ implementation of oneDAL, explaining the dual interface architecture and common patterns.

## 🏗️ C++ Architecture Overview

oneDAL provides **two distinct C++ interfaces** for different use cases and target platforms:

### 1. Traditional DAAL Interface (`cpp/daal/`)
- **Purpose**: Legacy interface for existing DAAL applications
- **Target**: CPU-focused implementations with SIMD optimizations
- **Style**: Object-oriented C++ with explicit memory management
- **Compatibility**: Backward compatible with Intel DAAL

### 2. Modern oneAPI Interface (`cpp/oneapi/`)
- **Purpose**: Modern, GPU-accelerated interface
- **Target**: CPU + GPU (SYCL) + distributed computing
- **Style**: Modern C++ with RAII and smart pointers
- **Future**: Primary development focus going forward

## 🔧 C++ Development Standards

### Language Features
- **C++ Standard**: C++14 minimum, C++17 preferred
- **Compiler Support**: GCC, Clang, MSVC (Windows)
- **Extensions**: Avoid compiler-specific extensions

### Code Organization
```
cpp/
├── daal/                    # Traditional DAAL interface
│   ├── include/            # Public headers
│   │   ├── algorithms/     # Algorithm implementations
│   │   ├── data_management/ # Data structures
│   │   └── services/       # Utility services
│   └── src/                # Implementation files
└── oneapi/                  # Modern oneAPI interface
    ├── dal/                 # Core oneAPI implementation
    │   ├── algo/           # Algorithm implementations
    │   ├── table/          # Data table abstractions
    │   └── backend/        # Backend implementations
    └── test/               # Unit tests
```

## 📚 Key C++ Patterns

### 1. Algorithm Interface Pattern
Both interfaces follow a consistent pattern for algorithms:

```cpp
// Training interface
template<typename Float, Method M>
class algorithm_training_batch {
public:
    using input_t = algorithm_training_input;
    using result_t = algorithm_training_result;
    
    void compute();
    void setInput(const input_t& input);
    result_t getResult();
};
```

### 2. Data Management
- **Numeric Tables**: Primary data structure for matrices
- **Memory Management**: Smart pointers and RAII
- **Data Sources**: CSV, binary, and streaming support

### 3. Service Layer
- **Environment Detection**: CPU features, threading
- **Memory Services**: Allocation and deallocation
- **Error Handling**: Exception-based error management

## 🎯 Implementation Guidelines

### Header Files
- **Include Guards**: Use `#pragma once` for oneAPI, traditional guards for DAAL
- **Forward Declarations**: Minimize header dependencies
- **Template Specializations**: Place in appropriate headers

### Source Files
- **Implementation**: Keep headers clean, implement in .cpp files
- **Exception Safety**: Provide strong exception guarantees
- **Thread Safety**: Document thread safety requirements

### Memory Management
- **Smart Pointers**: Use `std::unique_ptr` and `std::shared_ptr`
- **RAII**: Automatic resource management
- **Custom Allocators**: Support for custom memory allocation

## 🔍 Common Pitfalls to Avoid

### 1. Interface Mismatches
- **Don't mix** DAAL and oneAPI interfaces in the same code
- **Don't assume** compatibility between interfaces
- **Do use** the appropriate interface for your target platform

### 2. Memory Management
- **Don't use** raw pointers for ownership
- **Don't forget** to handle exceptions in destructors
- **Do use** smart pointers and RAII consistently

### 3. Threading
- **Don't use** standard threading primitives directly
- **Don't assume** thread safety without documentation
- **Do use** oneDAL threading layer abstractions

## 🧪 Testing and Validation

### Unit Tests
- **Coverage**: Aim for high test coverage
- **Mocking**: Use appropriate mocking for dependencies
- **Edge Cases**: Test boundary conditions and error cases

### Integration Tests
- **Examples**: Ensure examples build and run
- **Performance**: Validate performance characteristics
- **Compatibility**: Test with different compilers/platforms

## 📖 Further Reading

- **[cpp/daal/AGENTS.md](daal/AGENTS.md)** - Traditional DAAL interface details
- **[cpp/oneapi/AGENTS.md](oneapi/AGENTS.md)** - Modern oneAPI interface details
- **[dev/AGENTS.md](../dev/AGENTS.md)** - Build system and development tools
- **[docs/AGENTS.md](../docs/AGENTS.md)** - Documentation guidelines

---

**Next Steps**: For specific interface details, refer to the appropriate sub-AGENTS.md file in the `daal/` or `oneapi/` directories.
