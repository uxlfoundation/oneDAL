---
applyTo: ["**/docs/**", "**/*.rst", "**/*.md", "**/Doxyfile"]
---

# Documentation Instructions for GitHub Copilot

## Documentation Architecture

oneDAL uses a multi-format documentation system:
- **API Reference**: Doxygen-generated C++ API documentation
- **Developer Guide**: reStructuredText-based comprehensive guides  
- **Examples**: Code examples and tutorials
- **User Guides**: Installation and usage instructions

## C++ API Documentation (Doxygen)

### Header File Documentation
```cpp
/**
 * @brief K-means clustering algorithm implementation
 * 
 * @tparam Float Floating-point type for computations
 * @tparam Method Algorithm method (lloyd_dense, lloyd_csr)
 * 
 * @par Example
 * @code
 * auto desc = kmeans::descriptor<float>().set_cluster_count(10);
 * auto result = train(desc, data);
 * @endcode
 * 
 * @par Thread Safety
 * This class is not thread-safe.
 */
template<typename Float, Method Method>
class kmeans {
public:
    /**
     * @brief Sets the number of clusters
     * @param[in] count Number of clusters to generate
     * @return Reference to this descriptor for method chaining
     */
    auto& set_cluster_count(std::int64_t count);
};
```

### Implementation Documentation
```cpp
/**
 * @brief Computes K-means clustering
 * @param[in] desc Algorithm descriptor with parameters
 * @param[in] data Input data table
 * @return Training result containing the trained model
 * 
 * @par Exception Safety
 * Strong exception guarantee - if an exception is thrown, the program state remains unchanged.
 * 
 * @par Thread Safety
 * This function is not thread-safe.
 */
template<typename Float, Method Method>
auto train(const kmeans::descriptor<Float, Method>& desc, const table& data);
```

## reStructuredText Documentation

### Section Structure
```rst
K-Means Clustering
==================

Overview
--------
K-means clustering groups similar data points into clusters.

Usage Example
------------
.. code-block:: cpp

    auto desc = kmeans::descriptor<float>().set_cluster_count(10);
    auto result = train(desc, data);
```

## Documentation Standards

### Content Guidelines
- **Accuracy**: Ensure technical correctness
- **Completeness**: Cover all public APIs
- **Clarity**: Use clear, concise language
- **Examples**: Provide working code examples

### Maintenance Guidelines
- **Updates**: Keep documentation synchronized with code
- **Testing**: Validate all code examples compile and run
- **Links**: Test internal and external links regularly

## Cross-Reference
- **[AGENTS.md](/docs/AGENTS.md)** - Documentation context
- **[examples.md](/.github/instructions/examples.md)** - Example patterns