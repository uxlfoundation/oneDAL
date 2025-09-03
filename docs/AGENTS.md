
# Documentation Guidelines - AI Agents Context

> **Purpose**: Context for AI agents working with oneDAL's multi-format documentation system.

## 🏗️ Documentation Architecture

oneDAL uses **multi-format documentation** for different audiences:
- **API Reference**: Doxygen-generated C++ API documentation  
- **Developer Guide**: reStructuredText-based comprehensive guides
- **Examples**: Code examples and tutorials
- **User Guides**: Installation and usage instructions

**Key Tools**: Doxygen (C++ API), Sphinx (guides), reStructuredText, Python automation

## 📁 Structure
```
docs/
├── dalapi/          # Python API documentation
├── doxygen/         # C++ API documentation  
├── source/          # reStructuredText source files
├── Makefile         # Documentation build system
```

## 📚 Documentation Patterns

### C++ API Documentation (Doxygen)
```cpp
/**
 * @brief K-means clustering algorithm implementation
 * @tparam Float Floating-point type for computations
 * @tparam Method Algorithm method (lloyd_dense, lloyd_csr)
 */
template<typename Float, Method Method>
class kmeans {
    /**
     * @brief Sets the number of clusters
     * @param[in] count Number of clusters to generate
     * @return Reference to this descriptor for method chaining
     */
    auto& set_cluster_count(std::int64_t count);
};
```

### reStructuredText Documentation
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

### Python API Documentation
```python
def kmeans_training(data, n_clusters=10, max_iterations=100):
    """
    Perform K-means clustering training.
    
    Parameters
    ----------
    data : array-like
        Input data for clustering. Shape (n_samples, n_features).
    n_clusters : int, default=10
        Number of clusters to form.
    
    Returns
    -------
    result : KMeansResult
        Training result containing cluster centers and labels.
    """
```

## 🎯 Essential Guidelines

### Content Standards
- **Accuracy**: Ensure technical correctness
- **Completeness**: Cover all public APIs  
- **Clarity**: Use clear, concise language
- **Examples**: Provide working code examples

### Maintenance Rules
- **Sync**: Keep documentation synchronized with code
- **Testing**: Validate all code examples compile and run
- **Links**: Test internal and external links regularly

## 🚫 Common Pitfalls
- **Documentation drift**: Keep docs updated with code changes
- **Incomplete examples**: Always provide complete, runnable code
- **Poor navigation**: Ensure clear structure and search capability

## 🔧 Required Tools
- **Doxygen**: 1.8.13+ for C++ API generation
- **Sphinx**: 3.0+ for reStructuredText processing  
- **Python**: 3.7+ for build automation
- **Make**: For Unix/Linux builds

## 📖 Further Reading
- **[AGENTS.md](../AGENTS.md)** - Repository context
- **[cpp/AGENTS.md](../cpp/AGENTS.md)** - C++ implementation  
- **[examples/AGENTS.md](../examples/AGENTS.md)** - Example patterns
