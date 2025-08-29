
# Documentation Guidelines - AI Agents Context

> **Purpose**: This file provides context for AI agents working with the documentation system in oneDAL, explaining the structure, guidelines, and documentation patterns.

## 🏗️ Documentation Architecture

oneDAL uses a **multi-format documentation system** to serve different audiences and use cases:

### Documentation Types
- **API Reference**: Doxygen-generated C++ API documentation
- **Developer Guide**: reStructuredText-based comprehensive guides
- **Examples**: Code examples and tutorials
- **User Guides**: Installation and usage instructions

## 📁 Directory Structure

```
docs/
├── dalapi/                   # Python API documentation
│   ├── __init__.py          # Python package initialization
│   ├── directives.py        # Custom Sphinx directives
│   ├── doxypy/              # Doxygen to Python bridge
│   └── extension.py         # Sphinx extension
├── doxygen/                  # C++ API documentation
│   ├── custom/              # Custom Doxygen templates
│   │   ├── cpp/             # C++ specific templates
│   │   └── python/          # Python specific templates
│   ├── doxygen_conf_cpp.txt # C++ Doxygen configuration
│   └── oneapi/              # oneAPI specific documentation
├── source/                   # reStructuredText source files
│   ├── _static/             # Static assets (CSS, images)
│   ├── _templates/          # HTML templates
│   ├── api/                 # API documentation
│   ├── daal/                # DAAL interface documentation
│   ├── onedal/              # oneAPI interface documentation
│   └── ...                  # Other documentation sections
├── Makefile                  # Documentation build system
└── make.bat                 # Windows build script
```

## 🔧 Documentation Tools

### Primary Tools
- **Doxygen**: C++ API reference generation
- **Sphinx**: reStructuredText documentation
- **reStructuredText**: Markup language for documentation
- **Python**: Documentation build scripts and automation

### Build System
- **Make**: Unix/Linux documentation builds
- **Batch**: Windows documentation builds
- **Python**: Cross-platform build automation

## 📚 Documentation Guidelines

### 1. C++ API Documentation (Doxygen)

#### Header File Documentation
```cpp
/**
 * @brief K-means clustering algorithm implementation
 * 
 * This class provides an implementation of the K-means clustering algorithm
 * with support for both batch and distributed processing modes.
 * 
 * @tparam Float Floating-point type for computations
 * @tparam Method Algorithm method (lloyd_dense, lloyd_csr)
 * 
 * @par Example
 * @code
 * auto desc = kmeans::descriptor<float, kmeans::method::lloyd_dense>()
 *     .set_cluster_count(10);
 * auto result = train(desc, data);
 * @endcode
 */
template<typename Float, Method Method>
class kmeans {
public:
    /**
     * @brief Sets the number of clusters
     * 
     * @param[in] count Number of clusters to generate
     * @return Reference to this descriptor for method chaining
     */
    auto& set_cluster_count(std::int64_t count);
};
```

#### Implementation Documentation
```cpp
/**
 * @brief Computes K-means clustering
 * 
 * This method performs K-means clustering on the input data using the
 * specified parameters and method.
 * 
 * @param[in] desc Algorithm descriptor with parameters
 * @param[in] data Input data table
 * @return Training result containing the trained model
 * 
 * @par Exception Safety
 * This function provides strong exception guarantee.
 * 
 * @par Thread Safety
 * This function is not thread-safe.
 */
template<typename Float, Method Method>
auto train(const kmeans::descriptor<Float, Method>& desc,
           const table& data) -> kmeans::train_result<Float, Method>;
```

### 2. reStructuredText Documentation

#### Section Structure
```rst
K-Means Clustering
==================

Overview
--------

K-means clustering is an unsupervised learning algorithm that groups
similar data points into clusters.

.. _kmeans-algorithm:

Algorithm
---------

The K-means algorithm works as follows:

1. **Initialization**: Randomly select K cluster centers
2. **Assignment**: Assign each point to nearest center
3. **Update**: Recalculate cluster centers
4. **Repeat**: Until convergence or max iterations

.. _kmeans-usage:

Usage Example
------------

.. code-block:: cpp

    #include "oneapi/dal/algo/kmeans.hpp"
    
    auto desc = kmeans::descriptor<float>()
        .set_cluster_count(10);
    auto result = train(desc, data);
```

#### Code Examples
```rst
.. _kmeans-cpp-example:

C++ Example
-----------

Complete example showing K-means clustering:

.. literalinclude:: ../../examples/oneapi/cpp/source/kmeans_batch_dense.cpp
   :language: cpp
   :start-after: // BEGIN
   :end-before: // END
   :caption: K-means clustering example
```

### 3. Python API Documentation

#### Docstring Format
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
    max_iterations : int, default=100
        Maximum number of iterations for convergence.
    
    Returns
    -------
    result : KMeansResult
        Training result containing cluster centers and labels.
    
    Examples
    --------
    >>> from onedal.cluster import KMeans
    >>> kmeans = KMeans(n_clusters=3)
    >>> result = kmeans.fit(data)
    >>> centers = result.cluster_centers_
    >>> labels = result.labels_
    
    Notes
    -----
    This implementation uses the Lloyd algorithm with dense data support.
    For sparse data, consider using CSR format.
    """
    pass
```

## 🎯 Documentation Standards

### 1. Content Guidelines
- **Accuracy**: Ensure all information is technically correct
- **Completeness**: Cover all public APIs and features
- **Clarity**: Use clear, concise language
- **Examples**: Provide working code examples

### 2. Style Guidelines
- **Consistency**: Use consistent terminology and formatting
- **Accessibility**: Ensure documentation is accessible
- **Internationalization**: Support multiple languages when possible
- **Search**: Include proper indexing and search capabilities

### 3. Maintenance Guidelines
- **Updates**: Keep documentation synchronized with code
- **Versioning**: Document version-specific changes
- **Deprecation**: Mark deprecated features clearly
- **Migration**: Provide migration guides for breaking changes

## 🔍 Documentation Patterns

### 1. API Reference Pattern
```rst
.. _api-kmeans-descriptor:

kmeans::descriptor
------------------

.. doxygenclass:: oneapi::dal::kmeans::descriptor
   :project: onedal
   :members:
   :undoc-members:
   :private-members:
```

### 2. Example Integration Pattern
```rst
.. _examples-kmeans:

K-Means Examples
================

.. toctree::
   :maxdepth: 2
   :caption: Examples
   
   examples/cpp/kmeans_batch_dense
   examples/cpp/kmeans_distributed
   examples/dpc/kmeans_gpu
```

### 3. Configuration Pattern
```rst
.. _configuration-build:

Build Configuration
==================

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.16)
    project(onedal_example)
    
    find_package(oneDAL REQUIRED)
    target_link_libraries(example oneDAL::oneDAL)
```

## 🚫 Common Pitfalls to Avoid

### 1. Documentation Drift
- **Don't let** documentation become outdated
- **Don't ignore** code changes in documentation
- **Do keep** documentation synchronized with code

### 2. Incomplete Examples
- **Don't provide** incomplete code examples
- **Don't assume** user knowledge
- **Do test** all code examples

### 3. Poor Organization
- **Don't create** deep nesting without clear navigation
- **Don't ignore** user experience
- **Do provide** clear navigation and search

## 🧪 Documentation Testing

### Validation
- **Build Testing**: Ensure documentation builds without errors
- **Link Testing**: Validate all internal and external links
- **Example Testing**: Verify all code examples compile and run

### Quality Assurance
- **Content Review**: Technical accuracy review
- **Style Review**: Consistency and formatting review
- **User Testing**: Usability and accessibility testing

## 🔧 Documentation Tools

### Required Tools
- **Doxygen**: 1.8.13+ for C++ API generation
- **Sphinx**: 3.0+ for reStructuredText processing
- **Python**: 3.7+ for build automation
- **Make**: For Unix/Linux builds

### Optional Tools
- **Sphinx Extensions**: For enhanced functionality
- **Theme Customization**: For branding and styling
- **Automation Scripts**: For CI/CD integration

## 📖 Further Reading

- **[AGENTS.md](../AGENTS.md)** - Overall repository context
- **[cpp/AGENTS.md](../cpp/AGENTS.md)** - C++ implementation context
- **[examples/AGENTS.md](../examples/AGENTS.md)** - Example code context
- **[CONTRIBUTING.md](../CONTRIBUTING.md)** - Contribution guidelines

---

**Note**: Documentation is a critical part of the oneDAL project. Maintain high quality, accuracy, and usability for all documentation.
