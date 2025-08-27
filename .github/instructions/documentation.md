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
 * This class provides an implementation of the K-means clustering algorithm
 * with support for both batch and distributed processing modes.
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
 * 
 * @par Exception Safety
 * This class provides strong exception guarantee.
 */
template<typename Float, Method Method>
class kmeans {
public:
    /**
     * @brief Sets the number of clusters
     * 
     * @param[in] count Number of clusters to generate
     * @return Reference to this descriptor for method chaining
     * 
     * @par Exception Safety
     * This function provides strong exception guarantee.
     */
    auto& set_cluster_count(std::int64_t count);
};
```

### Implementation Documentation
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
 * 
 * @par Complexity
 * Time complexity: O(n * k * i) where n is number of points,
 * k is number of clusters, and i is number of iterations.
 */
template<typename Float, Method Method>
auto train(const kmeans::descriptor<Float, Method>& desc,
           const table& data) -> kmeans::train_result<Float, Method>;
```

### Class Documentation
```cpp
/**
 * @brief Numeric table for homogeneous data
 * 
 * This class represents a table with homogeneous data types.
 * It provides efficient storage and access to numerical data.
 * 
 * @tparam T Data type (float, double, int32_t, int64_t)
 * 
 * @par Memory Layout
 * Data is stored in row-major order for optimal cache performance.
 * 
 * @par Thread Safety
 * This class is not thread-safe for modifications.
 * Multiple threads can read from the same table simultaneously.
 */
template<typename T>
class homogen_table {
public:
    /**
     * @brief Creates a table from existing data
     * 
     * @param[in] data Pointer to the data array
     * @param[in] row_count Number of rows
     * @param[in] column_count Number of columns
     * @return Table wrapping the provided data
     * 
     * @note The table does not take ownership of the data.
     * The caller is responsible for ensuring the data remains
     * valid for the lifetime of the table.
     */
    static auto wrap(T* data, std::int64_t row_count, std::int64_t column_count);
};
```

## reStructuredText Documentation

### Section Structure
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

### Code Examples
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
   :name: kmeans-example

.. note::
   This example demonstrates the basic usage of K-means clustering.
   For production use, consider adding error handling and validation.
```

### API Reference Integration
```rst
.. _api-kmeans-descriptor:

kmeans::descriptor
------------------

.. doxygenclass:: oneapi::dal::kmeans::descriptor
   :project: onedal
   :members:
   :undoc-members:
   :private-members:
   :outline:

.. _api-kmeans-train:

kmeans::train
--------------

.. doxygenfunction:: oneapi::dal::train
   :project: onedal
   :outline:
```

### Configuration Examples
```rst
.. _configuration-build:

Build Configuration
==================

CMake Configuration
------------------

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.16)
    project(onedal_example)
    
    find_package(oneDAL REQUIRED)
    target_link_libraries(example oneDAL::oneDAL)

Bazel Configuration
------------------

.. code-block:: python

    cc_binary(
        name = "example",
        srcs = ["main.cpp"],
        deps = ["//cpp/oneapi:dal"],
    )
```

## Python API Documentation

### Docstring Format
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
    
    See Also
    --------
    KMeans : The main KMeans class
    KMeansResult : Result object containing training results
    """
    pass
```

### Class Documentation
```python
class KMeans:
    """
    K-Means clustering algorithm.
    
    This class implements the K-means clustering algorithm with support
    for both dense and sparse data formats.
    
    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.
    max_iterations : int, default=100
        Maximum number of iterations of the k-means algorithm.
    tolerance : float, default=1e-4
        Relative tolerance with regards to inertia to declare convergence.
    
    Attributes
    ----------
    cluster_centers_ : array of shape (n_clusters, n_features)
        Coordinates of cluster centers.
    labels_ : array of shape (n_samples,)
        Labels of each point.
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.
    
    Examples
    --------
    >>> from onedal.cluster import KMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> kmeans = KMeans(n_clusters=2, random_state=0)
    >>> kmeans.fit(X)
    KMeans(n_clusters=2, random_state=0)
    >>> kmeans.labels_
    array([1, 1, 1, 0, 0, 0], dtype=int32)
    >>> kmeans.predict([[0, 0], [12, 3]])
    array([1, 0], dtype=int32)
    >>> kmeans.cluster_centers_
    array([[10.,  2.],
           [ 1.,  2.]])
    """
    
    def __init__(self, n_clusters=8, max_iterations=100, tolerance=1e-4):
        """
        Initialize KMeans.
        
        Parameters
        ----------
        n_clusters : int, default=8
            The number of clusters to form.
        max_iterations : int, default=100
            Maximum number of iterations.
        tolerance : float, default=1e-4
            Convergence tolerance.
        """
        pass
```

## Documentation Standards

### Content Guidelines
- **Accuracy**: Ensure all information is technically correct
- **Completeness**: Cover all public APIs and features
- **Clarity**: Use clear, concise language
- **C++ Standard**: Document C++17 maximum constraint clearly
- **Examples**: Provide working code examples
- **Consistency**: Use consistent terminology and formatting

### Style Guidelines
- **Accessibility**: Ensure documentation is accessible
- **Internationalization**: Support multiple languages when possible
- **Search**: Include proper indexing and search capabilities
- **Navigation**: Provide clear navigation and structure

### Maintenance Guidelines
- **Updates**: Keep documentation synchronized with code
- **Versioning**: Document version-specific changes
- **Deprecation**: Mark deprecated features clearly
- **Migration**: Provide migration guides for breaking changes

## Documentation Patterns

### Example Integration
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

### Cross-Reference Patterns
```rst
.. _related-algorithms:

Related Algorithms
=================

- :ref:`dbscan-algorithm` - Density-based clustering
- :ref:`hierarchical-clustering` - Hierarchical clustering
- :ref:`spectral-clustering` - Spectral clustering

.. _see-also:

See Also
========

- :ref:`clustering-overview` - Overview of clustering algorithms
- :ref:`performance-comparison` - Performance comparison
- :ref:`tutorial-clustering` - Clustering tutorial
```

### Note and Warning Blocks
```rst
.. note::
   This algorithm requires the data to be normalized for best results.
   Use :func:`preprocessing.scale` to normalize your data.

.. warning::
   The current implementation is not suitable for very large datasets
   (> 1M samples). For large datasets, consider using distributed
   clustering or sampling-based approaches.

.. tip::
   For better performance, ensure your data is in contiguous memory
   layout and aligned to cache line boundaries.
```

## Common Pitfalls to Avoid

1. **Documentation Drift**: Don't let documentation become outdated
2. **Incomplete Examples**: Don't provide incomplete code snippets
3. **Poor Organization**: Don't create deep nesting without clear navigation
4. **Missing Context**: Don't assume user knowledge
5. **Inconsistent Style**: Don't mix different documentation styles


## Best Practices

1. **Keep Updated**: Maintain documentation with code changes
2. **Test Examples**: Verify all code examples compile and run
3. **User Focus**: Write for the intended audience
4. **Clear Structure**: Organize documentation logically
5. **Regular Review**: Periodically review and update documentation

---

**Note**: Good documentation is crucial for user adoption and understanding. Maintain high quality, accuracy, and usability for all documentation.
