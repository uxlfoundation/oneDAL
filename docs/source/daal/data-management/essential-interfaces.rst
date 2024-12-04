.. Copyright 2019 Intel Corporation
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..     http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

Essential Interfaces for Algorithms
===================================

In addition to Generic Interfaces, more
methods enable interfacing numeric tables with algorithms.

The getDataLayout method provides information about the data layout:

+-----------------------------------+-----------------------------------+
| Data Layout                       | Description                       |
+===================================+===================================+
| soa                               | Structure-Of-Arrays (SOA). Values |
|                                   | of individual data features are   |
|                                   | stored in contiguous memory       |
|                                   | blocks.                           |
+-----------------------------------+-----------------------------------+
| aos                               | Array-Of-Structures (AOS).        |
|                                   | Feature vectors are stored in     |
|                                   | contiguous memory block.          |
+-----------------------------------+-----------------------------------+
| csr_Array                         | Condensed-Sparse-Row (CSR).       |
+-----------------------------------+-----------------------------------+
| lowerPackedSymetricMatrix         | Lower packed symmetric matrix     |
+-----------------------------------+-----------------------------------+
| lowerPackedTriangularMatrix       | Lower packed triangular matrix    |
+-----------------------------------+-----------------------------------+
| upperPackedSymetricMatrix         | Upper packed symmetric matrix     |
+-----------------------------------+-----------------------------------+
| upperPackedTriangularMatrix       | Upper packed triangular matrix    |
+-----------------------------------+-----------------------------------+
| unknown                           | No information about data layout  |
|                                   | or unsupported layout.            |
+-----------------------------------+-----------------------------------+

Rather than access the entire in-memory data set, it is often more
efficient to process it by blocks. The key methods that |product|
algorithms use for per-block data access are ``getBlockOfRows()`` and
``getBlockOfColumnValues()``. The ``getBlockOfRows()`` method accesses a
block of feature vectors, while the ``getBlockOfColumnValues()`` method
accesses a block of values for a given feature. A particular
algorithm uses ``getBlockOfRows()``, ``getBlockOfColumnValues()``, or both
methods to access the data. The efficiency of data access highly
depends on the data layout and on whether the data type of the
feature is natively supported by the algorithm without type
conversions. Refer to the Performance Considerations section in the
description of a particular algorithm for a discussion of the optimal
data layout and natively supported data types.

When the data layout fits the per-block data access pattern and the
algorithm requests the data type that corresponds to the actual data
type, the ``getBlockOfRows()`` and ``getBlockOfColumnValues()`` methods avoid
data copying and type conversion. However, when the layout does not
fit the data access pattern or when type conversion is required, both
methods automatically re-pack and convert data as required.

When dealing with custom or unsupported data layouts, you must
implement NumericTableIface, DenseNumericTableIface interfaces, and
optionally CSRNumericTableIface or PackedNumericTableIface
interfaces.

Some algorithms, such as Moments of Low
Order, compute
basic statistics (minimums, maximums, and so on). The other
algorithms, such as Correlation and Variance-Covariance
Matrices or Principal Component
Analysis, require
some basic statistics on input. To avoid duplicated computation of
basic statistics, |product| provides methods to store and retrieve
basic statistics associated with a given numeric table:
``basicStatistics.set()`` and ``basicStatistics.get()``. The following basic
statistics are computed for each numeric table:

- minimum - minimum
- maximum - maximum
- sum - sum
- sumSquares - sum of squares

.. note::

    The default data type of basic statistics is float.


**Special Interfaces for the HomogenNumericTable and Matrix Classes**

- Use the assign method to initialize elements of a dense
  homogeneous numeric table with a certain value, that is, to set
  all elements of the matrix to zero.

- Use the operator [] method to access rows of a homogeneous dense
  numeric table.

**Special Interfaces for the PackedTriangularMatrix and PackedSymmetricMatrix Classes**

-   While you can use generic ``getArray()`` and ``setArray()`` methods to
    access the data in a packed format, in algorithms that have
    specific implementations for a packed data layout, you can use
    more specific ``getPackedValues()`` and ``releasePackedValues()`` methods.

**Special Interfaces for the CSRNumericTable Class**

-   To access three CSR arrays (values , columns, and rowIndex), use
    ``getArrays()`` and ``setArrays()`` methods instead of generic ``getArray()``
    and ``setArray()`` methods. For details of the arrays, see CSR data
    layout.

-   Similarly, in algorithms that have specific implementations for
    the CSR data layout, you can use more specific
    ``getBlockOfCSRValues()`` and ``releaseBlockOfCSRValues()`` methods.

**Special Interfaces for the MergedNumericTable Class**

- To add a new array to the object of the MergedNumericTable class, use the ``addNumericTable()`` method.


