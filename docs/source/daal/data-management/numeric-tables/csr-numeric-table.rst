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

CSR Numeric Table
=================

.. toctree::
   :maxdepth: 1
   :hidden:

|product| offers the ``CSRNumericTable`` class for a special version
of a homogeneous numeric table that encodes sparse data, that is,
the data with a significant number of zero elements. The library
uses the Condensed Sparse Row (CSR) format for encoding:

.. figure:: ./images/zero-based-csr.png
  :width: 600
  :alt:

  Condensed Sparse Row (CSR) 0-Based Encoding

.. figure:: ./images/one-based-csr.png
  :width: 600
  :alt:

  Condensed Sparse Row (CSR) 1-Based Encoding

Three arrays describe the sparse matrix M as follows:

- The array values contains non-zero elements of the matrix row-by-row.
- The j-th element of the array columns encodes the column index in
  the matrix M for the j-th element of the array values.
- The i-th element of the array rowIndex encodes the index in the
  array values corresponding to the first non-zero element in rows
  indexed i or greater. The last element in the array rowIndex
  encodes the number of non-zero elements in the matrix M.

The library supports 1-based CSR encoding only. In C++ you can
specify it by providingoneBased value through the indexing parameter
of type ``CSRIndexing`` in the constructor of ``CSRNumericTable``.

Examples
********

C++: :cpp_example:`datasource/datastructures_csr.cpp`

.. Python*: datastructures_csr.py
