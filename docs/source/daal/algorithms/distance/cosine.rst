.. Copyright 2020 Intel Corporation
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

Cosine Distance Matrix
======================

Given :math:`n` feature vectors :math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots x_n = (x_{n1}, \ldots, x_{np})`
of dimension :math:`p`,
the problem is to compute the symmetric :math:`n \times n` matrix :math:`D_{\text{cos}} = (d_{ij})`
of distances between feature vectors, where

.. math::

    d_{ij} = 1 - \frac
    {\sum_{k=1}^{p} x_{ik} x_{jk}}
    {\sqrt{ \sum_{k=1}^{p} x_{ik}^2 }
    \sqrt{ \sum_{k=1}^{p} x_{jk}^2 }}

.. math::
    i = \overline{1, n}

.. math::
    j = \overline{1, n}


Batch Processing
****************

Algorithm Input
---------------

The cosine distance matrix algorithm accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Cosine Distance Matrix (Batch Processing)
   :widths: 10 60
   :header-rows: 1

   * - Input ID
     - Input
   * - ``data``
     - Pointer to the :math:`n \times p` numeric table for which the distance is computed.

       The input can be an object of any class derived from ``NumericTable``.

Algorithm Parameters
--------------------

The cosine distance matrix algorithm has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for Cosine Distance Matrix (Batch Processing)
   :header-rows: 1
   :widths: 10 10 60
   :align: left
   :class: longtable

   * - Parameter
     - Default Value
     - Description
   * - ``algorithmFPType``
     - ``float``
     - The floating-point type that the algorithm uses for intermediate computations. Can be ``float`` or ``double``.
   * - ``method``
     - ``defaultDense``
     - Performance-oriented computation method, the only method supported by the algorithm.

Algorithm Output
----------------

The cosine distance matrix algorithm calculates the result described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Cosine Distance Matrix (Batch Processing)
   :widths: 10 60
   :header-rows: 1

   * - Result ID
     - Result
   * - ``cosineDistance``
     - Pointer to the numeric table that represents the :math:`n \times n` symmetric distance matrix :math:`D_\text{cos}`.

       By default, the result is an object of the ``PackedSymmetricMatrix`` class with the ``lowerPackedSymmetricMatrix`` layout.
       However, you can define the result as an object of any class derived from ``NumericTable`` except ``PackedTriangularMatrix`` and ``CSRNumericTable``.

Examples
********

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`cos_dist_dense_batch.cpp <distance/cos_dist_dense_batch.cpp>`

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`cosine_distance.py`

Performance Considerations
**************************

To get the best overall performance when computing the cosine distance matrix:

- If input data is homogeneous, provide the input data and store results in homogeneous numeric tables
  of the same type as specified in the ``algorithmFPType`` class template parameter.
- If input data is non-homogeneous, use :ref:`AOS <aos_layout>` layout rather than SOA layout.

.. include:: ../../../opt-notice.rst
