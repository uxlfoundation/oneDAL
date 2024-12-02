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

Cholesky Decomposition
======================

Cholesky decomposition is a matrix factorization technique that decomposes a symmetric positive-definite matrix
into a product of a lower triangular matrix and its conjugate transpose.

Because of numerical stability and superior efficiency in comparison with other methods,
Cholesky decomposition is widely used in numerical methods for solving symmetric linear systems.
It is also used in non-linear optimization problems, Monte Carlo simulation, and Kalman filtration.

Details
*******

Given a symmetric positive-definite matrix :math:`X` of size :math:`p \times p`,
the problem is to compute the Cholesky decomposition
:math:`X = {LL}^T`, where :math:`L` is a lower triangular matrix.

Batch Processing
****************

Algorithm Input
---------------

Cholesky decomposition accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Cholesky Decomposition (Batch Processing)
   :widths: 10 60
   :header-rows: 1

   * - Input ID
     - Input
   * - ``data``
     - Pointer to the :math:`p \times p` numeric table that represents
       the symmetric positive-definite matrix :math:`X` for which the Cholesky decomposition is computed.

       The input can be an object of any class derived from ``NumericTable`` that can represent symmetric matrices.
       For example, the ``PackedTriangularMatrix`` class cannot represent a symmetric matrix.

Algorithm Parameters
--------------------

Cholesky decomposition has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for Cholesky Decomposition (Batch Processing)
   :header-rows: 1
   :widths: 10 10 60
   :align: left

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

Cholesky decomposition calculates the result described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Cholesky Decomposition (Batch Processing)
   :widths: 10 60
   :header-rows: 1

   * - Result ID
     - Result
   * - ``choleskyFactor``
     - Pointer to the :math:`p \times p` numeric table that represents the lower triangular matrix :math:`L` (Cholesky factor).

       By default, the result is an object of the ``HomogenNumericTable`` class, but you can define the result as an object of any class
       derived from ``NumericTable`` except the ``PackedSymmetricMatrix`` class, ``СSRNumericTable`` class, and ``PackedTriangularMatrix`` class
       with the ``upperPackedTriangularMatrix`` layout.

Examples
********

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`cholesky_dense_batch.cpp <cholesky/cholesky_dense_batch.cpp>`

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`cholesky.py`


Performance Considerations
**************************

To get the best overall performance when Cholesky decomposition:

- If input data is homogeneous, for input matrix :math:`X` and output matrix :math:`L` use homogeneous numeric tables
  of the same type as specified in the ``algorithmFPType`` class template parameter.
- If input data is non-homogeneous, use :ref:`AOS <aos_layout>` layout rather than SOA layout.

.. include:: ../../../opt-notice.rst
