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

Pivoted QR Decomposition
========================

Given the matrix :math:`X` of size :math:`n \times p`, the problem is to compute the QR decomposition
with column pivoting :math:`XP = QR`, where

- :math:`Q` is an orthogonal matrix of size :math:`n \times n`
- :math:`R` is a rectangular upper triangular matrix of size :math:`n \times p`
- :math:`P` is a permutation matrix of size :math:`n \times n`

The library requires :math:`n > p`. In this case:

.. math::

    XP = QR = [Q_1, Q_2] \cdot \begin{bmatrix} R_1 \\ 0 \end{bmatrix} = Q_1 R_1

where the matrix :math:`Q_1` has the size :math:`n \times p` and :math:`R_1` has the size :math:`p \times p`.

Batch Processing
****************

Algorithm Input
---------------

Pivoted QR decomposition accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Pivoted QR Decomposition (Batch Processing)
   :widths: 10 60
   :header-rows: 1

   * - Input ID
     - Input
   * - ``data``
     - Pointer to the numeric table that represents the :math:`n \times p` matrix :math:`X` to be factorized.
       The input can be an object of any class derived from ``NumericTable``.

Algorithm Parameters
--------------------

Pivoted QR decomposition has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for Pivoted QR Decomposition (Batch Processing)
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
   * - ``permutedColumns``
     - Not applicable
     - Pointer to the numeric table with the :math:`1 \times p` matrix with the information for the permutation:

       - If the :math:`i`-th element is zero, the :math:`i`-th column of the input matrix is a free column
         and may be permuted with any other free column during the computation.

       - If the :math:`i`-th element is non-zero, the :math:`i`-th column of the input matrix is moved
         to the beginning of XP before the computation and remains in its place during the computation.

       .. note::
           By default, this parameter is an object of the ``HomogenNumericTable`` class, filled by zeros.
           However, you can define this parameter as an object of any class derived from ``NumericTable``
           except the ``PackedSymmetricMatrix`` class, ``CSRNumericTable`` class, and ``PackedTriangularMatrix`` class
           with the ``lowerPackedTriangularMatrix`` layout.

Algorithm Output
----------------

Pivoted QR decomposition calculates the results described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Pivoted QR Decomposition (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :class: longtable

   * - Result ID
     - Result
   * - ``matrixQ``
     - Pointer to the numeric table with the :math:`n \times p` matrix :math:`Q_1`.

       .. note::
           By default, this result is an object of the ``HomogenNumericTable`` class, but you can define the result as an object of any class
           derived from ``NumericTable`` except ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.

   * - ``matrixR``
     - Pointer to the numeric table with the :math:`p \times p` upper triangular matrix :math:`R_1`.

       .. note::
           By default, this result is an object of the ``HomogenNumericTable`` class, but you can define the result as an object of any class
           derived from ``NumericTable`` except the ``PackedSymmetricMatrix`` class, ``CSRNumericTable`` class,
           and ``PackedTriangularMatrix`` class with the ``lowerPackedTriangularMatrix`` layout.

   * - ``permutationMatrix``
     - Pointer to the numeric table with the :math:`1 \times p` matrix such that
       :math:`\text{permutationMatrix}(i) = k` if the column :math:`k` of the full matrix :math:`X` is permuted into the position :math:`i` in :math:`XP`.

       .. note::
           By default, this result is an object of the ``HomogenNumericTable`` class, but you can define the result as an object of any class
           derived from ``NumericTable`` except the ``PackedSymmetricMatrix`` class, ``CSRNumericTable`` class,
           and ``PackedTriangularMatrix`` class with the ``lowerPackedTriangularMatrix`` layout.

Examples
********

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`pivoted_qr_dense_batch.cpp <pivoted_qr/pivoted_qr_dense_batch.cpp>`


  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`pivoted_qr.py`
