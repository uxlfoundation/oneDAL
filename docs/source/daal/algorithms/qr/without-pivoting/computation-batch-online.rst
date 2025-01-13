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

Batch and Online Processing
===========================

Online processing computation mode assumes that the data arrives in blocks :math:`i = 1, 2, 3, \ldots \text{nblocks}`.

Algorithm Input
---------------

QR decomposition accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for QR Decomposition without Pivoting (Batch and Online Processing)
   :widths: 10 60
   :header-rows: 1

   * - Input ID
     - Input
   * - ``data``
     - Pointer to the numeric table that represents:

       - For batch processing: the entire :math:`n \times p` matrix :math:`X` to be factorized.
       - For online processing: the :math:`n_i \times p` submatrix of :math:`X` that represents
         the current data block in the online processing mode.
         Note that each current data block must have sufficient size: :math:`n_i > p`.

       The input can be an object of any class derived from ``NumericTable``.

Algorithm Parameters
--------------------

QR decomposition has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for QR Decomposition without Pivoting (Batch and Online Processing)
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

QR decomposition calculates the results described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for QR Decomposition without Pivoting (Batch and Online Processing)
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
