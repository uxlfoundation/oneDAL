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

Batch Processing
================

Input
*****

Initialization of item factors for the implicit ALS algorithm accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Input for Implicit Alternating Least Squares Initialization (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :align: left

   * - Input ID
     - Input
   * - ``data``
     - Pointer to the :math:`m \times n` numeric table with the mining data.

       The input can be an object of any class derived from ``NumericTable`` except
       ``PackedTriangularMatrix`` and ``PackedSymmetricMatrix``.

Parameters
**********

Initialization of item factors for the implicit ALS algorithm has the following parameters:

.. tabularcolumns::  |\Y{0.2}|\Y{0.2}|\Y{0.6}|

.. list-table:: Parameters for Implicit Alternating Least Squares Initialization (Batch Processing)
   :widths: 10 20 30
   :header-rows: 1
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
     - Available computation methods:

       + ``defaultDense`` - performance-oriented method
       + ``fastCSR`` - performance-oriented method for CSR numeric tables

   * - ``nFactors``
     - :math:`10`
     - The total number of factors.
   * - ``engine``
     - `SharePtr< engines:: mt19937:: Batch>()`
     - Pointer to the random number generator engine that is used internally at the initialization step.

Output
******

Initialization of item factors for the implicit ALS algorithm calculates the result described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Output for Implicit Alternating Least Squares Initialization (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :align: left

   * - Result ID
     - Result
   * - ``model``
     - The model with initialized item factors. The result can only be an object of the ``Model`` class.
