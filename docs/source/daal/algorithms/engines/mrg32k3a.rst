.. Copyright contributors to the oneDAL project
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

mrg32k3a
========

The engine based on a 32-bit combined multiple recursive generator
with two components of order 3, optimized for batch processing.

.. rubric:: Subsequence selection methods support

skipAhead (nskip)
    Supported
leapfrog (threadIdx, nThreads)
    Supported

Batch Processing
****************

mrg32k3a engine needs the initial condition (``seed``) for state initialization.
The seed can be either an integer scalar or a vector of :math:`p` integer elements, the inputs to the respective engine constructors.

.. rubric:: Algorithm Parameters

mrg32k3a engine has the following parameters:

.. tabularcolumns::  |\Y{0.2}|\Y{0.2}|\Y{0.6}|

.. list-table:: Algorithm Parameters for mcg58 engine (Batch Processing)
   :header-rows: 1
   :widths: 10 20 30
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
     - Performance-oriented computation method; the only method supported by the algorithm.
   * - ``seed``
     -
       - :math:`777` for a scalar seed
       - NA for a vector seed
     - Initial condition for state initialization, scalar or vector:

       - Scalar, value of ``size_t`` type
       - Vector, pointer to ``HomogenNumericTable`` of size :math:`1 \times p`
