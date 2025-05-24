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

Min-max
=======

Min-max normalization is an algorithm to linearly scale the observations by each feature (column) into the range :math:`[a, b]`.

Problem Statement
*****************

Given a set :math:`X` of :math:`n` feature vectors :math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})`
of dimension :math:`p`, the problem is to compute the matrix :math:`Y = (y_{ij})_{n \times p}` where the :math:`j`-th column
:math:`(Y)_j = (y_{ij})_{i = 1, \ldots, n}` is obtained as a result of normalizing the column
:math:`(X)_j = (x_{ij})_{i = 1, \ldots, n}`  of the original matrix as:

.. math::

    y_{ij} = a + \frac {x_{ij} - \min(j)}{\max(j) - \min(j)} (b-a),

where:

.. math::
    \min(j) = \min _{i = 1, \ldots, n} x_{ij},

.. math::
    \max(j) = \max _{i = 1, \ldots, n} x_{ij},

:math:`a` and :math:`b` are the parameters of the algorithm.

Batch Processing
****************

Algorithm Input
---------------

The min-max normalization algorithm accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Min-max (Batch Processing)
   :widths: 10 60
   :header-rows: 1

   * - Input ID
     - Input
   * - ``data``
     - Pointer to the numeric table of size :math:`n \times p`.

       .. note:: This table can be an object of any class derived from ``NumericTable``.

Algorithm Parameters
--------------------

The min-max normalization algorithm has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for Min-max (Batch Processing)
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
   * - ``lowerBound``
     - :math:`0.0`
     - The lower bound of the range to which the normalization scales values of the features.
   * - ``upperBound``
     - :math:`1.0`
     - The upper bound of the range to which the normalization scales values of the features.
   * - ``moments``
     - `SharedPtr<low_order_moments::Batch<algorithmFPType, low_order_moments::defaultDense> >`
     - Pointer to the low order moments algorithm that computes minimums and maximums
       to be used for min-max normalization with the defaultDense method.
       For more details, see :ref:`Batch Processing for Moments of Low Order <moments_batch>`.

Algorithm Output
----------------

The min-max normalization algorithm calculates the result described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
For more details, see ``Algorithms``.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Min-max (Batch Processing)
   :widths: 10 60
   :header-rows: 1

   * - Result ID
     - Result
   * - ``normalizedData``
     - Pointer to the :math:`n \times p` numeric table that stores the result of normalization.

       .. note::

          By default, the result is an object of the ``HomogenNumericTable`` class,
          but you can define the result as an object of any class derived from ``NumericTable``
          except ``PackedTriangularMatrix``, ``PackedSymmetricMatrix``, and ``CSRNumericTable``.

Examples
********

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`minmax_dense_batch.cpp <normalization/minmax_dense_batch.cpp>`

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`normalization_minmax.py`
