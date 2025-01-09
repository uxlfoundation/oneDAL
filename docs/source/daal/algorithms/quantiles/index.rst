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

Quantile
========

Quantile is an algorithm to analyze the distribution of observations.
Quantiles are the values that divide the distribution so that a given portion of observations is below the quantile.

Details
*******

Given a set :math:`X` of :math:`p` features :math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots x_n = (x_{n1}, \ldots, x_{np})`
and the quantile orders :math:`\beta = \beta_1, \ldots, \beta_m`,
the problem is to compute :math:`z_{ik}` that meets the following conditions:

.. math::
    P\{ \xi_i \leq z_{ik} \} \geq \beta_k

.. math::
    P\{\xi_i > z_{ik} \} \leq 1 - \beta_k

In the equations above:

- :math:`x_i = (x_{1i}, \ldots, x_{ni})` are observations of a random variable :math:`\xi_i` that represents the :math:`i`-th feature
- :math:`P` is the probability measure
- :math:`i = 1, \ldots, p`
- :math:`k = 1, \ldots, m`

Batch Processing
****************

Algorithm Input
---------------

The quantile algorithm accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Quantile (Batch Processing)
   :widths: 10 60
   :header-rows: 1

   * - Input ID
     - Input
   * - ``data``
     - Pointer to the :math:`n \times p` numeric table that contains the input data set.
       This table can be an object of any class derived from ``NumericTable``.

Algorithm Parameters
--------------------

The quantile algorithm has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for Quantile (Batch Processing)
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
     - Performance-oriented computation method, the only method supported by the algorithm.
   * - ``quantileOrders``
     - :math:`0.5`
     - The :math:`1 \times m` numeric table with quantile orders.

Algorithm Output
----------------

The quantile algorithm calculates the result described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Quantile (Batch Processing)
   :widths: 10 60
   :header-rows: 1

   * - Result ID
     - Result
   * - ``quantiles``
     - Pointer to the :math:`p \times m` numeric table with the quantiles.

       By default, this result is an object of the ``HomogenNumericTable`` class, but you can define the result as an object of any class
       derived from ``NumericTable`` except ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.

Examples
********

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`quantiles_dense_batch.cpp <quantiles/quantiles_dense_batch.cpp>`

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`quantiles.py`
