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

.. _mse:

Mean Squared Error Algorithm
=============================

.. note:: Mean Squared Error Algorithm is not supported on GPU.

Details
*******

Given :math:`x = (x_{i1}, \ldots, x_{ip}) \in R^p`, a set of feature vectors :math:`i \in \{1, \ldots, n\}`,
and a set of respective responses :math:`y_i`,
the mean squared error (MSE) objective function :math:`F(\theta; x, y)` is a function that has the format:

.. math::
    F(\theta; x, y) = \sum _{i=1}^{n} F_i(\theta; x, y) =
    \frac {1}{2n} \sum _{i=1}^{n} (y_i - h(\theta, x_i))^2

.. math::
    M(\theta) = 0

.. math::
    \mathrm{prox}_\gamma^M (\theta_j) = \theta_j, j = 1, \ldots, p

In |short_name| implementation of the MSE, the :math:`h(\theta, y_i)` is represented as:

.. math::
    h(\theta, y_i) = \theta_0 + \sum _{j=1}^{p} \theta_j x_{ij}


For a given set of the indices :math:`I = \{i_1, i_2, \ldots, i_m\}`,
:math:`1 \leq i_r < n`, :math:`l \in \{1, \ldots, m\}`, :math:`|I| = m`,
the value and the gradient of the sum of functions in the argument :math:`x` respectively have the format:

.. math::
    F_I(\theta; x, y) = \frac {1}{2m} \sum_{i_k \in I} (y_{i_k} - h(\theta, x_{i_k}))^2

.. math::
    \nabla F_I(\theta; x, y) =
    \left\{ \frac{\partial F_I}{\partial \theta_0}, \ldots, \frac{\partial F_I}{\partial \theta_p} \right\}

where

.. math::
    \frac{\partial F_I}{\partial \theta_0} =
    \frac{1}{m} \sum_{i_k \in I} (y_{i_k} - h(\theta, x_{i_k}))

.. math::
    \frac{\partial F_I}{\partial \theta_j} =
    \frac{1}{m} \sum_{i_k \in I} (y_{i_k} - h(\theta, x_{i_k})) x_{i_k j}, j = 1, \ldots, p

:math:`lipschitzConstant = \underset{i = 1, \ldots, n} \max \| x_i \|_2`

Computation
***********

Algorithm Input
---------------

The mean squared error algorithm accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for MSE Computation
   :widths: 10 60
   :align: left
   :class: longtable

   * - Input ID
     - Input
   * - ``argument``
     - A numeric table of size :math:`(p + 1) \times 1` with the input argument :math:`\theta` of the objective function.
   * - ``data``
     - A numeric table of size :math:`n \times p` with the data :math:`x_{ij}`.
   * - ``dependentVariables``
     - A numeric table of size :math:`n \times 1` with dependent variables :math:`y_i`.

Optional Algorithm Input
------------------------

The mean squared error algorithm accepts the optional input described below.
Pass the Optional ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Optional Algorithm Input for MSE Computation
   :widths: 10 60
   :align: left
   :class: longtable

   * - Input ID
     - Input
   * - ``weights``
     - Optional input.
       Pointer to the :math:`1 \times n` numeric table with weights of samples.
       The input can be an object of any class derived from ``NumericTable``
       except for ``PackedTriangularMatrix`` and ``PackedSymmetricMatrix``.

       By default, all weights are equal to :math:`1`.
   * - ``gramMatrix``
     - Optional input.
       Pointer to the :math:`p \times p` numeric table with pre-computed Gram matrix.
       The input can be an object of any class derived from ``NumericTable``
       except for ``PackedTriangularMatrix`` and ``PackedSymmetricMatrix``.

       By default, the table is set to empty numeric table.

Algorithm Parameters
--------------------

The mean squared error algorithm has the following parameters.
Some of them are required only for specific values of the computation method parameter method:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for MSE Computation
   :widths: 10 10 60
   :align: left
   :class: longtable

   * - Parameter
     - Default value
     - Description
   * - ``penaltyL1``
     - :math:`0`
     - The numeric table of size :math:`1 \times \mathrm{nDependentVariables}` with L1 regularized coefficients.
   * - ``penaltyL2``
     - :math:`0`
     - The numeric table of size :math:`1 \times \mathrm{nDependentVariables}` with L2 regularized coefficients.
   * - ``interceptFlag``
     - ``true``
     - Flag to indicate whether or not to compute the intercept.
   * - ``algorithmFPType``
     - ``float``
     - The floating-point type that the algorithm uses for intermediate computations. Can be ``float`` or ``double``.
   * - ``method``
     - ``defaultDense``
     - Performance-oriented computation method.
   * - ``numberOfTerms``
     - Not applicable
     - The number of terms in the objective function.
   * - batchIndices
     - Not applicable
     - The numeric table of size :math:`1 \times m`, where :math:`m` is the batch size, with a batch of indices to be used to compute the function results.
       If no indices are provided, the implementation uses all the terms in the computation.

       .. note::
            This parameter can be an object of any class derived from ``NumericTable``
            except for ``PackedTriangularMatrix`` and ``PackedSymmetricMatrix``.
   * - ``resultsToCompute``
     - ``gradient``
     - The 64-bit integer flag that specifies which characteristics of the objective function to compute.

       Provide one of the following values to request a single characteristic or use bitwise OR to request a combination of the characteristics:

       value
            Value of the objective function
       nonSmoothTermValue
            Value of non-smooth term of the objective function
       gradient
            Gradient of the smooth term of the objective function
       hessian
            Hessian of smooth term of the objective function
       proximalProjection
            Projection of proximal operator for non-smooth term of the objective function
       lipschitzConstant
            Lipschitz constant of the smooth term of the objective function

Algorithm Output
----------------

For the output of the mean squared error algorithm, see :ref:`objective_function_output` for objective functions.

Examples
********

.. tabs::

  .. tab:: C++ (CPU)

    - :cpp_example:`mse_dense_batch.cpp <optimization_solvers/mse_dense_batch.cpp>`
