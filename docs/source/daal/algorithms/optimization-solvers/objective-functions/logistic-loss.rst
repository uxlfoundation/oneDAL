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

.. _logistic_loss:

Logistic Loss
=============

Logistic loss is an objective function being minimized in the process of
logistic regression training when a dependent variable takes only one of two values, :math:`0` and :math:`1`.

Details
*******

Given :math:`n` feature vectors :math:`X = \{x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np}) \}`
of :math:`n` :math:`p`-dimensional feature vectors, a vector of class labels :math:`y = (y_1, \ldots, y_n)`,
where :math:`y_i \in \{0, 1\}` describes the class to which the feature vector :math:`x_i` belongs,
the logistic loss objective function :math:`K(\theta, X, y)`
has the following format :math:`K(\theta, X, y) = F(\theta, X, y) + M(\theta)`, where

- :math:`F(\theta, X, y)` is defined as

  .. math::
    F(\theta, X, y) = -\frac{1}{n}
    \sum_{i=1}^{n} \left(y_i \ln \left( \frac{1}{1 + e^{-(\theta_0 + \sum_{j=1}^{p}\theta_j x_{ij})}} \right) +
    (1 - y_i) \ln \left( \frac{1}{1 + e^{-(\theta_0 + \sum_{j=1}^{p}\theta_j x_{ij})}} \right) \right) +
    \lambda_2 \sum_{j=1}^{p} \theta_j^2

  with :math:`\sigma(x, \theta) = \frac{1}{1 + e^{-f(z, \theta)}}`,
  :math:`f(z, \theta) = \theta_0 + \sum_{k=1}^{p} \theta_k z_k`,
  :math:`\lambda_1 \geq 0`, :math:`\lambda_2 \geq 0`

- :math:`M(\theta) = \lambda_1 \sum_{j=1}^{p} |\theta_j|`

For a given set of the indices :math:`I = \{i_1, i_2, \ldots, i_m \}`,
:math:`1 \leq i_r \leq n`, :math:`r \in \{1, \ldots, m \}`:

- The value of the sum of functions has the format:

  .. math::
    F_I(\theta, X, y) = -\frac{1}{m}
    \sum_{i \in I} \left( y_i \ln \sigma(x_i, \theta) + (1 - y_i) \ln (1 - \sigma(x_i, \theta)) \right) +
    \lambda_2 \sum_{k=1}^{p} \theta_k^2

- The gradient of the sum of functions has the format:

  .. math::
    \nabla F_I(\theta, x, y) =
    \left\{ \frac{\partial F_I}{\partial \theta_0}, \ldots, \frac{\partial F_I}{\partial \theta_p} \right\},

  where

  .. math::
    \frac{\partial F_I}{\partial \theta_0} =
    \frac{1}{m} \sum_{i \in I} (\sigma(x_i, \theta) - y_i) + 2 \lambda_2 \theta_0,
    \frac{\partial F_I}{\partial \theta_p} =
    \frac{1}{m} \sum_{i \in I} (\sigma(x_i, \theta) - y_i) x_{ij} + 2 \lambda_2 \theta_j,
    j = 1, \ldots, p

:math:`\mathrm{prox}_\gamma^M (\theta_j) = \begin{cases}
\theta_J - \lambda_1 \gamma, & \theta_j > \lambda_1 \gamma\\
0, & |\theta_j| \leq \lambda_1 \gamma\\
\theta_j + \lambda_1 \gamma, & \theta_j < - \lambda_1 \gamma
\end{cases}`

:math:`lipschitzConstant = \underset{i = 1, \ldots, n} \max \| x_i \|_2 + \frac{\lambda_2}{n}`

For more details, see [Hastie2009]_.

Computation
***********

Algorithm Input
---------------

The logistic loss algorithm accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Logitic Loss Computation
   :widths: 10 60
   :align: left
   :class: longtable

   * - Input ID
     - Input
   * - ``argument``
     - A numeric table of size :math:`(p + 1) \times 1` with the input argument :math:`\theta` of the objective function.

       .. note::
            The sizes of the argument, gradient, and hessian numeric tables do not depend on ``interceptFlag``.
            When ``interceptFlag`` is set to ``false``, the computation of :math:`\theta_0` value is skipped,
            but the sizes of the tables should remain the same.
   * - ``data``
     - A numeric table of size :math:`n \times p` with the data :math:`x_ij`.

       .. note:: This parameter can be an object of any class derived from ``NumericTable``.
   * - ``dependentVariables``
     - A numeric table of size :math:`n \times 1` with dependent variables :math:`y_i`.

       .. note::
           This parameter can be an object of any class derived from ``NumericTable``,
           except for ``PackedTriangularMatrix`` , ``PackedSymmetricMatrix`` , and ``CSRNumericTable``.

Algorithm Parameters
--------------------

The logistic loss algorithm has the following parameters.
Some of them are required only for specific values of the computation method's parameter ``method``:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for Logitic Loss Computation
   :widths: 10 10 60
   :align: left
   :class: longtable

   * - Parameter
     - Default value
     - Description
   * - ``algorithmFPType``
     - ``float``
     - The floating-point type that the algorithm uses for intermediate computations. Can be ``float`` or ``double``.
   * - ``method``
     - ``defaultDense``
     - Performance-oriented computation method.
   * - ``numberOfTerms``
     - Not applicable
     - The number of terms in the objective function.
   * - ``batchIndices``
     - Not applicable
     - The numeric table of size :math:`1 \times m`, where :math:`m` is the batch size,
       with a batch of indices to be used to compute the function results.
       If no indices are provided, the implementation uses all the terms in the computation.

       .. note::
            This parameter can be an object of any class derived from ``NumericTable``
            except ``PackedTriangularMatrix`` and ``PackedSymmetricMatrix`` .
   * - ``resultsToCompute``
     - ``gradient``
     - The 64-bit integer flag that specifies which characteristics of the objective function to compute.

       Provide one of the following values to request a single characteristic or use bitwise OR
       to request a combination of the characteristics:

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

   * - ``interceptFlag``
     - ``true``
     - A flag that indicates a need to compute :math:`\theta_{0j}`.
   * - ``penaltyL1``
     - :math:`0`
     - L1 regularization coefficient
   * - ``penaltyL2``
     - :math:`0`
     - L2 regularization coefficient

Algorithm Output
----------------

For the output of the logistic loss algorithm, see :ref:`objective_function_output` for objective functions.

Examples
********

.. tabs::

    .. tab:: C++ (CPU)

        - :cpp_example:`sgd_log_loss_dense_batch.cpp <optimization_solvers/sgd_log_loss_dense_batch.cpp>`
