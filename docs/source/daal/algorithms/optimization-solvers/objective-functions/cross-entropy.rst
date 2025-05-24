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

.. _cross_entropy_loss:

Cross-entropy Loss
==================

Cross-entropy loss is an objective function minimized in the process of logistic regression training when a dependent variable takes more than two values.

Details
*******

Given :math:`n` feature vectors :math:`X = \{x_1 = (x_{11}, \ldots, x_{1p}),\ldots, x_n = (x_{n1}, \ldots, x_{np}) \}`
of :math:`n` :math:`p`-dimensional feature vectors, a vector of class labels :math:`y = (y_1, \ldots, y_n)`,
where :math:`y_i \in \{0, T-1\}` describes the class, to which the feature vector :math:`x_i` belongs,
where :math:`T` is the number of classes, optimization solver optimizes cross-entropy loss objective function by argument :math:`\theta`,
it is a matrix of size :math:`T \times (p + 1)`. The cross entropy loss objective function :math:`K(\theta, X, y)`
has the following format :math:`K(\theta, X, y) = F(\theta) + M(\theta)` where

- :math:`F(\theta) = -\frac{1}{n} \sum_{i=1}^{n} \log p_{y_i} (x_i, \theta) + \lambda_2 \sum_{t=0}^{T-1} \sum_{j=1}^{p} \theta_{tj}^2`,
  with :math:`p_t(z, \theta) = \frac{e^{f_t (z, \theta)}}{\sum_{i=0}^{K-1} e^{f_i (z, \theta)}}`
  and :math:`f_t (z, \theta) = \theta_{t0} + \sum_{j=1}^{p} \theta_{tj} z_j`, :math:`\lambda_1 \geq 0`, :math:`\lambda_2 \geq 0`
- :math:`M(\theta) = \lambda_1 \sum_{t=0}^{T-1} \sum_{j=1}^{p} |\theta_{tj}|`

For a given set of indices :math:`I = \{i_1, i_2, \ldots, i_m \}`, :math:`1 \leq i_r \leq n`, :math:`r \in \{1, \ldots, m \}`,
the value and the gradient of the sum of functions in the argument X respectively have the format:

.. math::
    F_I (\theta, X, y) = -\frac{1}{m} \sum_{i \in I}
    (\log p_{y_i} (x_i, \theta) + \lambda_2 \sum_{t=0}^{T-1} \sum_{j=1}^{p} \theta_{ij}^2)

.. math::
    \nabla F_I(\theta, x, y) =
    \left( \frac{\partial F_I}{\partial \theta_{00}}, \ldots, \frac{\partial F_I}{\partial \theta_{{T-1}p}} \right)^T

where

.. math::
    \frac{\partial F_I}{\partial \theta_{tj}} =
    \begin{cases}
        \frac{1}{m} \sum_{i \in I} g_t (\theta, x_i, y_i) + L_{tj}(\theta), & j = 0 \\
        \frac{1}{m} \sum_{i \in I} g_t (\theta, x_i, y_i) x_{ij} + L_{tj}(\theta), & j = 0
    \end{cases}

    g_t (\theta, x, y) =
    \begin{cases}
        p_k (x, \theta) - 1, & y = t \\
        p_t (x, \theta), & y \neq t
    \end{cases}

    L_{tj} (\theta) = 2 \lambda_2 \theta_{tj}

    t \in [0, T - 1]

    j \in [0, p]

Hessian matrix is a symmetric matrix of size :math:`S \times S`, where  :math:`S = T \times (p + 1)`

.. math::
    \left[\begin{array}{ccc}
        \frac
            {\partial^2 F_I}
            {\partial \theta_{00} \partial \theta_{00}} &
        \cdots &
        \frac
            {\partial^2 F_I}
        {\partial \theta_{00} \partial \theta_{{T-1} p}} \\
        \vdots & \ddots & \vdots \\
        \frac
            {\partial^2 F_I}
            {\partial \theta_{{T-1} p} \partial \theta_{00}} &
        \cdots &
        \frac
            {\partial^2 F_I}
            {\partial \theta_{{T-1} p} \partial \theta_{{T-1} p}}
    \end{array}\right]

.. math::
    \frac {\partial^2 F_I} {\partial \theta_{tj} \partial \theta_{pq}} =
    \begin{cases}
        \frac{1}{m} \sum_{i \in I} g_{tp} (\theta, x_i, y_i) + 2 \lambda_2,
            & j = 0, q = 0\\
        \frac{1}{m} \sum_{i \in I} g_{tp} (\theta, x_i, y_i) x_{ij},
            & j > 0, q = 0\\
        \frac{1}{m} \sum_{i \in I} g_{tp} (\theta, x_i, y_i) x_{iq},
            & j = 0, q > 0\\
        \frac{1}{m} \sum_{i \in I} g_{tp} (\theta, x_i, y_i) x_{ij} x_{iq},
            & j > 0, q > 0, j \neq q\\
        \frac{1}{m} \sum_{i \in I} g_{tp} (\theta, x_i, y_i) x_{ij} x_{iq} + 2 \lambda_2,
            & j > 0, q > 0, j = q\\\
    \end{cases}

    g_{tp} (\theta, x, y) =
        \begin{cases}
            p_p (x, \theta) (1 - p_t (x, \theta)), & p = t \\
            -p_t (x, \theta) p_p (x, \theta), & p \neq t
        \end{cases}

    t, p \in [0, T-1]

    j, q \in [0, p]

:math:`\mathrm{prox}_\gamma^M (\theta_j) = \begin{cases}
\theta_J - \lambda_1 \gamma, & \theta_j > \lambda_1 \gamma\\
0, & |\theta_j| \leq \lambda_1 \gamma\\
\theta_j + \lambda_1 \gamma, & \theta_j < - \lambda_1 \gamma
\end{cases}`, where :math:`\gamma` is the learning rate

:math:`lipschitzConstant = \underset{i = 1, \ldots, n} \max \| x_i \|_2 + \frac{\lambda_2}{n}`

For more details, see [Hastie2009]_.

Computation
***********

Algorithm Input
---------------

The cross entropy loss algorithm accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Cross-entropy Loss Computation
   :widths: 10 60
   :align: left
   :class: longtable

   * - Input ID
     - Input
   * - ``argument``
     - A numeric table of size :math:`(p + 1) \times \mathrm{nClasses}` with the input argument :math:`\theta` of the objective function.

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

The cross entropy loss algorithm has the following parameters.
Some of them are required only for specific values of the computation method's parameter ``method``:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for Cross-entropy Loss Computation
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
       gradientOverCertainFeature
          Certain component of gradient vector
       hessianOverCertainFeature
          Certain component of hessian diagonal
       proximalProjectionOfCertainFeature
          Certain component of proximal projection

   * - ``interceptFlag``
     - ``true``
     - A flag that indicates a need to compute :math:`\theta_{0j}`.
   * - ``penaltyL1``
     - :math:`0`
     - L1 regularization coefficient
   * - ``penaltyL2``
     - :math:`0`
     - L2 regularization coefficient
   * - ``nClasses``
     - Not applicable
     - The number of classes (different values of dependent variable)

Algorithm Output
----------------

For the output of the cross entropy loss algorithm, see :ref:`objective_function_output` for objective functions.

Examples
********

.. tabs::

    .. tab:: C++ (CPU)

        - :cpp_example:`lbfgs_cr_entr_loss_dense_batch.cpp <optimization_solvers/lbfgs_cr_entr_loss_dense_batch.cpp>`

    .. tab:: Python*

        - :daal4py_example:`lbfgs_cr_entr_loss.py`
