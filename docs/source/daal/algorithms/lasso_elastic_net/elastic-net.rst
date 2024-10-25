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

.. re-use for math equations:
.. |x_vector| replace:: :math:`(x_1, \ldots, x_p)`
.. |j_1_k| replace:: :math:`j = 1, \ldots, k`

.. _elastic_net:

Elastic Net
===========

Elastic Net is a method for modeling relationship between a dependent variable (which may be a vector)
and one or more explanatory variables by fitting regularized least squares model.
Elastic Net regression model has the special penalty, a sum of L1 and L2 regularizations,
that takes advantage of both :ref:`ridge` and :ref:`LASSO <lasso>` algorithms.
This penalty is particularly useful in a situation with many correlated predictor variables [Friedman2010]_.

Details
*******

Let |x_vector| be a vector of input variables and :math:`y = (y_1, \ldots, y_k)` be the response.
For each |j_1_k|, the Elastic Net model has the form similar to linear and ridge regression models [Hoerl70]_
with one exception: the coefficients are estimated by minimizing mean squared error (MSE) objective function that is
regularized by :math:`L_1` and :math:`L_2` penalties.

.. math::

    y_j = \beta_{0j} + x_1 \beta_{1j} + \ldots + x_p \beta_{pj}

Here :math:`x_i`, :math:`i = 1, \ldots, p`, are referred to as independent variables,
:math:`y_j`, |j_1_k|, is referred to as dependent variable or response.

Training Stage
--------------

Let :math:`(x_{11}, \ldots, x_{1p}, y_{11}, \ldots, y_{1k}) \ldots (x_{n1}, \ldots, x_{np}, y_{n1}, \ldots, y_{nk})` be a set of
training data (for regression task, :math:`n >> p`, and for feature selection :math:`p` could be greater than :math:`n`).
The matrix :math:`X` of size :math:`n \times p` contains observations :math:`x_{ij}`, :math:`i = 1, \ldots, n`,
:math:`j = 1, \ldots, p` of independent variables.

For each :math:`y_j`, :math:`j = 1, \ldots, k`, the Elastic Net regression estimates :math:`(\beta_{0j}, \beta_{1j}, \ldots, \beta_{pj})`
by minimizing the objective function:

.. math::

    F_j(\beta) = \frac{1}{2n} \sum_{i=1}^{n}(y_{ij} - \beta_{0j} - \sum_{q=1}^{p}{\beta_{qj}x_{iq})^2} +
    \lambda_{1j} \sum_{q=1}^{p}|\beta_{qj}| + \lambda_{2j} \frac{1}{2}\sum_{q=1}^{p}\beta_{qj}^{2}

In the equation above, the first term is a mean squared error function, the second and the third
are regularization terms that penalize the :math:`L_1` and :math:`L_2` norms of vector :math:`\beta_j`,
where :math:`\lambda_{1j} \geq 0`, :math:`\lambda_{2j} \geq 0`, |j_1_k|.

For more details, see [Hastie2009]_ and [Friedman2010]_.

By default, :ref:`Coordinate Descent <cda_solver>` iterative solver is used to minimize the objective
function. :ref:`SAGA <saga_solver>` solver is also applicable for minimization.

Prediction Stage
----------------

Prediction based on Elastic Net regression is done for input vector |x_vector| using the equation
:math:`y_j = \beta_{0j} + x_1 \beta_{1j} + \ldots + x_p \beta_{pj}` for each |j_1_k|.
