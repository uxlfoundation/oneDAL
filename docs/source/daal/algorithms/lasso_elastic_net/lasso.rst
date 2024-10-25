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

.. _lasso:

Least Absolute Shrinkage and Selection Operator (LASSO)
=======================================================

Least Absolute Shrinkage and Selection Operator (LASSO) is a method
for modeling relationship between a dependent variable (which may be
a vector) and one or more explanatory variables by fitting
regularized least squares model. Trained LASSO model can produce
sparse coefficients due to the use of :math:`L_1` regularization
term. LASSO regression is widely used in feature selection tasks. For
example, in the field of compressed sensing it is used to effectively
identify relevant features associated with the dependent variable
from a few observations with a large number of features. LASSO
regression is also used to overcome multicollinearity of feature
vectors in the training data set.

Details
*******

Let |x_vector| be a vector of input variables and :math:`y = (y_1, \ldots, y_k)` be the response.
For each |j_1_k|, the LASSO model has the form similar to linear and ridge regression model [Hoerl70]_,
except that the coefficients are trained by minimizing a regularized by :math:`L_1` penalty mean squared error (MSE)
objective function.

.. math::

    y_j = \beta_{0j} + x_1 \beta_{1j} + \ldots + x_p \beta_{pj}

Here :math:`x_i`, :math:`i = 1, \ldots, p` are referred to as independent variables,
:math:`y_j` is referred to as dependent variable or response and |j_1_k|.

Training Stage
--------------

Let :math:`(x_{11}, \ldots, x_{1p}, y_{11}, \ldots, y_{1k}) \ldots (x_{n1}, \ldots, x_{np}, y_{n1}, \ldots, y_{nk})` be a set of
training data (for regression task, :math:`n >> p`, and for feature selection :math:`p` could be greater than :math:`n`).
The matrix :math:`X` of size :math:`n \times p` contains observations :math:`x_{ij}`, :math:`i = 1, \ldots, n`,
:math:`j = 1, \ldots, p` of independent variables.

For each :math:`y_j`, |j_1_k|, the LASSO regression estimates :math:`(\beta_{0j}, \beta_{1j}, \ldots, \beta_{pj})`
by minimizing the objective function:

.. math::

    F_j(\beta) = \frac{1}{2n} \sum_{i=1}^{n}(y_{ij} - \beta_{0j} - \sum_{q=1}^{p}{\beta_{qj}x_{iq})^2} +
    \lambda_{1j} \sum_{q=1}^{p}|\beta_{qj}|

In the equation above, the first term is a mean squared error function and the second one is a regularization term that
penalizes the :math:`L_1` norm of vector :math:`\beta_j`

For more details, see [Hastie2009]_.

By default, :ref:`Coordinate Descent <cda_solver>` iterative solver is used to minimize the objective
function. :ref:`SAGA <saga_solver>` solver is also applicable for minimization.

Prediction Stage
----------------

For input vector of independent variables |x_vector|, prediction based on LASSO regression is done using the equation

.. math::

    y_j = \beta_{0j} + x_1 \beta_{1j} + \ldots + x_p \beta_{pj}

where |j_1_k|.
