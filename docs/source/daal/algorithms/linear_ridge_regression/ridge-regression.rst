.. Copyright 2019 Intel Corporation
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

.. _ridge:

Ridge Regression
================

.. toctree::
   :glob:
   :maxdepth: 4


The ridge regression method is similar to the least squares procedure
except that it penalizes the sizes of the regression coefficients.
Ridge regression is one of the most commonly used methods to overcome
data multicollinearity.

Details
*******

Let |x_vector| be a vector of input variables and
:math:`y = (y_1, \ldots, y_k)` be the response. For each :math:`j=1, \ldots, k`,
the ridge regression model has the form similar to the linear
regression model [Hoerl70]_, except that the coefficients are estimated by minimizing a
different objective function [James2013]_:

.. math::
  y_j = \beta_{0j} + \beta_{1j}x_1 + \ldots + \beta_{pj}x_p

Here :math:`x_i`, :math:`i=1, \ldots, p`, are referred to as independent
variables, and :math:`y_j` are referred to as dependent variables
or responses.

Training Stage
----------------

Let :math:`(x_{11}, \ldots, x_{1p}, y_{11}, \ldots, y_{1k}), \ldots, (x_{n1}, \ldots, x_{np}, y_{n1}, \ldots, y_{nk})`
be a set of training data, :math:`n \gg p`. The matrix :math:`X` of size :math:`n \times p`
contains observations :math:`x_ij`, :math:`i=1, \ldots, n`, :math:`j=1, \ldots, p`, of
independent variables.

For each :math:`y_j`, :math:`j = 1, \ldots, k`, the ridge regression estimates
:math:`(\beta_{0j}, \beta_{1j}, \ldots, \beta_{pj})` by minimizing the
objective function:

.. math::
  \sum_{i=1}^{n} (y_{ij} - \beta_{0j} - \sum_{q=1}^{p} (\beta_{qj}x_{iq}))^2 + \lambda_j \sum_{q=1}^{p} \beta_{qj}^2

where :math:`λ_j \geq 0` are ridge parameters [Hoerl70]_, [James2013]_.


Prediction Stage
----------------

Ridge regression based prediction is done for input vector |x_vector| using the
equation :math:`y_j = \beta_{0j} + \beta_{1j}x_1 + \ldots + \beta_{pj}x_p` for each :math:`j=1, \ldots, k`.
