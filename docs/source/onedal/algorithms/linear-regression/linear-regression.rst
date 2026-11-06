.. Copyright 2024 Intel Corporation
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

.. highlight:: cpp
.. default-domain:: cpp

.. _alg_linear-regression:

=================
Linear Regression
=================

.. include::  ../../../includes/linear-regression/linear-regression-introduction.rst

------------------------
Mathematical Formulation
------------------------

.. _linreg_t_math:

Training
--------

Given a :math:`(x_{11}, \ldots, x_{1p}, y_{11}, \ldots, y_{1r}, \ldots, x_{n1}, \ldots, x_{np}, y_{n1}, \ldots, y_{nr})` set  of
training data and the regularization parameter :math:`\alpha`. The :math:`\mathbf{X}` matrix of the :math:`n \times p` size contains
:math:`x_{ij}`, :math:`i=1, \ldots, n`, :math:`j = 1, \ldots, p` observations of independent
variables, where :math:`n` is the number of observations and :math:`p` is the number of features.
The :math:`\mathbf{y}` matrix of the :math:`n \times r` size contains :math:`y_{ij}`, :math:`i=1, \ldots, n`,
:math:`j = 1, \ldots, r` observations of dependent variables, where :math:`r` is the number of response variables.

To estimate the coefficients :math:`(\beta_{0j}, \ldots, \beta_{pj})`,
you can use the Normal Equation method.

.. _linreg_p_math:

Partial Training
----------------

Given a :math:`(x_{11}, \ldots, x_{1p}, y_{11}, \ldots, y_{1r}, \ldots, x_{n1}, \ldots, x_{np}, y_{n1}, \ldots, y_{nr})` block of
training data. The partial results of the algorithm are computed with the following formulas:

- When the intercept terms :math:`(\beta_{0*})` are not included:

.. list-table::
   :widths: 20 50
   :header-rows: 1
   :align: left

   * - Partial result
     - Definition
   * - :math:`\mathbf{X}^T \mathbf{X}`
     - :math:`(\mathbf{X}^T \mathbf{X})_{ij} = \sum_{k=1}^{n}(x_{ki} x_{kj})`, :math:`i = 1, \ldots, p`, :math:`j = 1, \ldots, p`
   * - :math:`\mathbf{X}^T \mathbf{y}`
     - :math:`(\mathbf{X}^T \mathbf{y})_{ij} = \sum_{k=1}^{n}(x_{ki} y_{kj})`, :math:`i = 1, \ldots, p`, :math:`j = 1, \ldots, r`

- When the intercept terms :math:`(\beta_{0*})` are included:

.. list-table::
   :widths: 20 50
   :header-rows: 1
   :align: left

   * - Partial result
     - Definition
   * - :math:`\tilde{\mathbf{X}}^T \tilde{\mathbf{X}}`
     - | :math:`(\tilde{\mathbf{X}}^T \tilde{\mathbf{X}})_{ij} = \sum_{k=1}^{n}(x_{ki} x_{kj})`, :math:`i = 1, \ldots, p`, :math:`j = 1, \ldots, p`
       | :math:`(\tilde{\mathbf{X}}^T \tilde{\mathbf{X}})_{i,p+1} = \sum_{k=1}^{n} x_{ki}`, :math:`i = 1, \ldots, p`
       | :math:`(\tilde{\mathbf{X}}^T \tilde{\mathbf{X}})_{p+1,i} = \sum_{k=1}^{n} x_{ki}`, :math:`i = 1, \ldots, p`
       | :math:`(\tilde{\mathbf{X}}^T \tilde{\mathbf{X}})_{p+1,p+1} = n`
   * - :math:`\tilde{\mathbf{X}}^T \mathbf{y}`
     - | :math:`(\tilde{\mathbf{X}}^T \mathbf{y})_{ij} = \sum_{k=1}^{n}(x_{ki} y_{kj})`, :math:`i = 1, \ldots, p`, :math:`j = 1, \ldots, r`
       | :math:`(\tilde{\mathbf{X}}^T \mathbf{y})_{p+1,j} = \sum_{k=1}^{n} y_{kj}`, :math:`j = 1, \ldots, r`

where :math:`\tilde{\mathbf{X}}` is the :math:`\mathbf{X}` matrix augmented with a column of ones for the intercept.

.. _linreg_f_math:

Finalize Training
-----------------

Given a partial result with partial products :math:`\mathbf{X}^T \mathbf{X}` and :math:`\mathbf{X}^T \mathbf{y}` and the regularization parameter :math:`\alpha`,
the resulting regression coefficients :math:`\beta` are computed with the following formula:

:math:`\beta = (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})^{+} \mathbf{X}^T \mathbf{y}`

.. _linreg_i_math:

Inference
---------
The linear regression-based prediction is made for the input vector :math:`(x_1, \ldots, x_p)`
using the :math:`y_j = \beta_{0j} + \beta_{1j}x_1 + \ldots + \beta_{pj}x_p` equation
for each :math:`j = 1, \ldots, r`.

.. _norm_eq:

Training method: *Normal Equation*
++++++++++++++++++++++++++++++++++

* If the number of observations :math:`n` is greater than or equal to the number of features :math:`p`,
  the least squares solution coefficients are computed:

  - When the intercept terms :math:`(\beta_{0*})` are not included:

    The Normal Equation method computes the coefficients :math:`(\beta_{1j}, \ldots, \beta_{pj})` by solving
    the equation :math:`(\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I}) \beta_j = \mathbf{X}^T \mathbf{y}_j`.

  - When the intercept terms :math:`(\beta_{0*})` are included:

    The Normal Equation method computes the coefficients :math:`(\beta_{0j}, \beta_{1j}, \ldots, \beta_{pj})`
    by solving the equation :math:`(\tilde{\mathbf{X}}^T \tilde{\mathbf{X}} + \alpha \mathbf{I}) \beta_j = \tilde{\mathbf{X}}^T \mathbf{y}_j`,
    where :math:`\tilde{\mathbf{X}}` is the :math:`\mathbf{X}` matrix augmented with a column of ones for the intercept.

* If the number of observations :math:`n` is less than the number of features :math:`p`,
  the dual problem is solved to compute the coefficients:

  The intercept term is handled by applying centering to the data and the response variable.

  :math:`(\mathbf{X}_c \mathbf{X}_c^T + \alpha \mathbf{I}) \beta_{dual} = \mathbf{y}_c`,
  where :math:`\mathbf{X}_c` and :math:`\mathbf{y}_c` are the centered data and response variable, respectively.

  :math:`x_{c,ij} = x_{ij} - \frac{1}{n} \sum_{i=1}^{n} x_{ij}`, :math:`i = 1, \ldots, n`, :math:`j = 1, \ldots, p`
  :math:`y_{c,ij} = y_{ij} - \frac{1}{n} \sum_{i=1}^{n} y_{ij}`, :math:`i = 1, \ldots, n`, :math:`j = 1, \ldots, r`.

  The dual coefficients :math:`\beta_{dual}` are then used to compute the primal coefficients :math:`\beta` as

  :math:`\beta = \mathbf{X}_c^T \beta_{dual}`

  and the intercept :math:`\beta_{0j}` is computed as

  :math:`\beta_{0j} = \bar{y}_j - \bar{\mathbf{X}}^T \beta_j` for each :math:`j = 1, \ldots, r`.
  where :math:`\bar{y}_j` is the mean of the original response variable :math:`y_j`, and :math:`\bar{\mathbf{X}}`
  is the mean vector of the original features.

---------------------
Programming Interface
---------------------

Refer to :ref:`API Reference: Linear Regression <api_linear-regression>`.

-----------
Online mode
-----------

The algorithm supports online mode.

----------------
Distributed mode
----------------

The algorithm supports distributed execution in SPMD mode.
