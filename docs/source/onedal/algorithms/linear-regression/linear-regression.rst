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
---------

Given a :math:`(x_{11}, \ldots, x_{1p}, y_1, \ldots, x_{n1}, \ldots, x_{np}, y_n)` set  of
training data, :math:`n \gg p`. The :math:`X` matrix of the :math:`n \times p` size contains
:math:`x_{ij}`, :math:`i=1, \ldots, n`, :math:`j = 1, \ldots, p` observations of independent
variables.

To estimate the coefficients :math:`(\beta_{0j}, \ldots, \beta_{pj})`,
you can use the Normal Equation method.

.. _linreg_p_math:

Partial Training
----------------

Given a :math:`(x_{11}, \ldots, x_{1p}, y_1, \ldots, x_{n1}, \ldots, x_{np}, y_n)` block of
training data, :math:`n \gg p`. The :math:`X` matrix of :math:`n \times p` size contains
:math:`x_{ij}`, :math:`i=1, \ldots, n`, :math:`j = 1, \ldots, p` observations of independent
variables.

To compute a cross product of Linear Regression, you can use the Normal Equation method.

.. _linreg_f_math:

Finalize Training
-----------------

Given a partial result with partial products.

To compute a cross product of Linear Regression, you can use the Normal Equation method.


.. _linreg_i_math:

Inference
---------
The linear regression-based prediction is made for the input vector
using the :math:`y_j = \beta_{0j} + \beta_{1j}x_1 + \ldots + \beta_{pj}x_p` equation
for each :math:`j=1, \ldots, k`.

.. _norm_eq:

Computation Method: *norm_eq*
-----------------------------
The method computes means, variance-covariance, or correlation matrix for the dense data.
This is the default and the only supported method.

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

The algorithm supports distributed execution in SPMD mode (only on GPU).
