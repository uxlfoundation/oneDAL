.. Copyright contributors to the oneDAL project
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

.. default-domain:: cpp

.. _alg_correlation_distance:

====================
Correlation Distance
====================

.. include::  ../../../includes/pairwise-distances/correlation-distance-introduction.rst

------------------------
Mathematical formulation
------------------------

.. _correlation_distance_c_math:

Computing
---------

Given a set :math:`X` of :math:`n` feature vectors :math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})`
of dimension :math:`p` and a set :math:`Y` of :math:`m`
feature vectors :math:`y_1 = (y_{11}, \ldots, y_{1p}), \ldots, y_m = (y_{m1}, \ldots, y_{mp})`,
the problem is to compute the correlation distance :math:`D(x_i, y_j)` for any pair of input vectors:

.. math::
   D(x_i, y_j) = 1 - \frac{\sum_{k=1}^{p}(x_{ik} - \bar{x}_i)(y_{jk} - \bar{y}_j)}{\sqrt{\sum_{k=1}^{p}(x_{ik} - \bar{x}_i)^2} \sqrt{\sum_{k=1}^{p}(y_{jk} - \bar{y}_j)^2}}

where :math:`\bar{x}_i = \frac{1}{p}\sum_{k=1}^{p}x_{ik}` and :math:`\bar{y}_j = \frac{1}{p}\sum_{k=1}^{p}y_{jk}` are the means of vectors :math:`x_i` and :math:`y_j`, respectively.

.. _correlation_distance_c_math_dense:

Computation method: *dense*
---------------------------
The method computes the correlation distance matrix :math:`D = D(X, Y), D \in \mathbb{R}^{n \times m}` for
dense :math:`X` and :math:`Y` matrices.

---------------------
Programming Interface
---------------------

Refer to :ref:`API Reference: Correlation Distance <api_correlation_distance>`.
