.. Copyright 2021 Intel Corporation
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

.. _alg_basic_statistics:

================
Basic Statistics
================

.. include::  ../../../includes/statistics/basic-statistics-introduction.rst

------------------------
Mathematical formulation
------------------------

.. _basic_statistics_c_math:

Computing
---------

Given a set :math:`X` of :math:`n` :math:`p`-dimensional feature vectors
:math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})`
and, optionally, a vector :math:`w = (w_1, \ldots, w_n)` of one weight per observation,
the problem is to compute the following sample characteristics for each feature in the data set.
The formulas are stated in terms of the weighted values :math:`\tilde{x}_{ij} = w_i x_{ij}`
defined in :ref:`Weights <basic_statistics_weights>` below; without weights
:math:`\tilde{x}_{ij} = x_{ij}`.

.. list-table::
   :widths: 20 60
   :header-rows: 1
   :align: left

   * - Statistic
     - Definition
   * - Minimum
     - :math:`min(j) = \smash{\displaystyle \min_i } \{\tilde{x}_{ij}\}`
   * - Maximum
     - :math:`max(j) = \smash{\displaystyle \max_i } \{\tilde{x}_{ij}\}`
   * - Sum
     - :math:`s(j) = \sum_i \tilde{x}_{ij}`
   * - Sum of squares
     - :math:`s_2(j) = \sum_i \tilde{x}_{ij}^2`
   * - Means
     - :math:`m(j) = \frac {s(j)} {n}`
   * - Second order raw moment
     - :math:`a_2(j) = \frac {s_2(j)} {n}`
   * - Sum of squared difference from the means
     - :math:`\text{SDM}(j) = \sum_i (\tilde{x}_{ij} - m(j))^2`
   * - Variance
     - :math:`k_2(j) = \frac {\text{SDM}(j) } {n - 1}`
   * - Standard deviation
     - :math:`\text{stdev}(j) = \sqrt {k_2(j)}`
   * - Variation coefficient
     - :math:`V(j) = \frac {\text{stdev}(j)} {m(j)}`

.. _basic_statistics_weights:

Weights
-------

A weight scales the whole observation it belongs to, so every statistic above is the
corresponding statistic of the scaled values

.. math::
   \tilde{x}_{ij} = w_i x_{ij}

When no weights are given, all :math:`w_i` are equal to :math:`1` and
:math:`\tilde{x}_{ij} = x_{ij}`.

Two consequences of this definition are worth spelling out:

- The order statistics are taken over the scaled values, so the maximum of feature :math:`j` is
  :math:`\max_i \{w_i x_{ij}\}` and not :math:`\max_i \{x_{ij}\}`. When the weights differ in
  sign, the observation attaining the maximum is not necessarily the one attaining
  :math:`\max_i \{x_{ij}\}`.
- The weights do not change the number of observations: :math:`n` is the number of feature
  vectors in :math:`X`, not :math:`\sum_i w_i`. The mean is therefore
  :math:`\frac{1}{n} \sum_i w_i x_{ij}` rather than the weighted average
  :math:`\frac{\sum_i w_i x_{ij}}{\sum_i w_i}`.

Weights are optional in the batch, online, and distributed modes of the dense method. The
sparse (CSR) method accepts weights in the online mode, where the weight of a row scales the
values that row stores and leaves its structural zeros at zero.

.. _basic_statistics_p_math:

Partial Computing
-----------------

Given a block of a :math:`X = \{ x_1, \ldots, x_n \}` dataset  with :math:`n` feature vectors of :math:`p` dimension,
and, optionally, the weights :math:`w = (w_1, \ldots, w_n)` of the observations in that block,
the sums is a :math:`1 \times p` matrix, the crossproduct is :math:`p \times p` square matrices.
The sums and the cross product are computed with the following formulas, where
:math:`\tilde{x}_{ij} = w_i x_{ij}` as in :ref:`Weights <basic_statistics_weights>`:

.. list-table::
   :widths: 20 60
   :header-rows: 1
   :align: left

   * - Statistic
     - Definition
   * - Partial Minimum
     - :math:`min(j) = \smash{\displaystyle \min_i } \{\tilde{x}_{ij}\}`
   * - Partial Maximum
     - :math:`max(j) = \smash{\displaystyle \max_i } \{\tilde{x}_{ij}\}`
   * - Partial Sum
     - :math:`s(j) = \sum_i \tilde{x}_{ij}`
   * - Partial Sum of squares
     - :math:`s_2(j) = \sum_i \tilde{x}_{ij}^2`

.. _basic_statistics_f_math:

Finalize Computing
------------------

Given a partial result with partial products,
the means is a :math:`1 \times p` matrix, the covariance and correlation matrices are :math:`p \times p` square matrices.
The means, the covariance, and the correlation are computed with the following formulas, where
:math:`i` runs over all the :math:`n` observations of the blocks accumulated into the partial
result and :math:`\tilde{x}_{ij} = w_i x_{ij}` as in :ref:`Weights <basic_statistics_weights>`:

.. list-table::
   :widths: 20 60
   :header-rows: 1
   :align: left

   * - Statistic
     - Definition
   * - Finalize Minimum
     - :math:`min(j) = \smash{\displaystyle \min_i } \{\tilde{x}_{ij}\}`
   * - Finalize Maximum
     - :math:`max(j) = \smash{\displaystyle \max_i } \{\tilde{x}_{ij}\}`
   * - Finalize Sum
     - :math:`s(j) = \sum_i \tilde{x}_{ij}`
   * - Finalize Sum of squares
     - :math:`s_2(j) = \sum_i \tilde{x}_{ij}^2`
   * - Finalize Means
     - :math:`m(j) = \frac {s(j)} {n}`
   * - Finalize Second order raw moment
     - :math:`a_2(j) = \frac {s_2(j)} {n}`
   * - Finalize Sum of squared difference from the means
     - :math:`\text{SDM}(j) = \sum_i (\tilde{x}_{ij} - m(j))^2`
   * - Finalize Variance
     - :math:`k_2(j) = \frac {\text{SDM}(j) } {n - 1}`
   * - Finalize Standard deviation
     - :math:`\text{stdev}(j) = \sqrt {k_2(j)}`
   * - Finalize Variation coefficient
     - :math:`V(j) = \frac {\text{stdev}(j)} {m(j)}`

.. _basic_statistics_c_math_dense:

Computation method: *dense*
---------------------------
The method computes the basic statistics for each feature in the data set.

---------------------
Programming Interface
---------------------

Refer to :ref:`API Reference: Basic statistics <api_basic_statistics>`.

-----------
Online mode
-----------

The algorithm supports online mode.

----------------
Distributed mode
----------------

The algorithm supports distributed execution in SPMD mode (only on GPU).

-------------
Usage Example
-------------

.. include:: ../../../includes/statistics/basic-statistics-usage-examples.rst

--------
Examples
--------

.. include:: ../../../includes/statistics/basic-statistics-examples.rst
