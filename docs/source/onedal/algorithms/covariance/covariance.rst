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

.. _alg_covariance:

================
Covariance
================

.. include::  ../../../includes/covariance/covariance-introduction.rst

------------------------
Mathematical formulation
------------------------

.. _covariance_c_math:

Computing
---------

Given a dataset :math:`X = \{ x_1, \ldots, x_n \}` with :math:`n` feature vectors of dimension :math:`p`,
the means is a :math:`1 \times p` matrix, the covariance and the correlation matrices are :math:`p \times p` square matrices.
The means, the covariance, and the correlation are computed with the following formulas:

.. list-table::
   :widths: 20 50
   :header-rows: 1
   :align: left

   * - Statistic
     - Definition
   * - Means
     - :math:`M = (m_{1}, \ldots , m_{p})`, where :math:`m_{j} = \frac{1}{n}\sum _{i}{x}_{ij}`
   * - Covariance matrix (sample)
     - :math:`Cov = (v_{ij})`, where :math:`v_{ij} = \frac{1}{n-1}\sum_{k=1}^{n}(x_{ki}-m_{i})(x_{kj}-m{j})`, :math:`i = \overline{1,p}`, :math:`j = \overline{1,p}`
   * - Covariance matrix (maximum likelihood)
     - :math:`Cov' = (v'_{ij})`, where :math:`v'_{ij} = \frac{1}{n}\sum_{k=1}^{n}(x_{ki}-m_{i})(x_{kj}-m{j})`, :math:`i = \overline{1,p}`, :math:`j = \overline{1,p}`
   * - Correlation matrix
     - :math:`Cor = (c_{ij})`, where :math:`c_{ij} = \frac{v_{ij}}{\sqrt{v_{ii}\cdot v_{jj}}}`, :math:`i = \overline{1,p}`, :math:`j = \overline{1,p}`

.. _covariance_p_math:

Partial Computing
-----------------

Given a block of a :math:`X = \{ x_1, \ldots, x_n \}` dataset with :math:`n` feature vectors of :math:`p` dimension,
the sums is a :math:`1 \times p` matrix, the cross product is :math:`p \times p` square matrices.
The sums and cross product are computed with the following formulas:

.. list-table::
   :widths: 20 50
   :header-rows: 1
   :align: left

   * - Statistic
     - Definition
   * - Sums
     - :math:`S = (m_{1}, \ldots , m_{p})`, where :math:`m_{j} = sum _{i}{x}_{ij}`
   * - Cross product matrix
     - :math:`Crossproduct = (v_{ij}) = \sum_{k=1}^{n}(x_{ki}-m_{i})(x_{kj}-m{j})`, :math:`i = \overline{1,p}`, :math:`j = \overline{1,p}`

.. _covariance_f_math:

Finalize Computing
------------------

Given a partial result with partial products,
the means is a :math:`1 \times p` matrix, the covariance and correlation matrices are :math:`p \times p` square matrices.
The means, the covariance, and the correlation are computed with the following formulas:

.. list-table::
   :widths: 20 50
   :header-rows: 1
   :align: left

   * - Statistic
     - Definition
   * - Means
     - :math:`M = (m_{1}, \ldots , m_{p})`, where :math:`m_{j} = \frac{1}{n}\sum _{i}{x}_{ij}`
   * - Covariance matrix (sample)
     - :math:`Cov = (v_{ij})`, where :math:`v_{ij} = \frac{1}{n-1}\sum_{k=1}^{n}(x_{ki}-m_{i})(x_{kj}-m{j})`, :math:`i = \overline{1,p}`, :math:`j = \overline{1,p}`
   * - Covariance matrix (maximum likelihood)
     - :math:`Cov' = (v'_{ij})`, where :math:`v'_{ij} = \frac{1}{n}\sum_{k=1}^{n}(x_{ki}-m_{i})(x_{kj}-m{j})`, :math:`i = \overline{1,p}`, :math:`j = \overline{1,p}`
   * - Correlation matrix
     - :math:`Cor = (c_{ij})`, where :math:`c_{ij} = \frac{v_{ij}}{\sqrt{v_{ii}\cdot v_{jj}}}`, :math:`i = \overline{1,p}`, :math:`j = \overline{1,p}`

.. _covariance_c_math_dense:

Computation method: *dense*
---------------------------
The method computes means, variance-covariance, or correlation matrix for the dense data.
This is the default and the only method supported.

---------------------
Programming Interface
---------------------

Refer to :ref:`API Reference: Covariance <api_covariance>`.

-----------
Online mode
-----------

The algorithm supports online mode.

----------------
Distributed mode
----------------

The algorithm supports distributed execution in SPMD mode (only on GPU).
