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

.. _alg_cosine_distance:

===============
Cosine distance
===============

.. include::  ../../../includes/pairwise-distances/cosine-distance-introduction.rst

------------------------
Mathematical formulation
------------------------

.. _cosine_distance_c_math:

Computing
---------

Given a set :math:`U` of :math:`n` feature vectors :math:`u_1 = (u_{11}, \ldots, u_{1k}),
\ldots, u_n = (u_{n1}, \ldots, u_{nk})` of dimension :math:`k` and a set :math:`V` of :math:`m`
feature vectors :math:`v_1 = (v_{11}, \ldots, v_{1k}), \ldots, v_m = (v_{m1}, \ldots, v_{mk})`
of dimension :math:`k`, the problem is to compute the Cosine distance
:math:`D_{cos}(u_i, v_j)` for any pair of input vectors:

.. math::
    D_{cos}(u_i, v_j) =  1 - \frac{\sum_{l=1}^{k}u_{il}v_{jl}}{\sqrt{\sum_{l=1}^{k}u_{il}^{2}}\sqrt{\sum_{l=1}^{k}v_{jl}^{2}}},

where :math:`\quad 1 \leq i \leq n, \quad 1 \leq j \leq m`.

.. _cosine_distance_c_math_dense:

Computation method: *dense*
---------------------------
The method defines Cosine distance metric, which is used in other algorithms
for the distance computation. There is no separate computation mode to compute the distance
manually.

---------------------
Programming Interface
---------------------

Refer to :ref:`API Reference: Cosine distance <api_cosine_distance>`.
