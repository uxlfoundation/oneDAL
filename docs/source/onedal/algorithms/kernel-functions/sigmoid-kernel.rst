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

.. _alg_sigmoid_kernel:

===================================
Sigmoid kernel
===================================

.. include::  ../../../includes/kernel-functions/sigmoid-kernel-introduction.rst

------------------------
Mathematical formulation
------------------------

.. _sigmoid_kernel_c_math:

Computing
---------

Given a set :math:`X` of :math:`n` feature vectors :math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})`
of dimension :math:`p` and a set :math:`Y` of :math:`m`
feature vectors :math:`y_1 = (y_{11}, \ldots, y_{1p}), \ldots, y_m = (y_{m1}, \ldots, y_{mp})`,
the problem is to compute the sigmoid kernel function :math:`K(x_i, y_j)` for any pair of input vectors:

.. math::
   K(x_i, y_j) = \tanh(k {x_i}^T y_j + b),

where :math:`k\in\mathbb{R},\ b\in\mathbb{R}, \quad 1 \leq i \leq n, \quad 1 \leq j \leq m`.

.. _sigmoid_kernel_c_math_dense:

Computation method: *dense*
---------------------------
The method computes the sigmoid kernel function :math:`Z=K(X, Y), Z \in \mathbb{R}^{n \times m}` for
dense :math:`X` and :math:`Y` matrices.

---------------------
Programming Interface
---------------------

Refer to :ref:`API Reference: Sigmoid kernel <api_sigmoid_kernel>`.
