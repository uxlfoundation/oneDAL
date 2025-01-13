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

.. default-domain:: cpp

.. _alg_rbf_kernel:

===================================
Radial Basis Function (RBF) kernel
===================================

.. include::  ../../../includes/kernel-functions/rbf-kernel-introduction.rst

------------------------
Mathematical formulation
------------------------

.. _rbf_kernel_c_math:

Computing
---------

Given a set :math:`X` of :math:`n` feature vectors :math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})`
of dimension :math:`p` and a set :math:`Y` of :math:`m`
feature vectors :math:`y_1 = (y_{11}, \ldots, y_{1p}), \ldots, y_m = (y_{m1}, \ldots, y_{mp})`,
the problem is to compute the RBF kernel function :math:`K(x_i, y_i)` for any pair of input vectors:

.. math::
   K\left({x}_{i},{y}_{j}\right)=exp\left(-\frac{{\left(\|{x}_{i}-{y}_{j}\|\right)}^{2}}{2{\sigma }^{2}}\right)

.. _rbf_kernel_c_math_dense:

Computation method: *dense*
---------------------------
The method computes the RBF kernel function :math:`Z=K(X, Y), Z \in \mathbb{R}^{n \times m}` for
dense :math:`X` and :math:`Y` matrices.

---------------------
Programming Interface
---------------------

Refer to :ref:`API Reference: Radial Basis Function (RBF) kernel <api_rbf_kernel>`.