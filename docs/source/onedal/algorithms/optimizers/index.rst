.. Copyright 2023 Intel Corporation
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

==========
Optimizers
==========

This chapter describes optimizers implemented in |short_name|.

.. _alg_newton_cg:

===================
Newton-CG Optimizer
===================

.. include::  ../../../includes/optimizers/newton-cg-introduction.rst

------------------------
Mathematical Formulation
------------------------

.. _newton_cg_c_math:

Computing
---------

The Newton-CG optimizer, also known as the hessian-free optimizer, minimizes convex functions
without calculating the Hessian matrix. Instead, it uses a Hessian product matrix operator.
In the Newton method, the descent direction is calculated using the formula :math:`d_k = -H_k^{-1} g_k`,
where :math:`g_k, H_k` are the gradient and hessian matrix of the loss function on the :math:`k`-th iteration.
The Newton-CG method uses the Conjugate Gradients solver to find the approximate solution to the equation :math:`H_k d_k = -g_k`.
This solver can find solutions to the system of linear equations :math:`Ax = b` taking
vector :math:`b` and functor :math:`f(p) = Ap` as input.


.. _newton_cg_c_math_dense:

Computation Method: *dense*
---------------------------
The method defines the Newton-CG optimizer used by other algorithms
for convex optimization. There is no separate computation mode to minimize a
function manually.

---------------------
Programming Interface
---------------------

Refer to :ref:`API Reference: Newton-CG optimizer <api_newton_cg>`.

