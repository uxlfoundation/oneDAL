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

.. _iterative_solver:

Iterative Solver
================

The iterative solver provides an iterative method to minimize an objective function
that can be represented as a sum of functions in composite form

.. math::
    \theta_{*} = \text{argmin}_{\theta \in R^p} K(\theta) =
    \text{argmin}_{\theta \in R^p} F(\theta) + M(\theta)

where:

- :math:`F(\theta) = \sum_{i=1}^{n} F_i(\theta)`, :math:`\theta \in R^p`,
  where :math:`F_i(\theta): R^p \to R` is a convex, continuously differentiable
  :math:`F_i(\theta) \in C^{l \geq 1}` (smooth) functions, :math:`i = 1, \ldots, n`
- :math:`M(\theta): R^p \to R` is a convex, non-differentiable (non-smooth) function

.. rubric:: The Algorithmic Framework of an Iterative Solver

All solvers presented in the library follow a common algorithmic framework.
Let :math:`S_t` be a set of intrinsic parameters of the iterative solver for updating the argument of the objective function.
This set is the algorithm-specific and can be empty. The solver determines the choice of :math:`S_0`.

To do the computations, iterate :math:`t` from :math:`1` until :math:`\text{nIterations}`:

#. Choose a set of indices without replacement :math:`I = \{ i_1, \ldots, i_b \}`, :math:`1 \leq i_j \leq n`,
   :math:`j = 1, \ldots, b`, where :math:`b` is the batch size.

#. Compute the gradient :math:`g(\theta_{t-1}) = \nabla F_I (\theta_{t-1})` where
   :math:`F_I (\theta_{t-1}) = \sum_{i \in I} F_i (\theta_{t-1})`

#. Convergence check:

   Stop if :math:`\frac {{|U|}_d} {\max (1, {|| \theta_{t-1} ||}_d )} < \epsilon` where :math:`U` is an algorithm-specific vector
   (argument or gradient) and d is an algorithm-specific power of `Lebesgue space <https://en.wikipedia.org/wiki/Lp_space>`_

#. Compute :math:`\theta_t` using the algorithm-specific transformation :math:`T` that updates the function's argument:

   .. math::
     \theta_t = T(\theta_{t-1}, g(\theta_{t-1}), S_{t-1})

#. Update :math:`S_t: S_t = U(S_{t-1})`  where :math:`U` is an algorithm-specific update of the set of intrinsic parameters.

The result of the solver is the argument :math:`\theta_{.}` and a set of parameters :math:`S_{.}` after the exit from the loop.

.. note::

    You can resume the computations to get a more precise estimate of the objective function minimum.
    To do this, pass to the algorithm the results :math:`\theta_{.}` and :math:`S_{.}` of the previous run of the optimization solver.
    By default, the solver does not return the set of intrinsic parameters.
    If you need it, set the ``optionalResultRequired`` flag for the algorithm.

.. toctree::
   :maxdepth: 1
   :caption: Iterative solvers

   solvers/computation.rst
   solvers/lbfgs.rst
   solvers/stochastic-gradient-descent-algorithm.rst
   solvers/adaptive-subgradient-method.rst
   solvers/coordinate-descent.rst
   solvers/stochastic-average-gradient-accelerated-method.rst
