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

.. _cda_solver:

Coordinate Descent Algorithm
============================

The Coordinate Descent algorithm follows the :ref:`algorithmic framework of iterative solver <iterative_solver>` with one exception:
the default method (``defaultDense``) of Coordinate Descent algorithm is a case of the iterative solver method
with the batch equal to the number of observations in the training data set.

Details
*******

The set of intrinsic parameters :math:`S_t` is empty.
Algorithmic-specific transformation :math:`T`, algorithm-specific vector :math:`U`,
and power :math:`d` of `Lebesgue space <https://en.wikipedia.org/wiki/Lp_space>`_ [Adams2003]_ are defined as follows:

.. math::
    T(\theta_{t-1}, F'(\theta_{t-1}), S_{t-1}, M(\theta_{t-1}))

#. Define the index :math:`j` to update the component of a coefficient as a remainder in the division of the number of current iteration (:math:`t`)
   by the number of features in the training data set (:math:`p`): :math:`j = \mathrm{mod}(t, p)`

   Alternatively, if ``selection`` parameter was set to ``random``, generate :math:`j` randomly.

#. If ``stepLengthSequence`` was not provided by the user, compute the learning rate: :math:`\eta = (F''(\theta_{t-1}))_{jj}`
   (the diagonal element of the Hessian matrix)

#. Update the :math:`j`-th component of vector :math:`\theta`:

   .. math::
        (\theta_t)_j = \mathrm{prox}_{\frac{1}{\eta}}^{M}
        \left( (\theta_{t-1})_j - \frac{1}{\max(\eta, \mathrm{eps})} (F'(\theta_{t-1}))_j\right)

   Note: for example, if a non-smooth term :math:`M = \lambda \sum_{i=1}^{p} |\theta_t|`,
   where :math:`p` is the number of features in the training data set, the objective function should compute prox operator as follows:

   .. math::
        \mathrm{prox}_{\frac{1}{\eta}}^{M} \left( (\theta_{t-1})_j \right) =
        \begin{cases}
            (\theta_{t-1})_j - \lambda \frac{1}{\eta}, & (\theta_{t-1})_j > \lambda \frac{1}{\eta}\\
            0, & |(\theta_{t-1})_j| \leq \lambda \frac{1}{\eta}\\
            (\theta_{t-1})_j + \lambda \frac{1}{\eta}, & (\theta_{t-1})_j < -\lambda \frac{1}{\eta}
        \end{cases}

Convergence check is performed each :math:`p` iterations:

- :math:`U = \theta_t - \theta_{t - \mathrm{nFeatures}}`, :math:`d = \infty`
- For :math:`x \in R^p`, the infinity norm (:math:`d = \infty`) is defined as follows:

.. math::
    |x|_{\infty} = \underset{i \in [0, p]} \max(|x_i|)

Computation
***********

Coordinate Descent algorithm is a special case of an iterative solver.
For parameters, input, and output of iterative solvers, see :ref:`Iterative Solver > Computation <iterative_solver_computation>`.

Algorithm Parameters
--------------------

In addition to the input of a iterative solver, Coordinate Descent algorithm accepts the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for Coordinate Descent Computation
   :widths: 10 10 60
   :header-rows: 1
   :align: left
   :class: longtable

   * - Parameter
     - Default Value
     - Description
   * - ``algorithmFPType``
     - ``float``
     - The floating-point type that the algorithm uses for intermediate computations. Can be ``float`` or ``double``.
   * - ``method``
     - ``defaultDense``
     - Performance-oriented method.
   * - ``engine``
     - `SharePtr< engines:: mt19937:: Batch>()`
     - Pointer to the random number generator engine that is used internally during each iteration
       to choose a random component of the minimum result vector to be updated.
   * - ``positive``
     - ``false``
     - A boolean value. When set to ``true``, it forces the coefficients to be positive.
   * - ``selection``
     - ``cyclic``
     - Value that specifies the strategy of certain coordinate selection on each iteration.
       Except for default ``cyclic`` value, Coordinate Descent also supports:

       - ``random`` – on each iteration the index of coordinate is selected randomly by the engine.
   * - ``skipTheFirstComponents``
     - ``false``
     - A boolean value. When set to ``true``, Coordinate Descent algorithm will skip the first component from optimization.

Examples
********

.. tabs::

  .. tab:: C++ (CPU)

    - :cpp_example:`cd_dense_batch.cpp <optimization_solvers/cd_dense_batch.cpp>`

