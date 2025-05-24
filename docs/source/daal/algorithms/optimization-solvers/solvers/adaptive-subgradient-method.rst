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

.. _adagrad_solver:

Adaptive Subgradient Method
===========================

The adaptive subgradient method (AdaGrad) [Duchi2011]_ follows the :ref:`algorithmic framework of an iterative solver <iterative_solver>`
with the algorithm-specific transformation :math:`T`, set of intrinsic parameters :math:`S_t` defined for the learning rate :math:`\eta`,
and algorithm-specific vector :math:`U` and power :math:`d` of `Lebesgue space <https://en.wikipedia.org/wiki/Lp_space>`_ defined as follows:

.. math::
    S_t = {G_t}

    G_t = (G_{t, i})_{i = 1, \ldots, p}

    G_0 \equiv 0

:math:`T(\theta_{t - 1}, g(\theta_{t - 1}), S_{t - 1})`:

#. :math:`G_{t, i} = G_{t - 1, i} + g_i^2(\theta_{t - 1})`,
   where :math:`g_i(\theta_{t - 1})` is the :math:`i`-th coordinate of the gradient :math:`g(\theta_{t - 1})`

#. :math:`\theta_t = \theta_{t - 1} - \frac {\eta}{\sqrt{G_t + \varepsilon}} g(\theta_{t - 1})`,
   where

   .. math::
        \frac {\eta}{\sqrt{G_t + \varepsilon}} g(\theta_{t - 1}) =
        \{\frac {\eta}{\sqrt{G_{t, 1} + \varepsilon}} g_1(\theta_{t - 1}), \ldots, \frac {\eta}{\sqrt{G_{t, 1} + \varepsilon}} g_p(\theta_{t - 1})\}

Convergence check: :math:`U = g(\theta_{t - 1}), d = 2`

Computation
***********

The adaptive subgradient (AdaGrad) method is a special case of an iterative solver.
For parameters, input, and output of iterative solvers, see :ref:`Computation for Iterative Solver<iterative_solver_computation>`.

Algorithm Input
---------------

In addition to the :ref:`input of the iterative solver <iterative_solver_computation_input>`,
the AdaGrad method accepts the following optional input:

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Adaptive Subgradient Method Computation
   :header-rows: 1
   :widths: 10 60
   :align: left

   * - OptionalDataID
     - Input
   * - ``gradientSquareSum``
     - A numeric table of size :math:`p \times 1` with the values of :math:`G_t`.
       Each value is an accumulated sum of squares of coordinate values of a corresponding gradient.

Algorithm Parameters
--------------------

In addition to :ref:`parameters of the iterative solver <iterative_solver_computation_parameters>`,
the AdaGrad method has the following parameters:

.. tabularcolumns::  |\Y{0.2}|\Y{0.2}|\Y{0.6}|

.. list-table:: Algorithm Parameters for Adaptive Subgradient Method Computation
   :header-rows: 1
   :align: left
   :widths: 10 10 30
   :class: longtable

   * - Parameter
     - Default Value
     - Description
   * - ``algorithmFPType``
     - ``float``
     - The floating-point type that the algorithm uses for intermediate computations. Can be ``float`` or ``double``.
   * - ``method``
     - ``defaultDense``
     - Default performance-oriented computation method.
   * - ``batchIndices``
     - ``NULL``
     - A numeric table of size :math:`\text{nIterations} \times \text{batchSize}` for the ``defaultDense`` method
       that represents 32-bit integer indices of terms in the objective function.
       If no indices are provided, the algorithm generates random indices.
   * - ``batchSize``
     - :math:`128`
     - The number of batch indices to compute the stochastic gradient.

       If ``batchSize`` equals the number of terms in the objective function, no random sampling is performed,
       and all terms are used to calculate the gradient.

       The algorithm ignores this parameter if the ``batchIndices`` parameter is provided.
   * - ``learningRate``
     - A numeric table of size :math:`1 \times 1` that contains the default step length equal to :math:`0.01`.
     - A numeric table of size :math:`1 \times 1` that contains the value of learning rate :math:`\eta`.

       .. note::
            This parameter can be an object of any class derived from ``NumericTable``,
            except for ``PackedTriangularMatrix``, ``PackedSymmetricMatrix``, and ``CSRNumericTable``.
   * - ``degenerateCasesThreshold``
     - :math:`1\mathrm{e}{-08}`
     - Value :math:`\varepsilon` needed to avoid degenerate cases when computing square roots.
   * - ``engine``
     - `SharePtr< engines:: mt19937:: Batch>()`
     - Pointer to the random number generator engine that is used internally for generation of 32-bit integer indices of terms in the objective function.

Algorithm Output
----------------

In addition to the :ref:`output of the iterative solver <iterative_solver_computation_output>`,
the AdaGrad method calculates the following optional result:

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Adaptive Subgradient Method Computation
   :header-rows: 1
   :widths: 10 60
   :align: left

   * - OptionalDataID
     - Output
   * - ``gradientSquareSum``
     - A numeric table of size :math:`p \times 1` with the values of :math:`G_t`.
       Each value is an accumulated sum of squares of coordinate values of a corresponding gradient.

Examples
********

.. tabs::

  .. tab:: C++ (CPU)

    - :cpp_example:`adagrad_dense_batch.cpp <optimization_solvers/adagrad_dense_batch.cpp>`
    - :cpp_example:`adagrad_opt_res_dense_batch.cpp <optimization_solvers/adagrad_opt_res_dense_batch.cpp>`

  .. tab:: Python*

      - :daal4py_example:`adagrad_mse.py`
