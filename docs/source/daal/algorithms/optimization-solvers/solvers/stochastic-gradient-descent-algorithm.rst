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

.. _sgd_solver:

Stochastic Gradient Descent Algorithm
=====================================

.. toctree::
   :glob:
   :maxdepth: 4

The stochastic gradient descent (SGD) algorithm is a special case of
an iterative solver. See :ref:`iterative_solver` for more details.

Computation methods
*******************
The following computation methods are available in |product| for the
stochastic gradient descent algorithm:

-  `Mini-batch method`_
-  `Default method`_ (a special case of mini-batch used by default)
-  `Momentum method`_

Mini-batch method
-----------------

The mini-batch method (miniBatch) of the stochastic gradient
descent algorithm [Mu2014]_
follows the algorithmic framework of an iterative
solver with an empty set of intrinsic parameters of the algorithm :math:`S_t`,
algorithm-specific transformation :math:`T` defined for the
learning rate sequence :math:`{\{\eta_t\}}_{t=1, \ldots, \text{nIterations}}`,
conservative sequence :math:`{\{\gamma_t\}}_{t=1, \ldots, \text{nIterations}}` and
the number of iterations in the internal loop :math:`L`,
algorithm-specific vector :math:`U` and power :math:`d` of `Lebesgue
space <https://en.wikipedia.org/wiki/Lp_space>`__ defined as
follows:

.. math::
	T\left({\theta }_{t-1}, g\left({\theta }_{t-1}\right), {S}_{t-1}\right)


For :math:`l` from :math:`1` until :math:`L`:

#.

 Update the function argument: :math:`{\theta }_{t}:= {\theta }_{t}-{\eta }_{t}\left(g\left({\theta }_{t}\right)+{\gamma }_{t}\left({\theta }_{t}-{\theta }_{t-1 }\right)\right)`

#.

 Compute the gradient: :math:`g\left({\theta }_{t}\right)=\nabla {F}_{I}\left({\theta }_{t}\right)`


Convergence check: :math:`U=g\left({\theta }_{t-1}\right), d=2`


Default method
--------------

The default method (defaultDense) is a particular case of the
mini-batch method with the batch size :math:`b=1`, :math:`L=1`, and conservative sequence :math:`{\gamma }_{t}\equiv 0`.

Momentum method
---------------

The momentum method (momentum) of the stochastic gradient descent
algorithm [Rumelhart86]_ follows the algorithmic framework of an iterative solver with the
set of intrinsic parameters :math:`S_t`, algorithm-specific
transformation :math:`T` defined for the learning rate sequence :math:`{\{\eta_t\}}_{t=1, \ldots, \text{nIterations}}`
and momentum parameter :math:`\mu in [0,1]`, and algorithm-specific vector :math:`U` and power :math:`d` of `Lebesgue
space <https://en.wikipedia.org/wiki/Lp_space>`__ defined as
follows:

.. math::
	T\left({\theta }_{t-1}, g\left({\theta }_{t-1}\right), {S}_{t-1}\right)


#. :math:`{v}_{t}=\mu \cdot {v}_{t-1}+{\eta }_{t}\cdot g\left({\theta }_{t-1}\right)`

#. :math:`{\theta }_{t}={\theta }_{t-1}-{v}_{t}`


For the momentum method of the SGD algorithm, the set of intrinsic
parameters :math:`S_t` only contains the last update vector :math:`v_t`.

Convergence check: :math:`U=g\left({\theta }_{t-1}\right), d=2`

Computation
***********

The stochastic gradient descent algorithm is a special case of an
iterative solver. For parameters, input, and output of iterative
solvers, see :ref:`iterative_solver_computation`.

Algorithm Parameters
---------------------

In addition to parameters of the iterative solver, the stochastic
gradient descent algorithm has the following parameters. Some of
them are required only for specific values of the computation
method parameter method:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.15}|\Y{0.55}|

.. list-table:: Algorithm Parameters for Stochastic Gradient Descent Algorithm Computation
   :widths: 10 10 10 30
   :header-rows: 1
   :align: left
   :class: longtable

   * - Parameter
     - method
     - Default Value
     - Description
   * - ``algorithmFPType``
     - ``defaultDense``, ``miniBatch``, ``momentum``
     - ``float``
     - The floating-point type that the algorithm uses for intermediate
       computations. Can be ``float`` or ``double``.
   * - ``method``
     - Not applicable
     - ``defaultDense``
     - Available computation methods:

       For CPU:

       - ``defaultDense``
       - ``miniBatch``
       - ``momentum``

       For GPU:

       - ``miniBatch``

   * - ``batchIndices``
     - ``defaultDense``, ``miniBatch``, ``momentum``
     - Not applicable
     - The numeric table with 32-bit integer indices of terms in the objective
       function. The method parameter determines the size of the numeric table:

       - ``defaultDense``: ``nIterations`` x 1
       - ``miniBatch`` and ``momentum``: ``nIterations`` x ``batchSize``

       If no indices are provided, the implementation generates random indices.

       .. include: ../../../includes/parameter_numeric_table

   * - ``batchSize``
     - ``miniBatch``,``momentum``
     - :math:`128`
     - The number of batch indices to compute the stochastic gradient.

       If ``batchSize`` equals the number of terms in the objective function, no random sampling
       is performed, and all terms are used to calculate the gradient.

       The algorithm ignores this parameter if the batchIndices parameter is
       provided.

       For the ``defaultDense`` value of method, one term is used to compute the
       gradient on each iteration.
   * - ``conservativeSequence``
     - ``miniBatch``
     - A numeric table of size :math:`1 \times 1` that contains the default conservative coefficient equal to 1.
     - The numeric table of size :math:`1 \times \text{nIterations}` or :math:`1 \times 1`. The contents of the
       table depend on its size:

       - size = :math:`1 \times \text{nIterations}`: values of the conservative coefficient sequence :math:`\gamma^k` for :math:`k = 1, \ldots, \text{nIterations}`.
       - size = :math:`1 \times 1` the value of conservative coefficient at each iteration :math:`\gamma^1 = \ldots = \gamma^\text{nIterations}`.

       .. include: ../../../includes/parameter_numeric_table

   * - ``innerNIterations``
     - ``miniBatch``
     - :math:`5`
     - The number of inner iterations for the miniBatch method.
   * - ``learningRateSequence``
     - ``defaultDense``, ``miniBatch``, ``momentum``
     - A numeric table of size :math:`1 \times 1` that contains the default step length equal to 1.
     - The numeric table of size :math:`1 \times \text{nIterations}` or :math:`1 \times 1`. The contents of the
       table depend on its size:

       - size = :math:`1 \times \text{nIterations}`: values of the learning rate sequence :math:`\eta^k` for :math:`k = 1, \ldots, \text{nIterations}`.
       - size = :math:`1 \times 1`: the value of learning rate at each iteration :math:`\eta^1 = \ldots = \eta^\text{nIterations}`.

       .. include: ../../../includes/parameter_numeric_table

   * - ``momentum``
     - ``momentum``
     - :math:`0.9`
     - The momentum value.
   * - ``engine``
     - ``defaultDense``, ``miniBatch``, ``momentum``
     - `SharePtr< engines:: mt19937:: Batch>()`
     - Pointer to the random number generator engine that is used internally
       for generation of 32-bit integer indices of terms in the objective function.

Examples
********

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`sgd_dense_batch.cpp <optimization_solvers/sgd_dense_batch.cpp>`
    - :cpp_example:`sgd_mini_dense_batch.cpp <optimization_solvers/sgd_mini_dense_batch.cpp>`
    - :cpp_example:`sgd_moment_dense_batch.cpp <optimization_solvers/sgd_moment_dense_batch.cpp>`
    - :cpp_example:`sgd_moment_opt_res_dense_batch.cpp <optimization_solvers/sgd_moment_opt_res_dense_batch.cpp>`

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`sgd_logistic_loss.py`
    - :daal4py_example:`sgd_mse.py`

.. Python*:

.. -  sgd_batch.py
.. -  sgd_mini_batch.py
.. -  sgd_moment_dense_batch.py
.. -  sgd_moment_opt_res_dense_batch.py
