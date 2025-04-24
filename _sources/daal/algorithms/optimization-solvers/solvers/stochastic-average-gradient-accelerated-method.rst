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

.. _saga_solver:

Stochastic Average Gradient Accelerated Method
==============================================

The Stochastic Average Gradient Accelerated (SAGA) [Defazio2014]_ follows
:ref:`the algorithmic framework of an iterative solver <iterative_solver>` with one exception.

The default method (``defaultDense``) of SAGA algorithm is a particular case of the iterative solver method with the batch size :math:`b = 1`.

Details
*******

Algorithmic-specific transformation :math:`T`, the set of intrinsic parameters :math:`S_t`
defined for the learning rate :math:`\eta`, and algorithm-specific vector :math:`U`
and power :math:`d` of `Lebesgue space <https://en.wikipedia.org/wiki/Lp_space>`_ are defined as follows:

.. math::
    S_t = \{ G^t \}

.. math::
    G^t = (G_i^t)_{i = 1, \ldots, n}

.. math::
    G^0 \equiv (G_i^0)_{i = 1, \ldots, n} \equiv F_i'(\theta_0)_{i = 1, \ldots, n}

:math:`S_t` is a matrix of the gradients of smooth terms at point :math:`\theta_t`, where

- :math:`t` is defined by the number of iterations the solver runs
- :math:`G_i^t` stores the gradient of :math:`f_i(\theta_t)`

:math:`T(\theta_{t-1}, F_j'(\theta_{t-1}), S_{t-1}, M(\theta_{t-1}))`:

#. :math:`W_t = \theta_{t-1} - \eta_j \left[ F_j'(\theta_{t-1}) - G_j^{t-1} + \frac{1}{n} \sum_{i=1}^{n} G_i^{t-1}\right]`

#. :math:`\theta_t = \mathrm{prox}_{\eta}^{M} (W_t)`

Update of the set of intrinsic parameters :math:`S_t`:

.. math::
    G_j^{t-1} = F_j'(\theta_{t-1})

.. note::
    The algorithm enables automatic step-length selection if learning rate :math:`\eta` was not provided by the user.
    Automatic step-length will be computed as :math:`\eta = \frac{1}{L}`,
    where :math:`L` is the Lipschitz constant returned by objective function.
    If the objective function returns ``nullptr`` to numeric table with ``lipschitzConstant`` Result ID,
    the library will use default step size :math:`0.01`.


Convergence checks:

- :math:`U = \theta_t - \theta_{t - 1}`, :math:`d = \infty`
- :math:`|x|_{\infty} = \underset{i \in [0, p]} \max(|x^i|)`, :math:`x \in R^p`

Computation
***********

The stochastic average gradient (SAGA) algorithm is a special case of an iterative solver.
For parameters, input, and output of iterative solvers, see :ref:`Iterative Solver > Computation <iterative_solver>`.

Algorithm Input
---------------

In addition to the :ref:`input of the iterative solver <iterative_solver_computation_input>`,
the SAGA optimization solver has the following optional input:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Input for Stochastic Average Gradient Accelerated Method Computation
   :widths: 10 10 60
   :align: left

   * - OptionalDataID
     - Default Value
     - Description
   * - ``gradientTable``
     - Not applicable
     - A numeric table of size :math:`n \times p` which represents :math:`G_0` matrix that contains gradients of
       :math:`F_i(\theta)`, :math:`1, \ldots, n` at the initial point :math:`\theta_0 \in R^p`.

       This input is optional: if the user does not provide the table of gradients for :math:`F_i(\theta)`, :math:`1, \ldots, n`,
       the library will compute it inside the SAGA algorithm.

       .. note::
            This parameter can be an object of any class derived from ``NumericTable``,
            except for ``PackedTriangularMatrix``, ``PackedSymmetricMatrix``, and ``CSRNumericTable``.

Algorithm Parameters
--------------------

In addition to parameters of the iterative solver, the SAGA optimization solver has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for Stochastic Average Gradient Accelerated Method Computation
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
   * - ``batchIndices``
     - :math:`1`
     - A numeric table of size :math:`\mathrm{nIterations} \times 1` with 32-bit integer indices of terms in the objective function.
       If no indices are provided, the implementation generates random index on each iteration.

       .. note::
            This parameter can be an object of any class derived from ``NumericTable``,
            except for ``PackedTriangularMatrix``, ``PackedSymmetricMatrix``, and ``CSRNumericTable``.
   * - ``learningRateSequence``
     - Not applicable
     - The numeric table of size :math:`1 \times \mathrm{nIterations}` or :math:`1 \times 1` that contains
       learning rate for each iterations is first case, otherwise constant step length will be used for all iterations.
       It is recommended to set diminishing learning rate sequence.

       If ``learningRateSequence`` is not provided, the learning rate will be computed automatically via ``constantOfLipschitz`` Result ID.

       .. note::
            This parameter can be an object of any class derived from ``NumericTable``,
            except for ``PackedTriangularMatrix``, ``PackedSymmetricMatrix``, and ``CSRNumericTable``.
   * - ``engine``
     - `SharedPtr<engines::mt19937::Batch<>`
     - Pointer to the random number generator engine that is used internally for generation of 32-bit integer index of term in the objective function.

Algorithm Output
----------------

In addition to the :ref:`output of the iterative solver <iterative_solver_computation_output>`,
the SAGA optimization solver calculates the following optional result:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Output for Stochastic Average Gradient Accelerated Method Computation
   :widths: 10 10 60
   :align: left

   * - OptionalDataID
     - Default Value
     - Description
   * - ``gradientTable``
     - Not applicable
     - A numeric table of size :math:`n \times p` that represents matrix :math:`G_t` updated after all iterations.

       This parameter can be an object of any class derived from ``NumericTable``,
       except for ``PackedTriangularMatrix``, ``PackedSymmetricMatrix``, and ``CSRNumericTable``.

Examples
********

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`saga_dense_batch.cpp <optimization_solvers/saga_dense_batch.cpp>`
    - :cpp_example:`saga_logistic_loss_dense_batch.cpp <optimization_solvers/saga_logistic_loss_dense_batch.cpp>`

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`saga.py`
