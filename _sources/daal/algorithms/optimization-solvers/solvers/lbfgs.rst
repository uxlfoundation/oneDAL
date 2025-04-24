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

.. _lbfgs_solver:

Limited-Memory Broyden-Fletcher-Goldfarb-Shanno Algorithm
=========================================================

.. toctree::
   :glob:
   :maxdepth: 4


The limited-memory Broyden-Fletcher-Goldfarb-Shanno (LBFGS) algorithm
[Byrd2015]_ follows the algorithmic framework of an iterative
solver with the algorithm-specific transformation :math:`T` and set of intrinsic
parameters :math:`S_t` defined for the memory parameter :math:`m`, frequency
of curvature estimates calculation :math:`L`, and step-length sequence :math:`\alpha_t > 0`, algorithm-specific vector :math:`U` and power :math:`d` of `Lebesgue
space <https://en.wikipedia.org/wiki/Lp_space>`__ defined as follows:

Transformation
**************

.. math::
	T(\theta_{t-1},g(\theta_{t-1}), S_{t-1}) 

.. math::
	\theta_{t} = \begin{cases}
	\theta_{t-1}-\alpha^{t}g(\theta_{t-1}), & t \leq 2 \\
	\theta_{t-1}-\alpha^{t}Hg(\theta_{t-1}), & \mathrm{otherwise} \end{cases}

where :math:`H` is an approximation of the inverse Hessian matrix computed
from m correction pairs by the Hessian Update Algorithm.

Convergence check: :math:`U=g\left(\theta_{t-1}\right), d=2`

Intrinsic Parameters
********************

For the LBFGS algorithm, the set of intrinsic parameters :math:`S_t` includes the following:

-  Correction pairs :math:`(s_j , y_j)`
-  Correction index k in the buffer that stores correction pairs
-  Index of last iteration t of the main loop from the previous run
-  Average value of arguments for the previous L iterations :math:`\overline{\theta_{k-1}}`
-  Average value of arguments for the last L iterations :math:`\overline{\theta_k}`

Below is the definition and update flow of the intrinsic
parameters :math:`(s_j , y_j)`. The index is set and
remains zero for the first :math:`2L-1`` iterations of the main loop.
Starting with iteration :math:`2L`, the algorithm executes the following
steps for each of :math:`L`` iterations of the main loop:

#.

 :math:`k:=k+1`

#.

 Choose a set of indices without replacement: :math:`I_H = \{i_1, i_2, \ldots, i_{b_H}\}`, :math:`1 \leq i_l < n`, :math:`l \in \{1, \ldots, b_H\}`,
 :math:`|I_H| = b_H = \mathrm{correctionPairBatchSize}`.

#.

 Compute the sub-sampled Hessian

 .. math::
	{\nabla }^{2}F\left(\overline{{\theta }_{k}}\right)=\frac{1}{{b}_{H}}\sum _{i\in {I}_{H}}{\nabla }^{2}{F}_{i}\left(\overline{{\theta }_{k}}\right)

 at the point :math:`\overline{{\theta }_{k}}=\frac{1}{L}\sum _{i=Lk}^{L\left(k+1\right)}{\theta }_{i}` for the objective function using Hessians of its terms

 .. math::
	{\nabla }^{2}{F}_{i}=\left[\begin{array}{ccc}\frac{\partial {F}_{i}}{\partial {\theta }_{0}\partial {\theta }_{0}}& \cdots & \frac{\partial {F}_{i}}{\partial {\theta }_{0}\partial {\theta }_{p}}\\ ⋮& \ddots & ⋮\\ \frac{\partial {F}_{i}}{\partial p\partial {\theta }_{0}}& \cdots & \frac{\partial {F}_{i}}{\partial {\theta }_{p}\partial {\theta }_{p}}\end{array}\right]

#.

  Compute the correction pairs :math:`(s_k , y_k)`:

  :math:`{s}_{k}=\overline{{\theta }_{k}}-\overline{{\theta }_{k-1}}`

  :math:`{y}_{k}={\nabla }^{2}F\left(\overline{{\theta }_{k}}\right){s}_{k}`

.. note::

   - The set :math:`S_k` of intrinsic parameters is updated once
     per :math:`L` iterations of the major loop and remains unchanged
     between iterations with the numbers that are multiples of :math:`L`

   - A cyclic buffer stores correction pairs. The algorithm fills
     the buffer with pairs one-by-one. Once the buffer is full,
     it returns to the beginning and overwrites the previous
     correction pairs.

Hessian Update Algorithm
************************

This algorithm computes the approximation of the inverse Hessian
matrix from the set of correction pairs
[Byrd2015]_.

For a given set of correction pairs :math:`(s_j, y_j)`, :math:`j = k - min(k, m) + 1, \ldots, k`:

#. Set :math:`H={s}_{k}^{T}{y}_{k}/{y}_{k}^{T}{y}_{k}`


#. Iterate :math:`j` from :math:`k - min (k, m) + 1` until :math:`k`:

   a. :math:`{\rho }_{j}=1/{y}_{j}^{T}{y}_{j}`

   b. :math:`H:=\left(I-{\rho }_{j}{s}_{j}{y}_{j}^{T}\right)H\left(I-{\rho }_{j}{y}_{j}{s}_{j}^{T}\right)+{\rho }_{j}{s}_{j}{s}_{j}^{T}.`

#. Return :math:`H`


Computation
***********

The limited-memory BFGS algorithm is a special case of an iterative
solver. For parameters, input, and output of iterative solvers, see :ref:`iterative_solver_computation`.

Algorithm Input
---------------

In addition to the input of the iterative solver,
the limited-memory BFGS algorithm accepts the following optional input:

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Limited-Memory Broyden-Fletcher-Goldfarb-Shanno Computation
   :widths: 10 60
   :header-rows: 1
   :align: left
   :class: longtable

   * - OptionalDataID
     - Input
   * - ``correctionPairs``
     - A numeric table of size :math:`2m \times p` where the rows represent correction pairs :math:`s`
       and :math:`y`. The row correctionPairs[j], :math:`0 \leq j < m`, is a correction vector
       :math:`s_j`, and the row correctionPairs[j], :math:`m \leq j < 2m`, is a correction
       vector :math:`y_j`.
   * - ``correctionIndices``
     - A numeric table of size :math:`1 \times 2` with 32-bit integer indexes. The first value
       is the index of correction pair :math:`t`, the second value is the index of last
       iteration :math:`k` from the previous run.
   * - ``averageArgumentLIterations``
     - A numeric table of size :math:`2 \times p`, where row 0 represents average arguments
       for previous :math:`L` iterations, and row 1 represents average arguments for
       last :math:`L` iterations. These values are required to compute :math:`s` correction
       vectors in the next step.

Algorithm Parameters
--------------------

In addition to parameters of the iterative solver,
the limited-memory BFGS algorithm has the following parameters:

.. tabularcolumns::  |\Y{0.2}|\Y{0.2}|\Y{0.6}|

.. list-table:: Algorithm Parameters for Limited-Memory Broyden-Fletcher-Goldfarb-Shanno Computation
   :widths: 10 20 30
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
     - Performance-oriented computation method
   * - ``batchIndices``
     - ``NULL``
     - The numeric table of size :math:`nIterations \times batchSize` with 32-bit integer
       indices of terms in the objective function to be used in step
       2 of the limited-memory BFGS algorithm. If no indices are provided, the
       implementation generates random indices.

       .. note::

          This parameter can be an object of any class derived from ``NumericTable``,
          except for ``PackedTriangularMatrix``, ``PackedSymmetricMatrix``, and ``CSRNumericTable``.
   * - ``batchSize``
     - :math:`10`
     - The number of observations to compute the stochastic gradient. The
       implementation of the algorithm ignores this parameter if the
       batchIndices numeric table is provided.

       If BatchSize equals the number of terms in the objective function, no
       random sampling is performed and all terms are used to calculate the
       gradient.
   * - ``correctionPairBatchSize``
     - :math:`100`
     - The number of observations to compute the sub-sampled Hessian for
       correction pairs computation. The implementation of the
       algorithm ignores this parameter if the correctionPairIndices numeric
       table is provided.

       If ``correctionPairBatchSize`` equals the number of terms in the objective
       function, no random sampling is performed and all terms are used to
       calculate the Hessian matrix.
   * - ``correctionPairIndices``
     - ``NULL``
     - The numeric table of size :math:`(nIterations/L) \times correctionPairBatchSize` with
       32-bit integer indices to be used instead of random values. If no indices are provided, the
       implementation generates random indices.

       .. note::

          This parameter can be an object of any class derived from ``NumericTable``,
          except for ``PackedTriangularMatrix``, ``PackedSymmetricMatrix``, and ``CSRNumericTable``.

       .. note::

          If the algorithm runs with no optional input data, :math:`(nIterations / L - 1)` rows
          of the table are used. Otherwise, it can use one more row, :math:`(nIterations / L)` in total.
   * - :math:`m`
     - :math:`10`
     - The memory parameter. The maximum number of correction pairs that define
       the approximation of the Hessian matrix.
   * - :math:`L`
     - :math:`10`
     - The number of iterations between calculations of the curvature estimates.
   * - ``stepLengthSequence``
     - A numeric table of size :math:`1 \times 1` that contains the default step length equal to :math:`1`.
     - The numeric table of size :math:`1 \times nIterations` or :math:`1 \times 1`. The contents of the table depend on its size:

       -  :math:`size = 1 \times nIterations`: values of the step-length sequence :math:`\alpha^k` for :math:`k = 1, \ldots, nIterations`.

       -  :math:`size = 1 \times 1`: the value of step length at each iteration :math:`\alpha^1 = \ldots = \alpha^{nIterations}`

       ..note::

          This parameter can be an object of any class derived from ``NumericTable``,
          except for ``PackedTriangularMatrix``, ``PackedSymmetricMatrix``, and ``CSRNumericTable``.

       The recommended data type for storing the step-length sequence is the
       floating-point type, either float or double, that the algorithm uses in
       intermediate computations.
   * - ``engine``
     - `SharePtr< engines:: mt19937:: Batch>()`
     - Pointer to the random number generator engine that is used internally
       for random choosing terms from the objective function.

Algorithm Output
----------------

In addition to the output of the iterative solver, the limited-memory
BFGS algorithm calculates the following optional results:

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Limited-Memory Broyden-Fletcher-Goldfarb-Shanno Computation
   :widths: 10 60
   :header-rows: 1
   :align: left
   :class: longtable

   * - OptionalDataID
     - Output
   * - ``correctionPairs``
     - A numeric table of size :math:`2m \times p` where the rows represent correction pairs :math:`s`
       and :math:`y`. The row correctionPairs[j], :math:`0 \leq j < m`, is a correction vector
       :math:`s_j`, and the row correctionPairs[j], :math:`m \leq j < 2m`, is a correction
       vector :math:`y_j`.
   * - ``correctionIndices``
     - A numeric table of size :math:`1 \times 2` with 32-bit integer indexes. The first value
       is the index of correction pair :math:`t`, the second value is the index of last
       iteration :math:`k` from the previous run.
   * - ``averageArgumentLIterations``
     - A numeric table of size :math:`2 \times p`, where row 0 represents average arguments
       for previous :math:`L` iterations, and row 1 represents average arguments for
       last :math:`L` iterations. These values are required to compute :math:`s` correction
       vectors in the next step.

Examples
--------

.. tabs::

  .. tab:: C++ (CPU)

     Batch Processing:

     - :cpp_example:`lbfgs_dense_batch.cpp <optimization_solvers/lbfgs_dense_batch.cpp>`
     - :cpp_example:`lbfgs_opt_res_dense_batch.cpp <optimization_solvers/lbfgs_opt_res_dense_batch.cpp>`


  .. tab:: Python*

     Batch Processing:

     - :daal4py_example:`lbfgs_cr_entr_loss.py`
     - :daal4py_example:`lbfgs_mse.py`
