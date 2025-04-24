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

.. _iterative_solver_computation:

Computation
***********

.. _iterative_solver_computation_input:

Algorithm Input
---------------

The iterative solver algorithm accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see Algorithms.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Iterative Solver Computation
   :widths: 10 60
   :header-rows: 1
   :align: left
   :class: longtable

   * - Input ID
     - Input
   * - ``inputArgument``
     - A numeric table of size :math:`p \times 1` with the value of start argument :math:`\theta_0`.
   * - ``optionalArgument``
     - Object of the ``OptionalArgument`` class that contains a set of algorithm-specific intrinsic parameters.
       For a detailed definition of the set, see the problem statement above and the description of a specific algorithm.

.. _iterative_solver_computation_parameters:

Algorithm Parameters
--------------------

The iterative solver algorithm has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for Iterative Solver Computation
   :widths: 10 10 60
   :header-rows: 1
   :align: left
   :class: longtable

   * - Parameter
     - Default Value
     - Description
   * - ``function``
     - Not applicable
     - Objective function represented as a sum of functions.
   * - ``nIterations``
     - :math:`100`
     - Maximum number of iterations of the algorithm.
   * - ``accuracyThreshold``
     - :math:`1.0-e5`
     - Accuracy of the algorithm. The algorithm terminates when this accuracy is achieved.
   * - ``optionalResultRequired``
     - ``false``
     - Indicates whether the set of the intrinsic parameters should be returned by the solver.

.. _iterative_solver_computation_output:

Algorithm Output
----------------

The iterative solver algorithm calculates the result described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
For more details, see Algorithms.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Iterative Solver Computation
   :widths: 10 60
   :header-rows: 1
   :align: left
   :class: longtable

   * - Result ID
     - Result
   * - ``minimum``
     - A numeric table of size :math:`p \times 1` with argument :math:`\theta_{*}`.
       By default, the result is an object of the HomogenNumericTable class, but you can define the result as an object of any class
       derived from NumericTable, except for PackedTriangularMatrix and PackedSymmetricMatrix.

   * - ``nIterations``
     - A numeric table of size :math:`1 \times 1` with a 32-bit integer number of iterations done by the algorithm.
       By default, the result is an object of the HomogenNumericTable class, but you can define the result as an object of any class
       derived from NumericTable, except for PackedTriangularMatrix, PackedSymmetricMatrix, and CSRNumericTable.

   * - ``optionalResult``
     - Object of the OptionalArgument class that contains a set of algorithm-specific intrinsic parameters.
       For a detailed definition of the set, see the problem statement above and the description of a specific algorithm.
