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

.. _objective_function_precomputed_characteristics:

Objective Function with Precomputed Characteristics Algorithm
=============================================================

Objective function with precomputed characteristics gives an ability to provide
the results of the objective function precomputed with the user-defined algorithm.

Set an earlier computed value and/or gradient and/or Hessian by allocating the result object
and setting the characteristics of this result object.
After that provide the modified result object to the algorithm for its further use with the iterative solver.

For more details on iterative solvers, refer to :ref:`iterative_solver`.
