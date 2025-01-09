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

Sum of Functions
================

The sum of functions :math:`F(\theta)` is a function that has the form of a sum:

.. math::
    F(\theta) = \sum _{i=1}{n} F_i(\theta), \theta \in \mathbb{R}^p

For given set of the indices :math:`I = \{i_1, i_2, \ldots , i_m\}`, :math:`1 \leq ik < n`,
:math:`k \in \{1, \ldots, m\}`, the value and the gradient of the sum of functions in the argument :math:`\theta` has the format:

.. math::
    F_I(\theta) = \sum _{i \in I} F_i(\theta)

.. math::
    \nabla_I F_I(\theta) = \sum _{i \in I} \nabla F_i(\theta)

The set of the indices :math:`I` is called a batch of indices.

Computation
***********

Algorithm Input
---------------

The sum of functions algorithm accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Sum of Functions Computation
   :widths: 10 60
   :header-rows: 1

   * - Input ID
     - Input
   * - ``argument``
     - A numeric table of size :math:`p \times 1` with the input argument of the objective function.

Algorithm Parameters
--------------------

The sum of functions algorithm has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for Sum of Functions Computation
   :widths: 10 10 60
   :header-rows: 1
   :align: left
   :class: longtable

   * - Parameter
     - Default Value
     - Description
   * - ``resultsToCompute``
     - ``gradient``
     - The 64-bit integer flag that specifies which characteristics of the objective function to compute.

       Provide one of the following values to request a single characteristic or use bitwise OR to request a combination of the characteristics:

       value
           Value of the objective function
       nonSmoothTermValue
           Value of non-smooth term of the objective function
       gradient
           Gradient of the smooth term of the objective function
       hessian
           Hessian of smooth term of the objective function
       proximalProjection
           Projection of proximal operator for non-smooth term of the objective function
       lipschitzConstant
           Lipschitz constant of the smooth term of the objective function
       gradientOverCertainFeature
           Certain component of gradient vector
       hessianOverCertainFeature
           Certain component of hessian diagonal
       proximalProjectionOfCertainFeature
           Certain component of proximal projection

Algorithm Output
----------------

For the output of the sum of functions algorithm, see :ref:`objective_function_output` for objective functions.