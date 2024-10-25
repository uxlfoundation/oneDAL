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

Computation
===========

Input
*****

The objective function accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Input for Objective Function Computation
   :widths: 10 60
   :align: left

   * - Input ID
     - Input
   * - ``argument``
     - A numeric table of size :math:`p \times 1` with the input argument of the objective function.

Parameters
**********

The objective function has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Parameters for Objective Function Computation
   :widths: 15 15 70
   :align: left
   :class: longtable

   * - Parameter
     - Default value
     - Description
   * - ``resultsToCompute``
     - ``gradient``
     - The 64-bit integer flag that specifies which characteristics of the objective function to compute.

       Provide one of the following values to request a single characteristic or use bitwise OR to request
       a combination of the characteristics:

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

       .. note:: On GPU, ``resultsToCompute`` only computes ``value``, ``gradient``, and ``hessian``.

.. _objective_function_output:

Output
******

The objective function calculates the result described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Output for Objective Function Computation
   :widths: 10 60
   :align: left
   :class: longtable

   * - Result ID
     - Result
   * - ``valueIdx``
     - A numeric table of size :math:`1 \times 1` with the value of the objective function in the given argument.
   * - ``nonSmoothTermValueIdx``
     - A numeric table of size :math:`1 \times 1` with the value of the non-smooth term of the
       objective function in the given argument.
   * - ``gradientIdx``
     - A numeric table of size :math:`p \times 1` with the gradient of the smooth term of the
       objective function in the given argument.
   * - ``hessianIdx``
     - A numeric table of size :math:`p \times p` with the Hessian of the smooth term of the
       objective function in the given argument.
   * - ``proximalProjectionIdx``
     - A numeric table of size :math:`p \times 1` with the projection of proximal operator
       for non-smooth term of the objective function in the given argument.
   * - ``lipschitzConstantIdx``
     - A numeric table of size :math:`1 \times 1` with Lipschitz constant of the smooth term of the objective function.
   * - ``gradientOverCertainFeatureIdx``
     - A numeric table of size :math:`1 \times 1` with certain component of gradient vector.
   * - ``hessianOverCertainFeatureIdx``
     - A numeric table of size :math:`1 \times 1` with certain component of hessian diagonal.
   * - ``proximalProjectionOverCertainFeatureIdx``
     - A numeric table of size :math:`1 \times 1` with certain component of proximal projection.

.. note::

  - If the function result is not requested through the resultsToCompute parameter,
    the respective element of the result contains a NULL pointer.

  - By default, each numeric table specified by the collection elements is an object of the HomogenNumericTable class,
    but you can define the result as an object of any class derived from NumericTable,
    except for PackedSymmetricMatrix, PackedTriangularMatrix, and CSRNumericTable.

  - Hessian matrix is computed for the objective function :math:`F(\theta) \in C^2`.
    For the objective functions :math:`F(\theta) \in C^p` with :math:`p < 2` the library
    will stop computations and report the status on non-availability of the computation of the Hessian.

  - If Lipschitz constant constantOfLipschitz is not estimated explicitly,
    pointer to result numeric table is required to be set to nullptr.
