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

LASSO and Elastic Net Computation
=================================

Batch Processing
****************

LASSO and Elastic Net algorithms follow the general workflow described in
:ref:`regression_usage_model`.

Training
--------

For a description of common input and output parameters, refer to
:ref:`regression_usage_model`.
Both LASSO and Elastic Net algorithms have the following input parameters in addition to the common input parameters:

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Training Input for LASSO and Elastic Net (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :align: left
   :class: longtable

   * - Input ID
     - Input
   * - ``weights``
     - Optional input.

       Pointer to the :math:`1 \times n` numeric table with weights of samples.
       The input can be an object of any class derived from NumericTable except for PackedTriangularMatrix,
       PackedSymmetricMatrix, and CSRNumericTable.

       By default, all weights are equal to 1.

   * - ``gramMatrix``
     - Optional input.

       Pointer to the :math:`p \times p` numeric table with pre-computed Gram Matrix.
       The input can be an object of any class derived from NumericTable except for CSRNumericTable.

       By default, the table is set to an empty numeric table.
       It is used only when the number of features is less than the number of observations.

Chosse the appropriate tab to see the parameters used in LASSO and Elastic Net batch training algorithms:

.. tabs::

  .. group-tab:: LASSO

    .. tabularcolumns::  |\Y{0.2}|\Y{0.2}|\Y{0.6}|

    .. list-table:: Training Parameters for LASSO (Batch Processing)
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
        - The computation method used by the LASSO regression. The only training method supported so far is the default dense method.
      * - ``interceptFlag``
        - ``True``
        - A flag that indicates whether or not to compute
      * - ``lassoParameters``
        - A numeric table of size :math:`1 \times 1` that contains the default LASSO parameter equal to :math:`0.1`.
        - :math:`L_1` coefficients: :math:`\lambda_i`

          A numeric table of size :math:`1 \times k` (where :math:`k` is the number of dependent variables) or :math:`1 \times 1`.
          The contents of the table depend on its size:

          - For the table of size :math:`1 \times k`, use the values of LASSO parameters :math:`\lambda_j` for :math:`j = 1, \ldots, k`.
          - For the table of size :math:`1 \times 1`, use the value of LASSO parameter for each dependant variable :math:`\lambda_1 = \ldots = \lambda_k`.

          This parameter can be an object of any class derived from NumericTable, except for PackedTriangularMatrix,
          PackedSymmetricMatrix, and CSRNumericTable.

      * - ``optimizationSolver``
        - :ref:`Coordinate Descent solver <cda_solver>`
        - Optimization procedure used at the training stage.
      * - ``optResultToCompute``
        - :math:`0`
        - The 64-bit integer flag that specifies which extra characteristics of the LASSO regression to compute.

          Provide the following value to request a characteristic:

          - ``computeGramMatrix`` for Computation Gram matrix

      * - ``dataUseInComputation``
        - ``doNotUse``
        - A flag that indicates a permission to overwrite input data.
          Provide the following value to restrict or allow modification of input data:

          - ``doNotUse`` – restricts modification
          - ``doUse`` – allows modification

  .. group-tab:: Elastic Net

    .. tabularcolumns::  |\Y{0.2}|\Y{0.2}|\Y{0.6}|

    .. list-table:: Training Parameters for Elastic Net (Batch Processing)
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
        - The computation method used by the Elastic Net regression. The only training method supported so far is the default dense method.
      * - ``interceptFlag``
        - ``True``
        - A flag that indicates whether or not to compute
      * - ``penaltyL1``
        - A umeric table of size :math:`1 \times 1` that contains the default Elastic Net parameter equal to :math:`0.5`.
        - L1 regularization coefficient (penaltyL1 is :math:`\lambda_1` as described in :ref:`elastic_net`).

          The numeric table of size :math:`1 \times k` (where :math:`k` is the number of dependent variables) or :math:`1 \times 1`.
          The contents of the table depend on its size:

          - For the table of size :math:`1 \times k`, the values of the Elastic Net parameters :math:`\lambda_{1j}` for :math:`j = 1, \ldots, k`.
          - For the table of size :math:`1 \times 1`, the values of the Elastic Net parameter for each dependent veriable :math:`\lambda_{11} = \ldots = \lambda_{1k}`.

          This parameter can be an object of any class derived from NumericTable,
          except for PackedTriangularMatrix, PackedSymmetricMatrix, and CSRNumericTable.

      * - ``penaltyL2``
        - A numeric table of size :math:`1 \times 1` that contains the default Elastic Net parameter equal to :math:`0.5`.
        - L2 regularization coefficient (penaltyL2 is :math:`\lambda_2` as described in :ref:`elastic_net`).

          The numeric table of size :math:`1 \times k` (where :math:`k` is the number of dependent variables) or :math:`1 \times 1`.
          The contents of the table depend on its size:

          - For the table of size :math:`1 \times k`, the values of the Elastic Net parameters :math:`\lambda_{2j}` for :math:`j = 1, \ldots, k`.
          - For the table of size :math:`1 \times 1`, the values of the Elastic Net parameter for each dependent veriable :math:`\lambda_{21} = \ldots = \lambda_{2k}`.

          This parameter can be an object of any class derived from NumericTable,
          except for PackedTriangularMatrix, PackedSymmetricMatrix, and CSRNumericTable.

      * - ``optimizationSolver``
        - :ref:`Coordinate Descent solver <cda_solver>`
        - Optimization procedure used at the training stage.
      * - ``optResultToCompute``
        - :math:`0`
        - The 64-bit integer flag that specifies which extra characteristics of the Elastic Net regression to compute.

          Provide the following value to request a characteristic:

          - ``computeGramMatrix`` for computation of the Gram Matrix

      * - ``dataUseInComputation``
        - ``doNotUse``
        - A flag that indicates a permission to overwrite input data.
          Provide the following value to restrict or allow modification of input data:

          - ``doNotUse`` – restricts modification
          - ``doUse`` – allows modification


    .. note::

      Common combinations of Elastic Net regularization parameters [Friedman2010]_ might be computed as shown below:

      - compromise between L1 (lasso penalty) and L2 (ridge-regression penalty) regularization:

        .. math::

          \text{alpha} = \frac{\text{penaltyL1}}{\text{penaltyL1} + \text{penaltyL2}}

      - control full regularization:

        .. math::

          \text{lambda} = \text{penaltyL1} + \text{penaltyL2}

In addition, both LASSO and Elastic Net algorithms have the following optional results:

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Training Output for LASSO and Elastic Net (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :align: left

   * - Result ID
     - Result
   * - ``gramMatrix``
     - Pointer to the computed Gram Matrix with size :math:`p \times p`

Prediction
----------

For a description of the input and output, refer to
:ref:`regression_usage_model`.

At the prediction stage, LASSO and Elastic Net algorithms have the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Prediction Parameters for LASSO and Elastic Net (Batch Processing)
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
     - Default performance-oriented computation method, the only method supported by the regression-based prediction.

Examples
--------

.. tabs::

  .. group-tab:: LASSO

     C++: :cpp_example:`lasso_reg_dense_batch.cpp <lasso_regression/lasso_reg_dense_batch.cpp>`

  .. group-tab:: Elastic Net

     C++: :cpp_example:`elastic_net_dense_batch.cpp <elastic_net/elastic_net_dense_batch.cpp>`

.. Python*: lasso_regression_batch.py
.. Python*: elastic_net_batch.py

Performance Considerations
**************************

For better performance when the number of samples is larger than the
number of features in the training data set, certain coordinates of
gradient and Hessian are computed via the component of Gram matrix.
When the number of features is larger than the number of
observations, the cost of each iteration via Gram matrix depends on
the number of features. In this case, computation is performed via
residual update [Friedman2010]_.

To get the best overall performance for LASSO and Elastic Net training, do the following:

- If the number of features is less than the number of samples, use :ref:`homogenous table <homogeneous_numeric_table>`.
- If the number of features is greater than the number of samples, use :ref:`SOA <soa_layout>` layout rather than AOS layout.
