.. Copyright 2019 Intel Corporation
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

Linear and Ridge Regressions Computation
****************************************

Batch Processing
================

Linear and ridge regressions in the batch processing mode follow the general workflow described in
:ref:`regression_usage_model`.

Training
--------

For a description of the input and output, refer to :ref:`regression_usage_model`.

The following table lists parameters of linear and ridge
regressions at the training stage. Some of these parameters or
their values are specific to a linear or ridge regression
algorithm.

.. tabs::

  .. group-tab:: Linear Regression

    .. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

    .. list-table:: Training Parameters for Linear Regression (Batch Processing)
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
        - Available methods for linear regression training:

          - ``defaultDense`` - the normal equations method
          - ``qrDense`` - the method based on QR decomposition

      * - ``interceptFlag``
        - ``true``
        - A flag that indicates a need to compute :math:`\beta_{0j}`.

  .. group-tab:: Ridge Regression

    .. tabularcolumns::  |\Y{0.25}|\Y{0.3}|\Y{0.45}|

    .. list-table:: Training Parameters for Ridge Regression (Batch Processing)
      :widths: 20 30 60
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
        - Default computation method used by the ridge regression.
          The only method supported at the training stage is the normal equations method.
      * - ``ridgeParameters``
        - A numeric table of size :math:`1 \times 1` that contains the default ridge parameter
          equal to :math:`1`.
        - The numeric table of size :math:`1 \times k` (:math:`k` is the number of dependent variables)
          or :math:`1 \times 1`. The contents of the table depend on its size:

          -  :math:`size = 1 \times k`: values of the ridge parameters :math:`\lambda_j` for :math:`j = 1, \ldots, k`.
          -  :math:`size = 1 \times 1`: the value of the ridge parameter for each dependent variable
             :math:`\lambda_1 = \ldots = \lambda_k`.

          .. note::

            This parameter can be an object of any class derived from ``NumericTable``,
            except for ``PackedTriangularMatrix``, ``PackedSymmetricMatrix``, and ``CSRNumericTable``.

      * - ``interceptFlag``
        - ``true``
        - A flag that indicates a need to compute :math:`\beta_{0j}`.


Prediction
----------

For a description of the input and output, refer to :ref:`regression_usage_model`.

At the prediction stage, linear and ridge regressions have the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Prediction Parameters for Linear and Ridge Regression (Batch Processing)
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
     - Default performance-oriented computation method, the only method supported by the regression based prediction.

Online Processing
=================

You can use linear and ridge regression in the online processing mode only at the training stage.

This computation mode assumes that the data arrives in blocks :math:`i = 1, 2, 3, \ldots \text{nblocks}`.

Training
--------

Linear and ridge regression training in the online processing mode follows the general workflow
described in :ref:`regression_usage_model`.

Linear and ridge regression training in the online processing mode accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Training Input for Linear and Ridge Regression (Online Processing)
   :widths: 10 60
   :header-rows: 1
   :class: longtable

   * - Input ID
     - Input
   * - ``data``
     - Pointer to the :math:`n_i \times p` numeric table that represents the current, :math:`i`-th, data block.
   * - ``dependentVariables``
     - Pointer to the :math:`n_i \times k` numeric table with responses associated with the current, :math:`i`-th, data block.

.. note:: Both input tables can be an object of any class derived from ``NumericTable``.

The following table lists parameters of linear and ridge regressions at the training stage in the online processing mode.

.. tabs::

  .. group-tab:: Linear Regression

    .. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

    .. list-table:: Training Parameters for Linear Regression (Online Processing)
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
        - Available methods for linear regression training:

          - ``defaultDense`` - the normal equations method
          - ``qrDense`` - the method based on QR decomposition

      * - ``interceptFlag``
        - ``true``
        - A flag that indicates a need to compute :math:`\beta_{0_j}`.

  .. group-tab:: Ridge Regression

    .. tabularcolumns::  |\Y{0.25}|\Y{0.3}|\Y{0.45}|

    .. list-table:: Training Parameters for Ridge Regression (Online Processing)
      :widths: 20 30 60
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
        - Default computation method used by the ridge regression.
          The only method supported at the training stage is the normal equations method.

      * - ``ridgeParameters``
        - A numeric table of size :math:`1 \times 1` that contains the default ridge parameter equal to :math:`1`.
        - The numeric table of size :math:`1 \times k` (:math:`k` is the number of dependent variables) or :math:`1 \times 1`.
          The contents of the table depend on its size:

          - size = :math:`1 \times k`: values of the ridge parameters :math:`\lambda_j` for :math:`j = 1, \ldots, k`.
          - size = :math:`1 \times 1`: the value of the ridge parameter for each dependent variable :math:`\lambda_1 = ... = \lambda_k`.

          .. note::

            This parameter can be an object of any class derived from ``NumericTable``, except for ``PackedTriangularMatrix``,
            ``PackedSymmetricMatrix``, and ``CSRNumericTable``.

      * - ``interceptFlag``
        - ``true``
        - A flag that indicates a need to compute :math:`\beta_{0_j}`.

For a description of the output, refer to :ref:`regression_usage_model`.

Distributed Processing
======================

You can use linear and ridge regression in the distributed processing mode only at the training stage.

This computation mode assumes that the data set is split in ``nblocks`` blocks across computation nodes.

Training
--------

Use the two-step computation schema for linear and ridge regression training in the distributed processing mode, as illustrated below:

- `Step 1 - on Local Nodes`_
- `Step 2 - on Master Node`_

Algorithm parameters
++++++++++++++++++++

The following table lists parameters of linear and ridge regressions at the training stage in the distributed processing mode.

.. tabs::

  .. group-tab:: Linear Regression

    .. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

    .. list-table:: Training Parameters for Linear Regression (Distributed Processing)
      :widths: 10 10 60
      :header-rows: 1
      :align: left
      :class: longtable

      * - Parameter
        - Default Value
        - Description
      * - ``computeStep``
        - Not applicable
        - The parameter required to initialize the algorithm. Can be:

          - ``step1Local`` - the first step, performed on local nodes
          - ``step2Master`` - the second step, performed on a master node

      * - ``algorithmFPType``
        - ``float``
        - The floating-point type that the algorithm uses for intermediate computations. Can be ``float`` or ``double``.
      * - ``method``
        - ``defaultDense``
        - Available methods for linear regression training:

          - ``defaultDense`` - the normal equations method
          - ``qrDense`` - the method based on QR decomposition
      * - ``interceptFlag``
        - ``true``
        - A flag that indicates a need to compute :math:`\beta_{0_j}`.


  .. group-tab:: Ridge Regression

    .. tabularcolumns::  |\Y{0.25}|\Y{0.3}|\Y{0.45}|

    .. list-table:: Training Parameters for Ridge Regression (Distributed Processing)
      :widths: 20 30 60
      :header-rows: 1
      :align: left
      :class: longtable

      * - Parameter
        - Default Value
        - Description
      * - ``computeStep``
        - Not applicable
        - The parameter required to initialize the algorithm. Can be:

          - ``step1Local`` - the first step, performed on local nodes
          - ``step2Master`` - the second step, performed on a master node

      * - ``algorithmFPType``
        - ``float``
        - The floating-point type that the algorithm uses for intermediate computations. Can be ``float`` or ``double``.
      * - ``method``
        - ``defaultDense``
        - Default computation method used by the ridge regression.
          The only method supported at the training stage is the normal equations method.
      * - ``ridgeParameters``
        - A numeric table of size :math:`1 \times 1` that contains the default ridge parameter equal to :math:`1`.
        - The numeric table of size :math:`1 \times k` (:math:`k` is the number of dependent variables) or :math:`1 \times 1`.
          The contents of the table depend on its size:

          - size = :math:`1 \times k`: values of the ridge parameters :math:`\lambda_j` for :math:`j = 1, \ldots, k`.
          - size = :math:`1 \times 1`: the value of the ridge parameter for each dependent variable :math:`\lambda_1 = ... = \lambda_k`.

          .. note::

            This parameter can be an object of any class derived from ``NumericTable``, except for ``PackedTriangularMatrix``,
            ``PackedSymmetricMatrix``, and ``CSRNumericTable``.

      * - ``interceptFlag``
        - ``true``
        - A flag that indicates a need to compute :math:`\beta_{0_j}`.

.. _lin_ridge_step_1:

Step 1 - on Local Nodes
+++++++++++++++++++++++

.. figure:: images/distributed-step-1.png
  :width: 600
  :alt:

  Linear and Ridge Regression Training: Distributed Processing, Step 1 - on Local Nodes

In this step, linear and ridge regression training accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Training Input for Linear and Ridge Regression (Distributed Processing, Step 1)
  :widths: 10 60
  :header-rows: 1
  :class: longtable

  * - Input ID
    - Input
  * - ``data``
    - Pointer to the :math:`n_i \times p` numeric table that represents the :math:`i`-th data block on the local node.
  * - ``dependentVariables``
    - Pointer to the :math:`n_i \times k` numeric table with responses associated with the :math:`i`-th data block.

.. note:: Both input tables can be an object of any class derived from ``NumericTable``.

In this step, linear and ridge regression training calculates the result described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Training Output for Linear and Ridge Regression (Distributed Processing, Step 1)
  :widths: 10 60
  :header-rows: 1

  * - Result ID
    - Result
  * - ``partialModel``
    - Pointer to the partial linear regression model that corresponds to the :math:`i`-th data block.

      The result can only be an object of the ``Model`` class.

.. _lin_ridge_step_2:

Step 2 - on Master Node
+++++++++++++++++++++++

.. figure:: images/distributed-step-2.png
  :width: 600
  :alt:

  Linear and Ridge Regression Training: Distributed Processing, Step 2 - on Master Node

In this step, linear and ridge regression training accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Training Input for Linear and Ridge Regression (Distributed Processing, Step 2)
  :widths: 10 60
  :header-rows: 1

  * - Input ID
    - Input
  * - ``partialModels``
    - A collection of partial models computed on local nodes in :ref:`Step 1 <lin_ridge_step_1>`.

      The collection contains objects of the ``Model`` class.

In this step, linear and ridge regression training calculates the result described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Training Output for Linear and Ridge Regression (Distributed Processing, Step 2)
  :widths: 10 60
  :header-rows: 1

  * - Result ID
    - Result
  * - ``model``
    - Pointer to the linear or ridge regression model being trained.

      The result can only be an object of the ``Model`` class.

Examples
========

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    -  :cpp_example:`lin_reg_norm_eq_dense_batch.cpp <linear_regression/lin_reg_norm_eq_dense_batch.cpp>`
    -  :cpp_example:`lin_reg_qr_dense_batch.cpp <linear_regression/lin_reg_qr_dense_batch.cpp>`
    -  :cpp_example:`ridge_reg_norm_eq_dense_batch.cpp <ridge_regression/ridge_reg_norm_eq_dense_batch.cpp>`


    Online Processing:

    - :cpp_example:`lin_reg_norm_eq_dense_online.cpp <linear_regression/lin_reg_norm_eq_dense_online.cpp>`
    - :cpp_example:`lin_reg_qr_dense_online.cpp <linear_regression/lin_reg_qr_dense_online.cpp>`
    - :cpp_example:`ridge_reg_norm_eq_dense_online.cpp <ridge_regression/ridge_reg_norm_eq_dense_online.cpp>`


    Distributed Processing:

    - :cpp_example:`lin_reg_norm_eq_dense_distr.cpp <linear_regression/>`
    - :cpp_example:`lin_reg_qr_dense_distr.cpp <linear_regression/>`
    - :cpp_example:`ridge_reg_norm_eq_dense_distr.cpp <ridge_regression/>`

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`linear_regression.py`
    - :daal4py_example:`ridge_regression.py`

    Online Processing:

    - :daal4py_example:`linear_regression_streaming.py`
    - :daal4py_example:`ridge_regression_streaming.py`

    Distributed Processing:

    - :daal4py_example:`linear_regression_spmd.py`
    - :daal4py_example:`ridge_regression_spmd.py`
