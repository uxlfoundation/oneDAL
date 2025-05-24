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

Regression Stump
================

A Regression Decision Stump is a model that consists of a one-level decision tree
where the root is connected to terminal nodes (leaves) [Friedman2017]_.
The library only supports stumps with two leaves based on regression decision trees.
The one method of split criteria is available: mse.
See :ref:`dt_regression` for details.

Batch Processing
****************

A regression stump follows the general workflow described in :ref:`regression_usage_model`.

Training
--------

For a description of the input and output, refer to :ref:`regression_usage_model`.

At the training stage, a regression decision stump has the following parameters:

.. tabularcolumns::  |\Y{0.2}|\Y{0.2}|\Y{0.6}|

.. list-table:: Training Parameters for Regression Stump (Batch Processing)
   :widths: 20 20 60
   :header-rows: 1
   :class: longtable

   * - Parameter
     - Default Value
     - Description
   * - ``algorithmFPType``
     - ``float``
     - The floating-point type that the algorithm uses for intermediate computations. Can be ``float`` or ``double``.
   * - ``method``
     - ``defaultDense``
     - Performance-oriented computation method, the only method supported by the algorithm.
   * - ``varImportance``
     - ``none``
     - .. note:: Variable importance computation is not supported for current version of the library.

Prediction
----------

For a description of the input and output, refer to :ref:`regression_usage_model`.

At the prediction stage, a regression stump has the following parameters:

.. tabularcolumns::  |\Y{0.2}|\Y{0.2}|\Y{0.6}|

.. list-table:: Prediction Parameters for Regression Stump (Batch Processing)
   :widths: 20 20 60
   :header-rows: 1
   :class: longtable

   * - Parameter
     - Default Value
     - Description
   * - ``algorithmFPType``
     - ``float``
     - The floating-point type that the algorithm uses for intermediate computations. Can be ``float`` or ``double``.
   * - ``method``
     - ``defaultDense``
     - Performance-oriented computation method, the only method supported by the algorithm.

Examples
********

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    :cpp_example:`stump_reg_mse_dense_batch.cpp <stump/stump_reg_mse_dense_batch.cpp>`

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`stump_regression.py`
