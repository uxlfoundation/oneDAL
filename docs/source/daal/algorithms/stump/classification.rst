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

Classification Stump
====================

A Classification Decision Stump is a model that consists of a one-level decision tree
where the root is connected to terminal nodes (leaves) [Friedman2017]_.
The library only supports stumps with two leaves.
Two methods of split criterion are available: gini and information gain.
See :ref:`dt_classification` for details.

Batch Processing
****************

A classification stump follows the general workflow described in :ref:`classification_usage_model`.

Training
--------

For a description of the input and output, refer to :ref:`classification_usage_model`.

At the training stage, a classification decision stump has the following parameters:

.. tabularcolumns::  |\Y{0.2}|\Y{0.2}|\Y{0.6}|

.. list-table:: Training Parameters for Classification Stump (Batch Processing)
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
   * - ``splitCriterion``
     - ``decision_tree::classification::gini``
     - Split criteria for classification stump. Two split criterion are available:

       - ``decision_tree::classification::gini``
       - ``decision_tree::classification::infoGain``

       See :ref:`dt_classification` chapter for details.

   * - ``varImportance``
     - ``none``
     - .. note:: Variable importance computation is not supported for current version of the library.
   * - ``nClasses``
     - :math:`2`
     - The number of classes.


Prediction
----------

For a description of the input and output, refer to :ref:`classification_usage_model`.

At the prediction stage, a classification stump has the following parameters:

.. tabularcolumns::  |\Y{0.2}|\Y{0.2}|\Y{0.6}|

.. list-table:: Training Parameters for Classification Stump (Batch Processing)
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
   * - ``nClasses``
     - :math:`2`
     - The number of classes.
   * - ``resultsToEvaluate``
     - ``classifier::computeClassLabels``
     - The form of computed result:

       - ``classifier::computeClassLabels`` – the result contains the ``NumericTable``
         of size :math:`n \times 1` with predicted labels

       - ``classifier::computeClassProbabilities`` – the result contains the ``NumericTable``
         of size :math:`n \times \text{nClasses}` with probabilities to belong to each class

Examples
********

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`stump_cls_gini_dense_batch.cpp <stump/stump_cls_gini_dense_batch.cpp>`
    - :cpp_example:`stump_cls_infogain_dense_batch.cpp <stump/stump_cls_infogain_dense_batch.cpp>`

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`stump_classification.py`
