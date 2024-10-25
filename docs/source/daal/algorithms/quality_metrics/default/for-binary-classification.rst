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

.. _quality_metrics_for_binary_classification:

Quality Metrics for Binary Classification Algorithms
====================================================

For two classes :math:`C_1` and :math:`C2`, given a vector :math:`X = (x_1, \ldots, x_n)` of class labels
computed at the prediction stage of the classification algorithm and
a vector :math:`Y = (y_1, \ldots, y_n)` of expected class labels, the problem is to evaluate the classifier
by computing the confusion matrix and connected quality metrics: precision, recall, and so on.

``QualityMetricsId`` for binary classification is ``confusionMatrix``.

Details
*******

Further definitions use the following notations:

.. tabularcolumns::  |\Y{0.2}|\Y{0.2}|\Y{0.6}|

.. list-table:: Notations for Quality Metrics for Binary Classification Algorithms
   :widths: 10 10 30
   :class: longtable

   * - :math:`\text{tp}`
     - true positive
     - the number of correctly recognized observations for class :math:`C_1`
   * - :math:`\text{tn}`
     - true negative
     - the number of correctly recognized observations that do not belong to the class :math:`C_1`
   * - :math:`\text{fp}`
     - false positive
     - the number of observations that were incorrectly assigned to the class :math:`C_1`
   * - :math:`\text{fn}`
     - false negative
     - the number of observations that were not recognized as belonging to the class :math:`C_1`

The library uses the following quality metrics for binary classifiers:

.. tabularcolumns::  |\Y{0.3}|\Y{0.7}|

.. list-table:: Definitions of Quality Metrics for Binary Classification Algorithms
   :widths: 10 10
   :header-rows: 1
   :class: longtable

   * - Quality Metric
     - Definition
   * - Accuracy
     - :math:`\frac {\text{tp} + \text{tn}}{\text{tp} + \text{fn} + \text{fp} + \text{tn}}`
   * - Precision
     - :math:`\frac {\text{tp}}{\text{tp} + \text{fp}}`
   * - Recall
     - :math:`\frac {\text{tp}}{\text{tp} + \text{fn}}`
   * - F-score
     - :math:`\frac {(\beta^2 + 1) \text{tp}}{(\beta^2 + 1) \text{tp} + \beta^2 \text{fn} + \text{fp}}`
   * - Specificity
     - :math:`\frac {\text{tn}}{\text{fp} + \text{tn}}`
   * - Area under curve (AUC)
     - :math:`\frac {1}{2}(\frac {\text{tp}}{\text{tp} + \text{fn}} + \frac {\text{tn}}{\text{tn} + \text{fp}})`

For more details of these metrics, including the evaluation focus, refer to [Sokolova09]_.

The confusion matrix is defined as follows:

.. tabularcolumns::  |\Y{0.3}|\Y{0.4}|\Y{0.3}|

.. list-table:: Confusion Matrix for Binary Classification Algorithms
    :header-rows: 1
    :stub-columns: 1

    * -
      - Classified as Class :math:`C_1`
      - Classified as Class :math:`C_2`
    * - Actual Class :math:`C_1`
      - `tp`
      - `fn`
    * - Actual Class :math:`C_2`
      - `fp`
      - `tn`

Batch Processing
****************

Algorithm Input
---------------

The quality metric algorithm for binary classifiers accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Quality Metrics for Binary Classification (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :class: longtable

   * - Input ID
     - Input
   * - ``predictedLabels``
     - Pointer to the :math:`n \times 1` numeric table that contains labels computed at the prediction stage of the classification algorithm.

       This input can be an object of any class derived from ``NumericTable`` except ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.
   * - ``groundTruthLabels``
     - Pointer to the :math:`n \times 1` numeric table that contains expected labels.

       This input can be an object of any class derived from ``NumericTable`` except ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.

Algorithm Parameters
--------------------

The quality metric algorithm has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for Quality Metrics for Binary Classification (Batch Processing)
   :header-rows: 1
   :widths: 10 10 60
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
     - Performance-oriented computation method, the only method supported by the algorithm.
   * - ``beta``
     - :math:`1`
     - The :math:`\beta` parameter of the F-score quality metric provided by the library.

Algorithm Output
----------------

The quality metric algorithm calculates the result described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Quality Metrics for Binary Classification (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :class: longtable

   * - Result ID
     - Result
   * - ``confusionMatrix``
     - Pointer to the :math:`2 \times 2` numeric table with the confusion matrix.

       .. note::
         By default, this result is an object of the ``HomogenNumericTable`` class,
         but you can define the result as an object of any class derived from NumericTable except ``PackedTriangularMatrix``, ``PackedSymmetricMatrix``, and ``CSRNumericTable``.
   * - ``binaryMetrics``
     - Pointer to the :math:`1 \times 6` numeric table that contains quality metrics, which you can access by an appropriate Binary Metrics ID:

       - ``accuracy`` - accuracy
       - ``precision`` - precision
       - ``recall`` - recall
       - ``fscore`` - F-score
       - ``specificity`` - specificity
       - ``AUC`` - area under the curve

       .. note::
           By default, this result is an object of the ``HomogenNumericTable`` class, but you can define the result as an object of any class
           derived from ``NumericTable`` except ``PackedTriangularMatrix``, ``PackedSymmetricMatrix``, and ``CSRNumericTable``.

Examples
********

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`svm_two_class_metrics_dense_batch.cpp <quality_metrics/svm_two_class_metrics_dense_batch.cpp>`

