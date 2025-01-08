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

.. _quality_metrics_for_multi_class_classification:

Quality Metrics for Multi-class Classification Algorithms
=========================================================

For :math:`l` classes :math:`C_1, \ldots, C_l`, given a vector :math:`X = (x_1, \ldots, x_n)`
of class labels computed at the prediction stage of the classification algorithm and
a vector :math:`Y = (y_1, \ldots, y_n)` of expected class labels,
the problem is to evaluate the classifier by computing the confusion matrix and
connected quality metrics: precision, error rate, and so on.

``QualityMetricsId`` for multi-class classification is ``confusionMatrix``.

Details
*******

Further definitions use the following notations:

.. tabularcolumns::  |\Y{0.2}|\Y{0.2}|\Y{0.6}|

.. list-table:: Notations for Quality Metrics for Multi-class Classification Algorithms
   :widths: 10 10 30
   :class: longtable

   * - :math:`\text{tp}_i`
     - true positive
     - the number of correctly recognized observations for class :math:`C_1`
   * - :math:`\text{tn}_i`
     - true negative
     - the number of correctly recognized observations that do not belong to the class :math:`C_1`
   * - :math:`\text{fp}_i`
     - false positive
     - the number of observations that were incorrectly assigned to the class :math:`C_1`
   * - :math:`\text{fn}_i`
     - false negative
     - the number of observations that were not recognized as belonging to the class :math:`C_1`

The library uses the following quality metrics for multi-class classifiers:

.. tabularcolumns::  |\Y{0.3}|\Y{0.7}|

.. list-table:: Definitions of Quality Metrics for Multi-class Classification Algorithms
   :widths: 10 10
   :header-rows: 1
   :class: longtable

   * - Quality Metric
     - Definition
   * - Average accuracy
     - :math:`\frac {\sum _{i = 1}^{l} \frac {\text{tp}_i + \text{tn}_i}{\text{tp}_i + \text{fn}_i + \text{fp}_i + \text{tn}_i}}{l}`
   * - Error rate
     - :math:`\frac {\sum _{i = 1}^{l} \frac {\text{fp}_i + \text{fn}_i}{\text{tp}_i + \text{fn}_i + \text{fp}_i + \text{tn}_i}}{l}`
   * - Micro precision (:math:`\text{Precision}_\mu`)
     - :math:`\frac {\sum _{i = 1}^{l} \text{tp}_i} {\sum _{i = 1}^{l} (\text{tp}_i + \text{fp}_i)}`
   * - Micro recall (:math:`\text{Recall}_\mu`)
     - :math:`\frac {\sum _{i = 1}^{l} \text{tp}_i} {\sum _{i = 1}^{l} (\text{tp}_i + \text{fn}_i)}`
   * - Micro F-score (:math:`\text{F-score}_\mu`)
     - :math:`\frac {(\beta^2 + 1)(\text{Precision}_\mu \times \text{Recall}_\mu)}{\beta^2 \times \text{Precision}_\mu + \text{Recall}_\mu}`
   * - Macro precision (:math:`\text{Precision}_M`)
     - :math:`\frac {\sum _{i = 1}^{l} \frac {\text{tp}_i}{\text{tp}_i + \text{fp}_i}}{l}`
   * - Macro recall (:math:`\text{Recall}_M`)
     - :math:`\frac {\sum _{i = 1}^{l} \frac {\text{tp}_i}{\text{tp}_i + \text{fn}_i}}{l}`
   * - Macro F-score (:math:`\text{F-score}_M`)
     - :math:`\frac {(\beta^2 + 1)(\text{Precision}_M \times \text{Recall}_M)}{\beta^2 \times \text{Precision}_M + \text{Recall}_M}`

For more details of these metrics, including the evaluation focus, refer to [Sokolova09]_.

The following is the confusion matrix:

.. list-table:: Confusion Matrix for Multi-class Classification Algorithms
    :header-rows: 1
    :stub-columns: 1

    * -
      - Classified as Class :math:`C_1`
      - :math:`\ldots`
      - Classified as Class :math:`C_i`
      - :math:`\ldots`
      - Classified as Class :math:`C_l`
    * - Actual Class :math:`C_1`
      - :math:`c_{11}`
      - :math:`\ldots`
      - :math:`c_{1i}`
      - :math:`\ldots`
      - :math:`c_{1l}`
    * - :math:`\ldots`
      - :math:`\ldots`
      - :math:`\ldots`
      - :math:`\ldots`
      - :math:`\ldots`
      - :math:`\ldots`
    * - Actual Class :math:`C_i`
      - :math:`c_{i1}`
      - :math:`\ldots`
      - :math:`c_{ii}`
      - :math:`\ldots`
      - :math:`c_{il}`
    * - :math:`\ldots`
      - :math:`\ldots`
      - :math:`\ldots`
      - :math:`\ldots`
      - :math:`\ldots`
      - :math:`\ldots`
    * - Actual Class :math:`C_l`
      - :math:`c_{l1}`
      - :math:`\ldots`
      - :math:`c_{li}`
      - :math:`\ldots`
      - :math:`c_{ll}`

The positives and negatives are defined through elements of the confusion matrix as follows:

.. math::
    \text{tp}_i = c_{ii}

.. math::
    \text{fp}_i = \sum _{n = 1}^{l} c_{ni} - \text{tp}_i

.. math::
    \text{fn}_i = \sum _{n = 1}^{l} c_{in} - \text{tp}_i

.. math::
    \text{tn}_i = \sum _{n = 1}^{l} \sum _{k = 1}^{l} c_{nk} - \text{tp}_i - \text{fp}_i - \text{fn}_i

Batch Processing
****************

Algorithm Input
---------------

The quality metric algorithm for multi-class classifiers accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Quality Metrics for Multi-class Classification Algorithms (Batch Processing)
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

       This input can be an object of any class derived from NumericTable except ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.

Algorithm Parameters
--------------------

The quality metric algorithm has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for Quality Metrics for Multi-class Classification Algorithms (Batch Processing)
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
   * - ``nClasses``
     - :math:`0`
     - 	The number of classes (:math:`l`).
   * - ``useDefaultMetrics``
     - ``true``
     - A flag that defines a need to compute the default metrics provided by the library.
   * - ``beta``
     - :math:`1`
     - The :math:`\beta` parameter of the F-score quality metric provided by the library.

Algorithm Output
----------------

The quality metric algorithm calculates the result described below. Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm. For more details, see Algorithms.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Quality Metrics for Multi-class Classification Algorithms (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :class: longtable

   * - Result ID
     - Result
   * - ``confusionMatrix``
     - Pointer to the :math:`\text{nClasses} \times \text{nClasses}` numeric table with the confusion matrix.

       .. note::
          By default, this result is an object of the ``HomogenNumericTable`` class, but you can define the result as an object
          of any class derived from NumericTable except ``PackedTriangularMatrix``, ``PackedSymmetricMatrix``, and ``CSRNumericTable``.
   * - ``multiClassMetrics``
     - Pointer to the :math:`1 \times 8` numeric table that contains quality metrics, which you can access by an appropriate Multi-class Metrics ID:

       - ``averageAccuracy`` - average accuracy
       - ``errorRate`` - error rate
       - ``microPrecision`` - micro precision
       - ``microRecall`` - micro recall
       - ``microFscore`` - micro F-score
       - ``macroPrecision`` - macro precision
       - ``macroRecall`` - macro recall
       - ``macroFscore`` - macro F-score

       .. note::
          By default, this result is an object of the ``HomogenNumericTable`` class, but you can define the result as an object
          of any class derived from NumericTable except ``PackedTriangularMatrix``, ``PackedSymmetricMatrix``, and ``CSRNumericTable``.

Examples
********

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`svm_multi_class_metrics_dense_batch.cpp <quality_metrics/svm_multi_class_metrics_dense_batch.cpp>`
