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

AdaBoost Multiclass Classifier
==============================

AdaBoost (short for "Adaptive Boosting") is a popular boosting classification algorithm.
The AdaBoost algorithm performs well on a variety of data sets except some noisy data ([Friedman98]_, [Zhu2005]_).
The library supports two methods for the algorithms:

- SAMME, or Stagewise Additive Modeling using a Multi-class Exponential loss function [Zhu2005]_
- SAMME.R, which is a modification of SAMME method for Real-valued returned probabilities from weak learner

Details
*******

Given :math:`n` feature vectors :math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots x_n = (x_{n1}, \ldots, x_{np})`
of size :math:`p`, a vector of class labels :math:`y = (y_1, \ldots, y_n)`
where :math:`y_i \in K = \{-1, 1\}` in case of binary classification and
:math:`y_i \in K = \{ 0, \ldots, C-1 \}`, where :math:`C` is a number of classes,
describes the class :math:`t` the feature vector :math:`x_i` belongs to,
and :math:`h_t` is a weak learner algorithm, the problem is to build an AdaBoost classifier.

Training Stage
**************

.. tabs::

    .. group-tab:: SAMME method

        The following scheme shows the major steps of the SAMME algorithm:

        #. Initialize weights :math:`D_1(i) = \frac{1}{n}` for :math:`i = 1, \ldots, n`
        #. For :math:`t = 1, \ldots, T`:

           - Train the weak learner :math:`h_t(i)` using weights :math:`D_t`.
           - Choose a confidence value :math:`\alpha_t = \log \frac{1 - \mathrm{err}_t}{\mathrm{err}_t} + \log(C-1)`,
             where :math:`\mathrm{err}_t = \frac{\sum_{i=1}{n} D_t(i) I(y_i \neq h_t(i))}{\sum_{i=1}{n} D_t(i)}`
           - Update :math:`D_{t+1}(i) = \frac{D_t{i} \mathrm{exp}(-\alpha_t I(y_i \neq h_t(i)))}{Z_t}`,
             where :math:`Z_t` is a normalization factor.

        #. Output the final hypothesis:

           .. math::
                H(x) = \underset{k} {\mathrm{argmax}} \sum_{t=1}^{T} \alpha_t I(h_t{x} = k)

        .. note:: SAMME algorithm in case of binary classification is equal to the AdaBoost algorithm from [Friedman98]_.

    .. group-tab:: SAMME.R method

        The following scheme shows the major steps of the SAMME.R algorithm:

        #. Initialize weights :math:`D_1(i) = \frac{1}{n}` for :math:`i = 1, \ldots, n`
        #. For :math:`t = 1, \ldots, T`:

           - Train the weak learner :math:`h_t(i)` using weights :math:`D_t`.
           - Receive the weighed class probability estimates from weak learner:

             .. math::
                p_k^t(x) = \mathrm{Prob}_w \{ c = k | x\},
                k = 0, \ldots, C-1

           - For :math:`k = 0, \ldots, C-1`, set :math:`s_k^t(x)`:

             .. math::
                s_k^t(x) = (C-1)
                \left( \log p_k^t(x) - \frac{1}{C} \sum_{k=0}^{C-1} \log p_k^t(x) \right)

           - For :math:`i = 1, \ldots, n`, update :math:`D_{t+1}(i)`:

             .. math::
                D_{t+1}(i) = \frac{1}{Z_t} \mathrm{exp}
                \left(- \frac{C-1}{C} z_i^T \log p^t (x) \right)

             where :math:`Z_t` is a normalization factor, :math:`z_i = (z_{i1}, \ldots, z_{iC})`,
             :math:`z_{ik} = \begin{cases}
             1, & k = y_i \\ - \frac{1}{K-1}, & k \neq y_i \end{cases}`

        #. Output the final hypothesis:

           .. math::
                H(x) = \underset{k} {\mathrm{argmax}} \sum_{t=1}^{T} s_k^t(x)

Prediction Stage
----------------

.. tabs::

    .. group-tab:: SAMME method

        Given the AdaBoost classifier and :math:`r` feature vectors :math:`x_1, \ldots, x_r`,
        the problem is to calculate the final class :math:`H(x)`:

        .. math::
            H(x) = \underset{k} {\mathrm{argmax}} \sum_{t=1}^{T} \alpha_t I(h_t{x} = k)

    .. group-tab:: SAMME.R method

        Given the AdaBoost classifier and :math:`r` feature vectors :math:`x_1, \ldots, x_r`,
        the problem is to calculate the final class :math:`H(x)`:

        .. math::
            H(x) = \underset{k} {\mathrm{argmax}} \sum_{t=1}^{T} s_k^t(x)

        where :math:`s_k^t(x)` is as defined above in `Training Stage`_.

Batch Processing
****************

AdaBoost classifier follows the general workflow described in :ref:`classification_usage_model`.

Training
--------

For a description of the input and output, refer to :ref:`classification_usage_model`.
At the training stage, an AdaBoost classifier has the following parameters:

.. tabularcolumns::  |\Y{0.2}|\Y{0.2}|\Y{0.6}|

.. list-table:: Training Parameters for AdaBoost Multiclass Classifier (Batch Processing)
   :header-rows: 1
   :widths: 10 20 30
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
     - Available methods for computation of the AdaBoost algorithm:

       - ``samme`` - uses the classifier that returns labels as weak learner
       - ``sammeR`` - uses the classifier that returns probabilities of belonging to class as weak learner
       - ``defaultDense`` is equal to ``samme`` method
   * - ``weakLearnerTraining``
     - Pointer to an object of the classification stump training class
     - Pointer to the training algorithm of the weak learner. By default, a classification stump weak learner is used.
   * - ``weakLearnerPrediction``
     - Pointer to an object of the classification stump prediction class
     - Pointer to the prediction algorithm of the weak learner. By default, a classification stump weak learner is used.
   * - ``accuracyThreshold``
     - :math:`0.01`
     - AdaBoost training accuracy.
   * - ``maxIterations``
     - :math:`100`
     - The maximal number of iterations for the algorithm.
   * - ``learningRate``
     - :math:`1.0`
     - Multiplier for each classifier to shrink its contribution.
   * - ``nClasses``
     - :math:`2`
     - The number of classes.
   * - ``resultsToCompute``
     - :math:`0`
     - The 64-bit integer flag that specifies which extra characteristics of AdaBoost to compute.
       Current version of the library only provides the following option: ``computeWeakLearnersErrors``

Output
------

In addition to classifier output, AdaBoost calculates the result described below.
Pass the ``Result ID`` as a parameter to the methods that access the result of your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Training Output for AdaBoost Multiclass Classifier (Batch Processing)
   :header-rows: 1
   :widths: 10 60
   :align: left

   * - Result ID
     - Result
   * - ``weakLearnersErrors``
     - A numeric table :math:`1 \times \mathrm{maxIterations}` containing weak learner's classification errors
       computed when the ``computeWeakLearnersErrors`` option is on.

       .. note::
            By default, this result is an object of the ``HomogenNumericTable`` class,
            but you can define the result as an object of any class derived from ``NumericTable``.

Prediction
----------

For a description of the input and output, refer to :ref:`classification_usage_model`.
At the prediction stage, an AdaBoost classifier has the following parameters:

.. tabularcolumns::  |\Y{0.2}|\Y{0.2}|\Y{0.6}|

.. list-table:: Prediction Parameters for AdaBoost Multiclass Classifier (Batch Processing)
   :header-rows: 1
   :widths: 10 20 30
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
     - Performance-oriented computation method, the only method supported by the AdaBoost classifier at the prediction stage.
   * - ``weakLearnerPrediction``
     - Pointer to an object of the classification stump prediction class
     - Pointer to the prediction algorithm of the weak learner. By default, a classification stump weak learner is used.
   * - ``nClasses``
     - :math:`2`
     - The number of classes.


Examples
********

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`adaboost_samme_two_class_batch.cpp <boosting/adaboost_samme_two_class_batch.cpp>`
    - :cpp_example:`adaboost_sammer_two_class_batch.cpp <boosting/adaboost_sammer_two_class_batch.cpp>`
    - :cpp_example:`adaboost_samme_multi_class_batch.cpp <boosting/adaboost_samme_multi_class_batch.cpp>`
    - :cpp_example:`adaboost_sammer_multi_class_batch.cpp <boosting/adaboost_sammer_multi_class_batch.cpp>`
