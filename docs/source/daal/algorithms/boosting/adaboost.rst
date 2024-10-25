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

AdaBoost Classifier
===================

AdaBoost (short for "Adaptive Boosting") is a popular boosting classification algorithm.
AdaBoost algorithm performs well on a variety of data sets except some noisy data [Freund99]_.

AdaBoost is a binary classifier.
For a multi-class case, use :ref:`svm_multi_class` framework of the library.

Details
*******

Given :math:`n` feature vectors :math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})` of size :math:`p`
and a vector of class labels :math:`y= (y_1, \ldots, y_n)`, where :math:`y_i \in K = \{-1, 1\}`
describes the class to which the feature vector :math:`x_i` belongs, and a weak learner algorithm,
the problem is to build an AdaBoost classifier.

Training Stage
--------------

The following scheme shows the major steps of the algorithm:

#. Initialize weights :math:`D_1(i) = \frac{1}{n}` for :math:`i = 1, \ldots, n`.

#. For :math:`t = 1, \ldots, T`:

   #. Train the weak learner :math:`h_t(t) \in \{-1, 1\}` using weights :math:`D_t.`
   #. Choose a confidence value :math:`\alpha_t`.
   #. Update :math:`D_{t+1}(i) = \frac {D_t(i)\exp(-\alpha_t Y_i h_t(x_i))} {Z_t}`, where :math:`Z_t` is a normalization factor.

#. Output the final hypothesis:

   .. math::
        H(x_i) = \mathrm{sign} \left( \sum _{t=1}^{T} \alpha_t h_t(x_i)\right)

Prediction Stage
----------------

Given the AdaBoost classifier and :math:`r` feature vectors :math:`x_1, \ldots, x_r`, the problem is to calculate the final class:

.. math::
    H(x_i) = \mathrm{sign} \left( \sum _{t=1}^{T} \alpha_t h_t(x_i)\right)

Batch Processing
****************

AdaBoost classifier follows the general workflow described in :ref:`classification_usage_model`.

Training
--------

For a description of the input and output, refer to :ref:`classification_usage_model`.

At the training stage, an AdaBoost classifier has the following parameters:

.. tabularcolumns::  |\Y{0.2}|\Y{0.2}|\Y{0.6}|

.. list-table:: Training Parameters for AdaBoost Classifier (Batch Processing)
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
     - The computation method used by the AdaBoost classifier. The only training method supported so far is the Y. Freund's method.
   * - ``weakLearnerTraining``
     - Pointer to an object of the stump training class
     - Pointer to the training algorithm of the weak learner. By default, a stump weak learner is used.
   * - ``weakLearnerPrediction``
     - Pointer to an object of the stump prediction class
     - Pointer to the prediction algorithm of the weak learner. By default, a stump weak learner is used.
   * - ``accuracyThreshold``
     - :math:`0.01`
     - AdaBoost training accuracy.
   * - ``maxIterations``
     - :math:`100`
     - The maximal number of iterations for the algorithm.


Prediction
----------

For a description of the input and output, refer to :ref:`classification_usage_model`.

At the prediction stage, an AdaBoost classifier has the following parameters:

.. tabularcolumns::  |\Y{0.2}|\Y{0.2}|\Y{0.6}|

.. list-table:: Prediction Parameters for AdaBoost Classifier (Batch Processing)
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
     - Pointer to an object of the stump prediction class
     - Pointer to the prediction algorithm of the weak learner. By default, a stump weak learner is used.

Examples
********

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`adaboost_dense_batch.cpp <boosting/adaboost_dense_batch.cpp>`

  .. tab:: Python*

    - :daal4py_example:`adaboost.py`
