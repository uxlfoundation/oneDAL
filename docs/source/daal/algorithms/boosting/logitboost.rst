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

LogitBoost Classifier
=====================

LogitBoost is a boosting classification algorithm.
LogitBoost and AdaBoost are close to each other in the sense that both perform an additive logistic regression.
The difference is that AdaBoost minimizes the exponential loss, whereas LogitBoost minimizes the logistic loss.

LogitBoost within |short_name| implements a multi-class classifier.

Details
*******

Given :math:`n` feature vectors :math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})` of size :math:`p`
and a vector of class labels :math:`y= (y_1, \ldots, y_n)`, where :math:`y_i \in K = \{0, \ldots, J-1\}`
describes the class to which the feature vector :math:`x_i` belongs and :math:`J` is the number of classes,
the problem is to build a multi-class LogitBoost classifier.

Training Stage
--------------

The LogitBoost model is trained using the Friedman method [Friedman00]_.

Let :math:`y_{i,j} = I \{x_i \in j\}` is the indicator that the :math:`i`-th feature vector belongs to class :math:`j`.
The scheme below, which uses the stump weak learner, shows the major steps of the algorithm:

#. Start with weights :math:`w_{ij} = \frac{1}{n}`, :math:`F_j(x) = 0`, :math:`p_j(x) = \frac {1}{J}`, :math:`i = 1, \ldots, n`, :math:`j = 0, \ldots, J-1`.
#. For :math:`m = 1, \ldots, M`:

   Do

     For :math:`j = 1, \ldots, J`

     Do

     #. Compute working responses and weights in the j-th class:

        .. math::
            w_{ij} = p_i(x_i) (1 - p_i (x_i)), w_{ij} = max(z_{ij},\text{Thr1})

        .. math::
            z_{ij} = \frac {(y_{ij} - p_i(x_i))} {w_{ij}}, z_{ij} = \min(\max(z_{ij},-\text{Thr2}), \text{Thr2})

     #. Fit the function :math:`f_{mj}(x)` by a weighted least-squares regression of :math:`z_{ij}` to
        :math:`x_i` with weights :math:`w_{ij}` using the stump-based approach.

     End do

     :math:`f_{mj}(x) = \frac {J-1}{J} (f_{mj}(x) - \frac{1}{J} \sum _{k=1}^{J} f_{mk}(x))`

     :math:`F_j(x) = F_j(x) + f_{mj}(x)`

     :math:`p_j(x) = \frac {e^{F_j(x)}}{\sum _{k=1}^{J} e^{F_k(x)}}`

   End do


The result of the model training is a set of :math:`M` stumps.

Prediction Stage
----------------

Given the LogitBoost classifier and :math:`r` feature vectors :math:`x_1, \ldots, x_r`,
the problem is to calculate the labels :math:`\underset{j}{\mathrm{argmax}} F_j(x)` of the classes to which the feature vectors belong.

Batch Processing
****************

LogitBoost classifier follows the general workflow described in :ref:`classification_usage_model`.

Training
--------

For a description of the input and output, refer to :ref:`classification_usage_model`.

At the training stage, a LogitBoost classifier has the following parameters:

.. tabularcolumns::  |\Y{0.2}|\Y{0.2}|\Y{0.6}|

.. list-table:: Training Parameters for LogitBoost Classifier (Batch Processing)
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
     - The computation method used by the LogitBoost classifier. The only training method supported so far is the Friedman method.
   * - ``weakLearnerTraining``
     - **DEPRECATED**: Pointer to an object of the stump training class.

       **USE INSTEAD**: Pointer to an object of the regression stump training class.
     - **DEPRECATED**: Pointer to the training algorithm of the weak learner. By default, a stump weak learner is used.

       **USE INSTEAD**: Pointer to the regression training algorithm. By default, a regression stump with mse split criterion is used.
   * - ``weakLearnerPrediction``
     - **DEPRECATED**: Pointer to an object of the stump prediction class.

       **USE INSTEAD**: Pointer to an object of the regression stump prediction class.
     - **DEPRECATED**: Pointer to the prediction algorithm of the weak learner. By default, a stump weak learner is used.

       **USE INSTEAD**: Pointer to the regression prediction algorithm. By default, a regression stump with mse split criterion is used.
   * - ``accuracyThreshold``
     - :math:`0.01`
     - LogitBoost training accuracy.
   * - ``maxIterations``
     - :math:`100`
     - The maximal number of iterations for the LogitBoost algorithm.
   * - ``nClasses``
     - Not applicable
     - The number of classes, a required parameter.
   * - ``weightsDegenerateCasesThreshold``
     - :math:`1\mathrm{e}-10`
     - The threshold to avoid degenerate cases when calculating weights :math:`w_{ij}`.
   * - ``responsesDegenerateCasesThreshold``
     - :math:`1\mathrm{e}-10`
     - The threshold to avoid degenerate cases when calculating responses :math:`z_{ij}`.

Prediction
----------

For a description of the input and output, refer to :ref:`classification_usage_model`.

At the prediction stage, a LogitBoost classifier has the following parameters:

.. tabularcolumns::  |\Y{0.2}|\Y{0.2}|\Y{0.6}|

.. list-table:: Prediction Parameters for LogitBoost Classifier (Batch Processing)
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
     - Performance-oriented computation method, the only method supported by the LogitBoost classifier at the prediction stage.
   * - ``weakLearnerPrediction``
     - **DEPRECATED**: Pointer to an object of the stump prediction class.

       **USE INSTEAD**: Pointer to an object of the regression stump prediction class.
     - **DEPRECATED**: Pointer to the prediction algorithm of the weak learner. By default, a stump weak learner is used.

       **USE INSTEAD**: Pointer to the regression prediction algorithm. By default, a regression stump with mse split criterion is used.
   * - ``nClasses``
     - Not applicable
     - The number of classes, a required parameter.

.. note::

    The algorithm terminates if it achieves the specified accuracy or reaches the specified maximal number of iterations.
    To determine the actual number of iterations performed,
    call the ``getNumberOfWeakLearners()`` method of the ``LogitBoostModel`` class and divide it by ``nClasses``.

Examples
********

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`logitboost_dense_batch.cpp <boosting/logitboost_dense_batch.cpp>`

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`logitboost.py`
