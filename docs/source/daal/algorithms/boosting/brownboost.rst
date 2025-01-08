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

BrownBoost Classifier
=====================

BrownBoost is a boosting classification algorithm.
It is more robust to noisy data sets than other boosting classification algorithms [Freund99]_.

BrownBoost is a binary classifier.
For a multi-class case, use :ref:`svm_multi_class` framework of the library.

Details
*******

Given :math:`n` feature vectors :math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})` of size :math:`p`
and a vector of class labels :math:`y= (y_1, \ldots, y_n)`, where :math:`y_i \in K = \{-1, 1\}` describes the class
to which the feature vector :math:`x_i` belongs, and a weak learner algorithm,
the problem is to build a two-class BrownBoost classifier.

Training Stage
--------------

The model is trained using the Freund method [Freund01]_ as follows:

#. Calculate :math:`c = \mathrm{erfinv}^2(1 - \varepsilon)`, where:

   - :math:`\mathrm{erfinv}(x)` is an inverse error function,
   - :math:`\varepsilon` is a target classification error of the algorithm defined as
     :math:`\frac {1}{n} \sum _{i=1}^{n} |p(x_i) - y_i|`
   - :math:`p(x) = \text{erf} \left(\frac {\sum _{i=1}^{M} \alpha_i h_i(x)}{\sqrt{c}}\right)`
   - :math:`\mathrm{erf}(x)` is the error function,
   - :math:`h_i(x)` is a hypothesis formulated by the :math:`i`-th weak learner, :math:`i = 1, \ldots, M`,
   - :math:`\alpha_i` is the weight of the hypothesis.
#. Set initial prediction values: :math:`r_1(x, y) = 0`.
#. Set "remaining timing": :math:`s_1 = c`.
#. Do for :math:`i=1, 2, \ldots` until :math:`s_{i+1} \leq 0`

   #. With each feature vector and its label of positive weight, associate :math:`W_i(x, y) = e^{\frac {-(r_i(x, y) + s_i)^2}{c}}`.
   #. Call the weak learner with the distribution defined by normalizing :math:`W_i(x, y)` to receive a hypothesis :math:`h_i(x)`.
   #. Solve the differential equation

      .. math::
        \frac {dt}{d\alpha} = \gamma =
        \frac {\sum _{(x,y)} \exp (-\frac{1}{c} (r_i(x, y) + \alpha h_i(x) y + s_i - t)^2)h_i(x)y}
        {\sum _{(x,y)} \exp (-\frac{1}{c} (r_i(x, y) + \alpha h_i(x) y + s_i - t)^2)}

      with given boundary conditions :math:`t = 0` and :math:`\alpha = 0` to find :math:`t_i = t^{*} > 0`
      and :math:`\alpha_i = \alpha^{*}` such that either :math:`\gamma \leq ν` or :math:`t^{*} = s_i`,
      where :math:`ν` is a given small constant needed to avoid degenerate cases.
   #. Update the prediction values: :math:`r_{i+1}(x, y) = r_i(x, y) + \alpha_i h_i(x) y`.
   #. Update "remaining time": :math:`s_{i+1} = s_i - t_i`.

   End do

The result of the model training is the array of :math:`M` weak learners :math:`h_i`.

Prediction Stage
----------------

Given the BrownBoost classifier and :math:`r` feature vectors :math:`x_1, \ldots, x_r`,
the problem is to calculate the final classification confidence, a number from the interval :math:`[-1, 1]`, using the rule:

.. math::
    p(x) = \text{erf} \left(\frac {\sum _{i=1}^{M} \alpha_i h_i (x)}{\sqrt{c}}\right)

Batch Processing
****************

BrownBoost classifier follows the general workflow described in :ref:`classification_usage_model`.

Training
--------

For a description of the input and output, refer to :ref:`classification_usage_model`.

At the training stage, a BrownBoost classifier has the following parameters:

.. tabularcolumns::  |\Y{0.2}|\Y{0.2}|\Y{0.6}|

.. list-table:: Training Parameters for BrownBoost Classifier (Batch Processing)
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
     - The computation method used by the BrownBoost classifier. The only training method supported so far is the Y. Freund's method.
   * - ``nClasses``
     - :math:`2`
     - The number of classes.
   * - ``weakLearnerTraining``
     - **DEPRECATED**: Pointer to an object of the weak learner training class

       **USE INSTEAD**: Pointer to an object of the classification stump training class
     - **DEPRECATED**: Pointer to the training algorithm of the weak learner. By default, a stump weak learner is used.

       **USE INSTEAD**: Pointer to the classifier training algorithm. Be default, a classification stump with gini split criterion is used.
   * - ``weakLearnerPrediction``
     - **DEPRECATED**: Pointer to an object of the weak learner prediction class

       **USE INSTEAD**: Pointer to an object of the classification stump prediction class
     - **DEPRECATED**: Pointer to the prediction algorithm of the weak learner. By default, a stump weak learner is used.

       **USE INSTEAD**: Pointer to the classifier prediction algorithm. Be default, a classification stump with gini split criterion is used.
   * - ``accuracyThreshold``
     - :math:`0.01`
     - BrownBoost training accuracy :math:`\varepsilon`.
   * - ``maxIterations``
     - :math:`100`
     - The maximal number of iterations for the BrownBoost algorithm.
   * - ``newtonRaphsonAccuracyThreshold``
     - :math:`1.0\mathrm{e}-3`
     - Accuracy threshold of the Newton-Raphson method used underneath the BrownBoost algorithm.
   * - ``newtonRaphsonMaxIterations``
     - :math:`100`
     - The maximal number of Newton-Raphson iterations in the algorithm.
   * - ``degenerateCasesThreshold``
     - :math:`1.0\mathrm{e}-2`
     - The threshold used to avoid degenerate cases.

Prediction
----------

For a description of the input and output, refer to :ref:`classification_usage_model`.

At the prediction stage, a BrownBoost classifier has the following parameters:

.. tabularcolumns::  |\Y{0.2}|\Y{0.2}|\Y{0.6}|

.. list-table:: Prediction Parameters for BrownBoost Classifier (Batch Processing)
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
     - Performance-oriented computation method, the only method supported by the BrownBoost classifier.
   * - ``nClasses``
     - :math:`2`
     - The number of classes.
   * - ``weakLearnerPrediction``
     - **DEPRECATED**: Pointer to an object of the weak learner prediction class

       **USE INSTEAD**: Pointer to an object of the classification stump prediction class
     - **DEPRECATED**: Pointer to the prediction algorithm of the weak learner. By default, a stump weak learner is used.

       **USE INSTEAD**: Pointer to the classifier prediction algorithm. Be default, a classification stump with gini split criterion is used.
   * - ``accuracyThreshold``
     - :math:`0.01`
     - BrownBoost training accuracy :math:`\varepsilon`.

Examples
********

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`brownboost_dense_batch.cpp <boosting/brownboost_dense_batch.cpp>`

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`brownboost.py`
