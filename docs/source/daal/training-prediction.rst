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

Training and Prediction
=======================

Training and prediction algorithms in |full_name| (|short_name|) include a range of popular machine learning algorithms:

.. toctree::
   :maxdepth: 1

   algorithms/decision_forest/index
   algorithms/decision_tree/index
   algorithms/gradient_boosted_trees/index
   algorithms/stump/index.rst
   algorithms/linear_ridge_regression/index.rst
   algorithms/lasso_elastic_net/index.rst
   algorithms/k_nearest_neighbors/k-nearest-neighbors-knn-classifier.rst
   algorithms/implicit_als/implicit-alternating-least-squares.rst
   algorithms/logistic_regression/logistic-regression.rst
   algorithms/naive_bayes/naive-bayes-classifier.rst
   algorithms/svm/support-vector-machine-classifier.rst
   algorithms/svm_multi_class/multi-class-classifier.rst
   algorithms/boosting/index.rst

Unlike :ref:`analysis` algorithms, which are intended to characterize the structure of data sets, machine learning algorithms model the data.
Modeling operates in two major stages:

- **Training**, when the algorithm estimates model parameters based on a training data set.
- **Prediction or decision making**, when the algorithm uses the trained model to predict the outcome based on new data.

Training is typically a much more computationally complex problem than prediction.
Therefore, certain end-to-end analytics usage scenarios require that training and prediction phases are done on different devices,
the training is done on more powerful devices, while prediction is done on smaller devices.
Because smaller devices may have stricter memory footprint requirements,
|short_name| separates Training, Prediction, and respective Model in three different class hierarchies to minimize the footprint.

Training Alternative
********************

An alternative to training your model with algorithms implemented in |short_name|
is to build a trained model from pre-calculated model parameters, for example,
coefficients :math:`\beta` for Linear Regression.
This enables you to use |short_name| only to get predictions based on the model parameters computed elsewhere.

The Model Builder class provides an interface for adding all the necessary parameters and building a trained model
ready for the prediction stage.

The following schema illustrates the use of Model Builder class:

.. figure:: images/model_builders.png
   :width: 600
   :alt:

The Model Builder class is implemented for the following algorithms:

- :ref:`linear_regression`
- :ref:`svm`
- :ref:`svm_multi_class`
- :ref:`logistic_regression`
- :ref:`gbt_regression`
- :ref:`gbt_classification`
- :ref:`df_classification`
