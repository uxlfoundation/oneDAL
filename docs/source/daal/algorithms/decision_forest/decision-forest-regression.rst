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

.. _df_regression:

Regression Decision Forest
--------------------------

.. toctree::
   :glob:
   :maxdepth: 4

Decision forest regression is a special case of the :ref:`decision_forest` model.

Details
=======

Given:

- :math:`n` feature vectors :math:`X = \{x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np}) \}`
  of size :math:`p`;
- their non-negative sample weights :math:`w = (w_1, \ldots, w_n)`;
- the vector of responses :math:`y = (y_1, \ldots, y_n)`

The problem is to build a decision forest regression model that minimizes the Mean-Square Error (MSE) between the predicted and true value.

Training Stage
**************

Decision forest regression follows the algorithmic framework of decision forest training algorithm
based on the mean-squared error (MSE) [Breiman84]_.
If sample weights are provided as input, the library uses a weighted version of the algorithm.

MSE is an impurity metric (:math:`D` is a set of observations that reach the node), calculated as follows:

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Decision Forest Regression: impurity calculations
   :widths: 10 10
   :header-rows: 1
   :align: left
   :class: longtable

   * - Without sample weights
     - With sample weights
   * - :math:`I_{\mathrm{MSE}}\left(D\right) = \frac{1}{W(D)} \sum _{i=1}^{W(D)}{\left(y_i - \frac{1}{W(D)} \sum _{j=1}^{W(D)} y_j \right)}^{2}`
     - :math:`I_{\mathrm{MSE}}\left(D\right) = \frac{1}{W(D)} \sum _{i \in D}{w_i \left(y_i - \frac{1}{W(D)} \sum _{j \in D} w_j y_j \right)}^{2}`
   * - :math:`W(S) = \sum_{s \in S} 1`, which is equivalent to the number of elements in :math:`S`
     - :math:`W(S) = \sum_{s \in S} w_s`


Prediction Stage
****************

Given decision forest regression model and vectors :math:`x_1, \ldots, x_r`, the problem is to calculate the responses for those
vectors. To solve the problem for each given query vector :math:`x_i`, the algorithm finds the leaf node in a tree in the
forest that gives the response by that tree as the mean of
dependent variables. The forest predicts the response as the mean
of responses from trees.


Out-of-bag Error
****************

Decision forest regression follows the algorithmic framework for
calculating the decision forest out-of-bag (OOB) error, where
aggregation of the out-of-bag predictions in all trees and
calculation of the OOB error of the decision forest is done as
follows:

-  For each vector :math:`x_i` in the dataset :math:`X`, predict its response :math:`\hat{y_i}`
   as the mean of prediction from the trees that contain :math:`x_i` in their OOB set:

   :math:`\hat{y_i} = \frac{1}{{|B}_{i}|}\sum _{b=1}^{|B_i|}\hat{y_{ib}}`, where :math:`B_i= \bigcup{T_b}: x_i \in \overline{D_b}` and :math:`\hat{y_{ib}}` is the result of prediction
   :math:`x_i` by :math:`T_b`.

-  Calculate the OOB error of the decision forest T as the Mean-Square Error (MSE):

   .. math::
		OOB(T) = \frac{1}{|{D}^{\text{'}}|}\sum _{{y}_{i} \in {D}^{\text{'}}}\sum {(y_i-\hat{y_i})}^{2}, \text{where } {D}^{\text{'}}={\bigcup}_{b=1}^{B}\overline{{D}_{b}}

-  If OOB error value per each observation is required, then calculate the prediction error for :math:`x_i`:

   .. math::
		OOB(x_i) = {(y_i-\hat{y_i})}^{2}


Batch Processing
================

Decision forest regression follows the general workflow described in :ref:`decision_forest`.

Training
********

For the description of the input and output, refer to :ref:`regression_usage_model`.

Decision forest regression training parameters are described in :ref:`df_batch`


Output
******

In addition to the output of regression described in  :ref:`regression_usage_model`,
decision forest regression calculates the result of decision forest.
For more details, refer to :ref:`df_batch`.

Prediction
**********

For the description of the input and output, refer to :ref:`regression_usage_model`.

In addition to the parameters of regression, decision forest
regression has the following parameters at the prediction stage:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Prediction Parameters for Decision Forest Regression (Batch Processing)
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
     - The computation method used by the decision forest regression. The
       only prediction method supported so far is the default dense method.

Examples
********

.. tabs::

  .. tab:: oneAPI DPC++

    Batch Processing:

    - :ref:`dpc_df_reg_hist_batch.cpp`

  .. tab:: oneAPI C++

    Batch Processing:

    - :ref:`cpp_df_reg_dense_batch.cpp`

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`df_reg_default_dense_batch.cpp <decision_forest/df_reg_default_dense_batch.cpp>`
    - :cpp_example:`df_reg_hist_dense_batch.cpp <decision_forest/df_reg_hist_dense_batch.cpp>`
    - :cpp_example:`df_reg_traverse_model.cpp <decision_forest/df_reg_traverse_model.cpp>`

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`decision_forest_regression_default_dense.py`
    - :daal4py_example:`decision_forest_regression_hist.py`
    - :daal4py_example:`decision_forest_regression_traverse.py`
