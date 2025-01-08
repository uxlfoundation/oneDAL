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

.. _gbt_regression:

Regression Gradient Boosted Trees
=================================

Gradient boosted trees regression is the special case of gradient
boosted trees. For more details, see :ref:`gb_trees`.

Details
*******

Given n feature vectors :math:`X = \{x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np}) \}` of :math:`n`
:math:`p`-dimensional feature vectors and a vector of dependent variables :math:`y = (y_1, \ldots, y_n)`, the problem is to build a gradient
boosted trees regression model that minimizes the loss function based
on the predicted and true value.

Training Stage
--------------

Gradient boosted trees regression follows the algorithmic
framework of gradient boosted trees training with following loss

.. math::
	L(y, f)=\frac{1}{2}({y-f(x))}^{2}

Prediction Stage
----------------

Given the gradient boosted trees regression model and vectors :math:`(x_1, \ldots, x_r)`, the problem is to calculate responses
for those vectors. To solve the problem for each given feature
vector :math:`x_i`, the algorithm finds the leaf node in a tree in
the ensemble, and the leaf node gives the tree response. The
algorithm result is a sum of responses of all the trees.

Usage of Training Alternative
*****************************

To build a Gradient Boosted Trees Regression model using methods of the Model Builder class
of Gradient Boosted Tree Regression, complete the following steps:

- Create a Gradient Boosted Tree Regression model builder using a constructor with the required number of classes and trees.
- Create a decision tree and add nodes to it:

  - Use the ``createTree`` method with the required number of nodes in a tree and a label of the class for which the tree is created.
  - Use the ``addSplitNode`` and ``addLeafNode`` methods to add split and leaf nodes to the created tree.
    See the note below describing the decision tree structure.
  - After you add all nodes to the current tree, proceed to creating the next one in the same way.

- Use the ``getModel`` method to get the trained Gradient Boosted Trees Regression model after all trees have been created.

.. note::

  Each tree consists of internal nodes (called non-leaf or split nodes) and external nodes (leaf nodes).
  Each split node denotes a feature test that is a Boolean expression, for example,
  f < ``featureValue`` or f = ``featureValue``, where f is a feature and ``featureValue`` is a constant.
  The test type depends on the feature type: continuous, categorical, or ordinal.
  For more information on the test types, see :ref:`decision_tree`.

  The inducted decision tree is a binary tree, meaning that each non-leaf node has exactly two branches: true and false.
  Each split node contains ``featureIndex``, the index of the feature used for the feature test in this node, and ``featureValue``,
  the constant for the Boolean expression in the test. Each leaf node contains a ``classLabel``, the predicted class for this leaf.
  For more information on decision trees, see :ref:`decision_tree`.

  Add nodes to the created tree in accordance with the pre-calculated structure of the tree.
  Check that the leaf nodes do not have children nodes and that the splits have exactly two children.

Examples
--------

.. tabs::

  .. tab:: C++ (CPU)

    - :cpp_example:`gbt_reg_traversed_model_builder.cpp <gradient_boosted_trees/gbt_reg_traversed_model_builder.cpp>`

  .. tab:: Python*

    - :daal4py_example:`gradient_boosted_regression_traverse.py`


Batch Processing
****************

Gradient boosted trees regression follows the general workflow
described in :ref:`gb_trees` and :ref:`regression_usage_model`.

Training
--------

In addition to parameters of the gradient boosted trees described in :ref:`gb_trees_batch`,
the gradient boosted trees regression training algorithm has the
following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Training Parameters for Gradient Boosted Trees Regression (Batch Processing)
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
     - The computation method used by the gradient boosted trees regression. The only training method supported so far is the default dense method.
   * - ``loss``
     - ``squared``
     - Loss function type.

Prediction
----------

In addition to the common regression parameters, the gradient boosted
trees regression has the following parameters at the prediction stage:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Prediction Parameters for Gradient Boosted Trees Regression (Batch Processing)
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
     - The computation method used by the gradient boosted trees regression. The only training method supported so far is the default dense method.
   * - ``numIterations``
     - :math:`0`
     - An integer parameter that indicates how many trained iterations of the
       model should be used in prediction. The default value :math:`0` denotes no
       limit. All the trained trees should be used.

Examples
********

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`gbt_reg_dense_batch.cpp <gradient_boosted_trees/gbt_reg_dense_batch.cpp>`

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`gradient_boosted_regression.py`
    - :daal4py_example:`gradient_boosted_regression_traverse.py`
