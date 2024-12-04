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

.. _svm_multi_class:

Multi-class Classifier
======================

.. toctree::
   :glob:
   :maxdepth: 4

While some classification algorithms naturally permit the use of more
than two classes, some algorithms, such as Support Vector Machines
(SVM), are by nature solving a two-class problem only. These
two-class (or binary) classifiers can be turned into multi-class
classifiers by using different strategies, such as One-Against-Rest
or One-Against-One.

|product| implements a Multi-Class Classifier using the
One-Against-One strategy.

Multi-class classifiers, such as SVM, are based on two-class
classifiers, which are integral components of the models trained with
the corresponding multi-class classifier algorithms.


Details
*******

Given :math:`n` feature vectors :math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})`
of size :math:`p`, the number of classes :math:`K`, and a vector of class labels
:math:`y = (y_1, \ldots, y_n)`, where :math:`y_i \in \{0, 1, \ldots, K-1\}`,
the problem is to build a multi-class classifier using a two-class
(binary) classifier, such as a two-class SVM.

Training Stage
--------------

The model is trained with the One-Against-One method that uses the
binary classification described in
[Hsu02]_ as follows:
For each pair of classes :math:`(i, j)`, train a binary classifier, such
as SVM. The total number of such binary classifiers is :math:`\frac{K(K-1)}{2}`.

Prediction Stage
----------------

Given a new feature vector :math:`x_i`, the classifier determines
the class to which the vector belongs.

|product| provides two methods for class label prediction:

-  Wu method. According to the algorithm 2 for computation of the
   class probabilities described in [Wu04]_.
   The library returns the index of the class with the largest
   probability.
-  Vote-based method. If the binary classifier predicts the
   feature vector to be in :math:`i`-th class, the number of votes for the
   class i is increased by one, otherwise the vote is given to the
   j-th class. If two classes have equal numbers of votes, the
   class with the smallest index is selected.

Usage of Training Alternative
*****************************

To build a Multi-class Classifier model using methods of the Model Builder class of Multi-class Classifier,
complete the following steps:

- Create a Multi-class Classifier model builder using a constructor with the required number of features and classes.
- Use the ``setTwoClassClassifierModel`` method for each pair of classes to add the pre-trained two-class classifiers to the model.
  In the parameters to the method specify the classes' indices and the pointer to the pre-trained two-class classifier for this pair of classes.
  You need to do this for each pair of classes, because the One-Against-One strategy is used.
- Use the ``getModel`` method to get the trained Multi-class Classifier model.
- Use the ``getStatus`` method to check the status of the model building process.
  If ``DAAL_NOTHROW_EXCEPTIONS`` macros is defined, the status report contains the list of errors
  that describe the problems API encountered (in case of API runtime failure).

Examples
--------

.. tabs::

  .. tab:: oneAPI C++

    Batch Processing

    - :ref:`cpp_svm_two_class_thunder_dense_batch.cpp`

  .. tab:: C++ (CPU)

    Batch Processing

    - :cpp_example:`svm_multi_class_model_builder.cpp <svm/svm_multi_class_model_builder.cpp>`

  .. tab:: Python*

    :daal4py_example:`svm_multiclass.py`

Batch Processing
****************

Multi-class classifier follows the general workflow described in :ref:`classification_usage_model`.

Training
--------

At the training stage, a multi-class classifier has the following
parameters:

.. tabularcolumns::  |\Y{0.25}|\Y{0.3}|\Y{0.45}|

.. list-table:: Training Parameters for  Multi-class Classifier (Batch Processing)
   :widths: 10 20 30
   :header-rows: 1
   :align: left
   :class: longtable

   * - Parameter
     - Default Value
     - Description
   * - ``algorithmFPType``
     - ``float``
     - The floating-point type that the algorithm uses for intermediate
       computations. Can be ``float`` or ``double``.
   * - ``method``
     - ``defaultDense``
     - The computation method used by the multi-class classifier. The only
       training method supported so far is One-Against-One.
   * - ``training``
     - Pointer to an object of the SVM training class
     - Pointer to the training algorithm of the two-class classifier. By default, the SVM two-class classifier is used.
   * - ``nClasses``
     - Not applicable
     - The number of classes. A required parameter.

Prediction
----------

At the prediction stage, a multi-class classifier has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.15}|\Y{0.55}|

.. list-table:: Prediction Parameters for  Multi-class Classifier (Batch Processing)
   :widths: 10 10 10 30
   :header-rows: 1
   :align: left
   :class: longtable

   * - Parameter
     - Method
     - Default Value
     - Description
   * - ``algorithmFPType``
     - ``defaultDense`` or ``voteBased``
     - ``float``
     - The floating-point type that the algorithm uses for intermediate
       computations. Can be ``float`` or ``double``.
   * - ``pmethod``
     - Not applicable
     - ``defaultDense``
     - Available methods for multi-class classifier prediction stage:

       -  ``defaultDense`` - the method described in [Wu04]_
       -  ``voteBased`` - the method based on the votes obtained from two-class classifiers.

   * - ``tmethod``
     - ``defaultDense`` or ``voteBased``
     - `training::oneAgainstOne`
     - The computation method that was used to train the multi-class classifier model.
   * - ``prediction``
     - ``defaultDense`` or ``voteBased``
     - Pointer to an object of the SVM prediction class
     - Pointer to the prediction algorithm of the two-class classifier. By default, the SVM two-class classifier is used.
   * - ``nClasses``
     - ``defaultDense`` or ``voteBased``
     - Not applicable
     - The number of classes. A required parameter.
   * - ``maxIterations``
     - ``defaultDense``
     - :math:`100`
     - The maximal number of iterations for the algorithm.
   * - ``accuracyThreshold``
     - ``defaultDense``
     - 1.0e-12
     - The prediction accuracy.
   * - ``resultsToEvaluate``
     - ``voteBased``
     - ``computeClassLabels``
     - The 64-bit integer flag that specifies which extra characteristics of the decision function to compute.

       Provide one of the following values to request a single characteristic
       or use bitwise OR to request a combination of the characteristics:

       - ``computeClassLabels`` for `prediction`
       - ``computeDecisionFunction`` for `decisionFunction`

Output
------

In addition to classifier output, multiclass classifier calculates the result described below.
Pass the ``Result ID`` as a parameter to the methods that access the result of your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Output for  Multi-class Classifier (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :align: left

   * - Result ID
     - Result
   * - ``decisionFunction``
     - A numeric table of size :math:`n \times \frac{K(K-1)}{2}` containing the results of the decision function
       computed for all binary models when the ``computeDecisionFunction`` option is enabled.

.. note::

  If `resultsToEvaluate` does not contain `computeDecisionFunction`, the result of `decisionFunction` table is `NULL`.

  By default, each numeric table of this result is an object of the ``HomogenNumericTable`` class,
  but you can define the result as an object of any class derived from ``NumericTable``
  except for ``PackedSymmetricMatrix`` and ``PackedTriangularMatrix``.


Examples
--------

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    -  :cpp_example:`svm_multi_class_boser_csr_batch.cpp <svm/svm_multi_class_boser_csr_batch.cpp>`
    -  :cpp_example:`svm_multi_class_boser_dense_batch.cpp <svm/svm_multi_class_boser_dense_batch.cpp>`
    -  :cpp_example:`svm_multi_class_thunder_csr_batch.cpp <svm/svm_multi_class_thunder_csr_batch.cpp>`
    -  :cpp_example:`svm_multi_class_thunder_dense_batch.cpp <svm/svm_multi_class_thunder_dense_batch.cpp>`

  .. tab:: Python*

    Batch Processing:

    -  :daal4py_example:`svm_multiclass.py`
