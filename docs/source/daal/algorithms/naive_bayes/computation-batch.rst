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

Batch Processing
****************

Naïve Bayes classifier in the batch processing mode follows the general workflow described
in :ref:`classification_usage_model`.

Training
--------

At the training stage, Naïve Bayes classifier has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Training Parameters for Naïve Bayes Classifier (Batch Processing)
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
     - Available computation methods for the Naïve Bayes classifier:

       - ``defaultDense`` - default performance-oriented method
       - ``fastCSR`` - performance-oriented method for CSR numeric tables

   * - ``nClasses``
     - Not applicable
     - The number of classes. A required parameter.
   * - ``priorClassEstimates``
     - :math:`1/\text{nClasses}`
     - Vector of size ``nClasses`` that contains prior class estimates. The default value applies to each vector element.
   * - ``alpha``
     - :math:`1`
     - Vector of size :math:`p` that contains the imagined occurrences of features. The default value applies to each vector element.


Prediction
----------

At the prediction stage, Naïve Bayes classifier has the following parameters:

.. tabularcolumns::  |\Y{0.2}|\Y{0.2}|\Y{0.6}|

.. list-table:: Prediction Parameters for Naïve Bayes Classifier (Batch Processing)
   :widths: 10 20 30
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
     - Performance-oriented computation method, the only method supported by the algorithm.
   * - ``nClasses``
     - Not applicable
     - The number of classes. A required parameter.
