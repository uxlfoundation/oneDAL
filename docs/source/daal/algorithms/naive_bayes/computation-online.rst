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

Online Processing
*****************

You can use the Naïve Bayes classifier algorithm in the online processing mode only at the training stage.

This computation mode assumes that the data arrives in blocks :math:`i = 1, 2, 3, \ldots, \text{nblocks}`.

Training
--------

Naïve Bayes classifier training in the online processing mode follows the general workflow described in :ref:`classification_usage_model`.

Naïve Bayes classifier in the online processing mode accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Training Input for Naïve Bayes Classifier (Online Processing)
   :widths: 10 60
   :header-rows: 1
   :class: longtable

   * - Input ID
     - Input
   * - ``data``
     - Pointer to the :math:`n_i \times p` numeric table that represents the current data block.
   * - ``labels``
     - Pointer to the :math:`n_i \times 1` numeric table with class labels associated with the current data block.

.. note:: These tables can be objects of any class derived from ``NumericTable``.

Naïve Bayes classifier in the online processing mode has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Training Parameters for Naïve Bayes Classifier (Online Processing)
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

For a description of the output, refer to :ref:`classification_usage_model`.