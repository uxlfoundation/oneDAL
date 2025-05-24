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

In this step, the DBSCAN algorithm has the following parameters:

.. include:: distributed-steps/includes/parameters.rst

In this step, the DBSCAN algorithm accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for DBSCAN (Distributed Processing, Step 9)
   :widths: 10 60
   :header-rows: 1

   * - Input ID
     - Input
   * - ``partialNClusters``
     - Pointer to the collection of :math:`1 \times 1` numeric table containing the number of clusters found on each node.

       .. include:: ./../../includes/input_data_collection_with_exceptions.rst

Algorithm Output
++++++++++++++++

In this step, the DBSCAN algorithms calculates the results and partial results described below.
Pass the ``Result ID`` as a parameter to the methods that access the result and partial result of your algorithm.
For more details, :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for DBSCAN (Distributed Processing, Step 9)
   :widths: 10 60
   :header-rows: 1

   * - Result ID
     - Result
   * - ``step9NClusters``
     - Pointer to :math:`1 \times 1` numeric table containing the number of clusters found on all nodes.

       .. include:: ./../../includes/default_result_numeric_table.rst

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Partial Results for DBSCAN (Distributed Processing, Step 9)
   :widths: 10 60
   :header-rows: 1

   * - Partial Result ID
     - Result
   * - ``clusterOffsets``
     - Pointer to the collection of :math:`1 \times 1` numeric tables containing offsets for cluster numeration
       for each node. Numeric tables with offsets are given in the same order as in the collection for ``partialNClusters`` ``Input ID``.

       .. include:: ./../../includes/default_result_data_collection.rst
