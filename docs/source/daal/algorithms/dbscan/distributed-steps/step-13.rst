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

.. list-table:: Algorithm Input for DBSCAN (Distributed Processing, Step 13)
   :widths: 10 60
   :header-rows: 1

   * - Input ID
     - Input
   * - ``partialAssignmentQueries``
     - Pointer to the collection of numeric tables with :math:`2` columns and arbitrary number of rows
       containing clusters assigning queries that should be processed on the local node collected from all nodes.

       .. include:: ./../../includes/input_data_collection_with_exceptions.rst

Algorithm Output
++++++++++++++++

In this step, the DBSCAN algorithms calculates the results and partial results described below.
Pass the ``Result ID`` as a parameter to the methods that access the result and partial result of your algorithm.
For more details, :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for DBSCAN (Distributed Processing, Step 13)
   :widths: 10 60
   :header-rows: 1

   * - Result ID
     - Result
   * - ``step13Assignments``
     - Pointer to the :math:`n \times 1` numeric table with assignments of cluster indices to observations
       processed on :ref:`step 1 <dbscan_step_1>` on the local node.
       :term:`Noise observations <noise observation>` have the assignment equal to :math:`-1`.

       .. include:: ./../../includes/default_result_numeric_table.rst

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Partial Results for DBSCAN (Distributed Processing, Step 13)
   :widths: 10 60
   :header-rows: 1

   * - Partial Result ID
     - Result
   * - ``step13AssignmentsQueries``
     - Pointer to the numeric table with :math:`2` columns and arbitrary number of rows
       containing clusters assigning queries that should be processed on the local node.

       .. include:: ./../../includes/default_result_numeric_table.rst
