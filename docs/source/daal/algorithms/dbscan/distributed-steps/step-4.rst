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

.. include:: distributed-steps/includes/parameters_blocks_left_right.rst

In this step, the DBSCAN algorithm accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for DBSCAN (Distributed Processing, Step 4)
   :widths: 10 60
   :header-rows: 1
   :class: longtable

   * - Input ID
     - Input
   * - ``partialData``
     - Pointer to the collection of numeric tables with :math:`p` columns and arbitrary number of rows, containing observations to be clustered.

       .. include:: ./../../includes/input_data_collection.rst

   * - ``step4PartialOrders``
     - Pointer to the collection of numeric table with :math:`2` columns and arbitrary number of rows containing information about observations:
       identifier of initial block and index in initial block.

       .. include:: ./../../includes/input_data_collection_with_exceptions.rst

   * - ``step4PartialSplits``
     - Pointer to the collection of the :math:`1 \times 2` numeric table containing information about split computed on
       :ref:`step 3 <dbscan_step_3>` and collected from all nodes
       participating in current iteration of geometric repartitioning process.

       .. include:: ./../../includes/input_data_collection_with_exceptions.rst

Algorithm Output
++++++++++++++++

In this step, the DBSCAN algorithms calculates the partial results described below.
Pass the ``Partial Result ID`` as a parameter to the methods that access the partial result of your algorithm.
For more details, :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Partial Results for DBSCAN (Distributed Processing, Step 4)
   :widths: 10 60
   :header-rows: 1

   * - Partial Result ID
     - Result
   * - ``partitionedData``
     - Pointer to the collection of (``leftBlocks`` + ``rightBlocks``) numeric tables with :math:`p` columns and arbitrary number of rows
       containing observations for processing on nodes participating in current iteration of geometric repartitioning.

       - First ``leftBlocks`` numeric tables in collection have the value of selected split feature smaller than selected split value.
       - Next ``rightBlocks`` numeric tables in collection have the value of selected split feature larger than selected split value.

       .. include:: ./../../includes/default_result_numeric_table.rst
