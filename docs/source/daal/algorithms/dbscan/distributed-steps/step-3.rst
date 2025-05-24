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

.. list-table:: Algorithm Input for DBSCAN (Distributed Processing, Step 3)
   :widths: 10 60
   :header-rows: 1
   :class: longtable

   * - Input ID
     - Input
   * - ``partialData``
     - Pointer to the collection of numeric tables with :math:`p` columns and arbitrary number of rows, containing observations to be clustered.

       .. include:: ./../../includes/input_data_collection.rst

   * - ``step3PartialBoundingBoxes``
     - Pointer to the collection of the :math:`2 \times p` numeric tables containing bounding boxes computed on :ref:`step 2 <dbscan_step_2>` and collected from all nodes
       participating in current iteration of geometric repartitioning process.

       .. note::

         The numeric tables in collection can be an object of any class
         derived from ``NumericTable`` except for ``PackedTriangularMatrix``, ``PackedSymmetricMatrix``, and ``CSRNumericTable``.

Algorithm Output
++++++++++++++++

In this step, the DBSCAN algorithms calculates the partial results described below.
Pass the ``Partial Result ID`` as a parameter to the methods that access the partial result of your algorithm.
For more details, :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Partial Results for DBSCAN (Distributed Processing, Step 3)
   :widths: 10 60
   :header-rows: 1

   * - Partial Result ID
     - Result
   * - ``split``
     - Pointer to the :math:`1 \times 2` numeric table containing information about split for current iteration of geometric repartitioning.

       .. include:: ./../../includes/default_result_numeric_table.rst
