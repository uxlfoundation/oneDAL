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

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for DBSCAN (Distributed Processing, Step 5)
   :widths: 10 10 60
   :header-rows: 1
   :class: longtable

   * - Parameter
     - Default Valude
     - Description
   * - ``algorithmFPType``
     - ``float``
     - The floating-point type that the algorithm uses for intermediate computations. Can be ``float`` or ``double``.
   * - ``method``
     - ``defaultDense``
     - Available methods for computation of DBSCAN algorithm:

       - ``defaultDense`` – uses brute-force for neighborhood computation

   * - ``blockIndex``
     - Not applicable
     - Unique identifier of block initially passed for computation on the local node.
   * - ``nBlocks``
     - Not applicable
     - The number of blocks initially passed for computation on all nodes.
   * - ``epsilon``
     - Not applicable
     - The maximum distance between observations lying in the same neighborhood.

In this step, the DBSCAN algorithm accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for DBSCAN (Distributed Processing, Step 5)
   :widths: 10 60
   :header-rows: 1
   :class: longtable

   * - Input ID
     - Input
   * - ``partialData``
     - Pointer to the collection of numeric tables with :math:`p` columns and arbitrary number of rows, containing observations to be clustered.

       .. include:: ./../../includes/input_data_collection.rst

   * - ``step5PartialBoundingBoxes``
     - Pointer to the collection of :math:`2 \times p` numeric table containing bounding boxes computed on :ref:`step 2 <dbscan_step_2>` and collected from all nodes.
       Numeric tables in collection should be ordered by the identifiers of initial block of nodes.

       .. include:: ./../../includes/input_data_collection_with_exceptions.rst

Algorithm Output
++++++++++++++++

In this step, the DBSCAN algorithms calculates the partial results described below.
Pass the ``Partial Result ID`` as a parameter to the methods that access the partial result of your algorithm.
For more details, :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Partial Results for DBSCAN (Distributed Processing, Step 5)
   :widths: 10 60
   :header-rows: 1
   :class: longtable

   * - Partial Result ID
     - Result
   * - ``partitionedHaloData``
     - Pointer to the collection of ``nBlocks`` numeric tables with :math:`p` columns and arbitrary number of rows containing observations
       from current node that should be used as halo observations on each node.

       Numeric tables in the collection are ordered by the identifiers of initial block of nodes.

   * - ``partitionedHaloDataIndices``
     - Pointer to the collection of ``nBlocks`` numeric tables with :math:`1` column and arbitrary number of rows containing indices of observations
       from current node that should be used as halo observations on each node.

       Numeric tables in the collection are ordered by the identifiers of initial block of nodes.

.. include:: ./../../includes/default_result_data_collection.rst
