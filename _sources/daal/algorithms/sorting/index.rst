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

Sorting
=======

In |short_name| sorting is an algorithm to sort the observations by each feature (column) in the ascending order.

The result of the sorting algorithm applied to the matrix
:math:`X = (x_{ij})_{n \times p}` is the matrix :math:`Y = (y_{ij})_{n \times p}`
where the :math:`j`-th column :math:`(Y)_j = ( y_{ij} )`, :math:`i = 1, \ldots, n`,
is the column :math:`(X)_j = ( x_{ij} )`, :math:`i = 1, \ldots, n`, sorted in the ascending order.

Batch Processing
****************

Algorithm Input
---------------

The sorting algorithm accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Sorting (Batch Processing)
   :widths: 10 60
   :header-rows: 1

   * - Input ID
     - Input
   * - ``data``
     - Pointer to the :math:`n \times p` numeric table that contains the input data set.

       This table can be an object of any class derived from ``NumericTable`` except
       ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.

Algorithm Parameters
--------------------

The sorting algorithm has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for Sorting (Batch Processing)
   :header-rows: 1
   :align: left
   :widths: 10 10 60
   :class: longtable

   * - Parameter
     - Default Value
     - Description
   * - ``algorithmFPType``
     - ``float``
     - The floating-point type that the algorithm uses for intermediate computations. Can be ``float`` or ``double``.
   * - ``method``
     - ``defaultDense``
     - The radix method for sorting a data set, the only method supported by the algorithm.


Algorithm Output
----------------

The sorting algorithm function calculates the result described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Sorting (Batch Processing)
   :widths: 10 60
   :header-rows: 1

   * - Result ID
     - Result
   * - ``sortedData``
     - Pointer to the :math:`n \times p` numeric table that stores the results of sorting.

.. note::

    If the number of feature vectors is greater than or equal to :math:`2^{31}`,
    the library uses the quick sort method instead of radix sort.

Examples
********

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`sorting_dense_batch.cpp <sorting/sorting_dense_batch.cpp>`


  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`sorting.py`
