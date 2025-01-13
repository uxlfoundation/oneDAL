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

Merged Numeric Table
====================

.. toctree::
   :maxdepth: 1
   :hidden:

|product| offers the ``MergedNumericTable`` class for tables that
provides access to data sets comprising several logical components,
such as a set of feature vectors and corresponding labels. This type
of tables enables you to read those data components from one data
source. This special type of numeric tables can hold several numeric
tables of any type but ``CSRNumericTable``. In a merged numeric table,
arrays are joined by columns and therefore can have different numbers
of columns. In the case of different numbers of rows in input
matrices, the number of rows in a merged table equals :math:`min(r_1, r_2, \ldots, r_m)`,
where :math:`r_i` is the number of
rows in the i-th matrix, :math:`i = 1, 2, 3, \ldots, m`.

.. figure:: ./images/merged-numeric-table.png
  :width: 400
  :alt:

  Merged Numeric Table

Examples
********

C++: :cpp_example:`datasource/datastructures_merged.cpp`

.. Python*:datastructures_merged.py
