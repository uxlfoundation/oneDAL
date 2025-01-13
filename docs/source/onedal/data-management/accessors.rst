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

.. highlight:: cpp
.. default-domain:: cpp

.. _dm_accessors:

=========
Accessors
=========

This section defines :txtref:`requirements <accessor_reqs>` to an
:txtref:`accessor` implementation and introduces several
:txtref:`accessor_types`.

.. _accessor_reqs:

------------
Requirements
------------

Each accessor implementation:

#. Defines a single :term:`format of the data <Data format>` for the
   access. Every accessor type returns and use only one data format.

#. Provides read-only access to the data in the :txtref:`table` types.

#. Provides the :code:`pull()` method for obtaining the values from the table.

#. Is lightweight. Its constructors do not have computationally intensive
   operations such data copy, reading, or conversion. These operations are
   performed by method :code:`pull()`.

#. The :code:`pull()` method avoids data copy and conversion when it is
   possible to return the pointer to the memory block in the table. This is
   applicable for cases such as when the :capterm:`data format` and
   :capterm:`data types <data type>` of the data within the table are the same as the
   :capterm:`data format` and :capterm:`data type` for the access.


.. _accessor_types:

--------------
Accessor Types
--------------

|short_name| defines a set of accessor classes. Each class supports one
specific way of obtaining data from the :txtref:`table`.

All accessor classes in |short_name| are listed below:

.. tabularcolumns::  |\Y{0.25}|\Y{0.5}|\Y{0.25}|

.. list-table:: Accessor Types
   :header-rows: 1
   :widths: 25 50 25
   :class: longtable

   * - Accessor type
     - Description
     - List of supported types
   * - :txtref:`row_accessor`
     - Provides access to the range of rows as one :term:`contiguous
       <Contiguous data>` :term:`homogeneous <Homogeneous data>` block of memory.
     - :txtref:`homogen_table`
   * - :txtref:`column_accessor`
     - Provides access to the range of values within a single column as one
       :term:`contiguous <Contiguous data>` :term:`homogeneous <Homogeneous
       data>` block of memory.
     - :txtref:`homogen_table`

-------
Details
-------

.. toctree::

   accessor/column.rst
   accessor/row.rst
