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

.. _api_tables:

======
Tables
======

Refer to :ref:`Developer Guide: Tables <dm_tables>`.

.. _table_programming_interface:

---------------------
Programming interface
---------------------

All types and functions in this section are declared in the
``oneapi::dal`` namespace and be available via inclusion of the
``oneapi/dal/table/common.hpp`` header file.

Table
-----

A base implementation of the :txtref:`table` concept.
The ``table`` type and all of its subtypes are :term:`reference-counted <Reference-counted object>`:

1. The instance stores a pointer to table implementation that holds all
   property values and data

2. The reference count indicating how many table objects refer to the same implementation.

3. The table increments the reference count
   for it to be equal to the number of table objects sharing the same implementation.

4. The table decrements the reference count when the
   table goes out of the scope. If the reference count is zero, the table
   frees its implementation.


.. onedal_class:: oneapi::dal::table

.. _metadata_programming_interface:

Table metadata
--------------

An implementation of the :txtref:`table_metadata` concept. Holds additional
information about data within the table. The objects of ``table_metadata``
are :term:`reference-counted <Reference-counted object>`.

.. onedal_class:: oneapi::dal::table_metadata

.. _api_tables_data_layout:

Data layout
-----------

An implementation of the :capterm:`data layout` concept.

::

   enum class data_layout { unknown, row_major, column_major };

.. .. namespace:: oneapi::dal
.. .. enum-class:: data_layout

``data_layout::unknown``
   Represents the :capterm:`data layout` that is undefined or unknown at this moment.

``data_layout::row_major``
   The data block elements are stored in raw-major layout.

``data_layout::column_major``
   The data block elements are stored in column_major layout.

.. _api_tables_feature_type:

Feature type
------------

An implementation of the logical data types.

::

   enum class feature_type { nominal, ordinal, interval, ratio };

.. .. namespace:: oneapi::dal
.. .. enum-class:: feature_type

``feature_type::nominal``
   Represents the type of :capterm:`Nominal feature`.

``feature_type::ordinal``
   Represents the type of :capterm:`Ordinal feature`.

``feature_type::interval``
   Represents the type of :capterm:`Interval feature`.

``feature_type::ratio``
   Represents the type of :capterm:`Ratio feature`.


.. _api_tables_sparse_indexing:

Sparse Indexing
---------------

An implementation of the sparse indexing formats.

::

   enum class sparse_indexing { zero_based, one_based };

.. .. namespace:: oneapi::dal
.. .. enum-class:: sparse_indexing

``sparse_indexing::zero_based``
   The indices of the sparse table are stored in zero-based format.

``sparse_indexing::one_based``
   The indices of the sparse table are stored in one-based format.

.. toctree::

   table/homogen.rst
   table/csr.rst
