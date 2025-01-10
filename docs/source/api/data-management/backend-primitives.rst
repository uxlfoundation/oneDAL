.. Copyright contributors to the oneDAL project
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

.. _api_backend_primitives:

==================
Backend Primitives
==================

Refer to :ref:`Developer Guide: Backend Primitives <dm_backend_primitives>`.

.. _backend_primitives_programming_interface:

---------------------
Programming interface
---------------------

All types and functions in this section are declared in the
``oneapi::dal::backend::primitives`` namespace and be available via inclusion of the
``oneapi/dal/backend/primitives/ndarray.hpp`` header file.

Multidimensional data view (ndview)
-----------------------------------

An implementation of a multidimensional data container that provides a view of the homogeneous
data stored in an externally-managed memory block.

All the ``ndview`` class methods can be divided into several groups:

#. The group of ``wrap()`` methods that are used to create an ``ndview`` object from external,
   mutable or immutable memory.

#. The group of ``wrap_mutable()`` methods that are used to create a mutable ``ndview`` object from
   ``dal::array`` object.

#. The methods that are used to access the data.

#. The methods like ``t()`` and ``reshape()`` that are used to change the shape and layout of the data view.

#. The group of data slicing methods that are used to create a new ``ndview`` object that is a
   view of the original data slice along some dimension.

#. The group of data transfering methods that are used to produce a new ``ndview`` object that
   contains the data copied from the original one, but at the different memory location.

Multidimensional array (ndarray)
--------------------------------

An implementation of multidimensional data array that provides a way to store and manipulate
homogeneous data in a multidimensional structure.

All the ``ndarray`` class methods can be divided into several groups:

#. The group of ``wrap()`` and ``wrap_mutable()`` methods that are used to create an ``ndarray``
   object from external, mutable or immutable memory.

#. The group of ``wrap()`` and ``wrap_mutable()`` methods that are used to create an ``ndarray``
   that shares its data with another data object.

#. The group of methods like ``zeros()``, ``full()``, ``arange()``, etc. that are used to create an ``ndarray``
   object with the specified shape and values.

#. The methods like ``t()`` and ``reshape()`` that are used to change the shape and layout
   of the multidimensional array.

#. The group of data slicing methods that are used to create a new ``ndarray`` object that is a view
   of the original data slice along some dimension.

#. The group of methods like ``fill()``, ``assign()``, ``assign_from_host()``, etc. that are used to
   fill the array with the specified values.

.. _api_table2ndarray:

Table to ndarray conversion functions
-------------------------------------

.. onedal_func:: oneapi::dal::backend::primitives::table2ndarray

.. onedal_func:: oneapi::dal::backend::primitives::table2ndarray_1d

.. onedal_func:: oneapi::dal::backend::primitives::table2ndarray_rm

.. onedal_func:: oneapi::dal::backend::primitives::table2ndarray_cm

.. onedal_func:: oneapi::dal::backend::primitives::homogen_table2ndarray_cm

.. onedal_func:: oneapi::dal::backend::primitives::table2ndarray_variant

.. _api_ndorder:

Multidimensional array order
----------------------------

Refers to data indexing order, or how a linear sequence is translated into a multi-dimensional array.

.. onedal_enumclass:: oneapi::dal::backend::primitives::ndorder

.. _api_ndshape:

Multidimensional array shape
----------------------------

.. onedal_class:: oneapi::dal::backend::primitives::ndshape

.. toctree::

   backend/ndview.rst
   backend/ndarray.rst
