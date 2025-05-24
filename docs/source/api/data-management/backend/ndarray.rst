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

.. _api_ndarray:

======================
Multidimensional array
======================

The ``ndarray`` class provides a way to store and manipulate homogeneous data
in a multidimensional structure.
The pointer to the data within the ``ndarray`` object is :term:`reference-counted <Reference-counted object>`:.

---------------------
Programming interface
---------------------

All types and functions in this section are declared in the
``oneapi::dal::backend::primitives`` namespace and be available via inclusion of the
``oneapi/dal/table/ndarray.hpp`` header file.

.. onedal_class:: oneapi::dal::backend::primitives::ndarray
