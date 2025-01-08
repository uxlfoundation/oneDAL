.. Copyright 2021 Intel Corporation
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

.. _api_array:

=====
Array
=====

Refer to :ref:`Developer Guide: Array <dm_array>`.

---------------------
Programming interface
---------------------

All types and functions in this section are declared in the
``oneapi::dal`` namespace and be available via inclusion of the
``oneapi/dal/array.hpp`` header file.

All the ``array`` class methods can be divided into several groups:

1. Constructors that are used to create an array from external, mutable or
   immutable memory.

2. Constructors and assignment operators that are used to create an array that shares its data
   with another one.

3. The group of ``reset()`` methods that are used to re-assign an array to another external
   memory block.

4. The group of ``reset()`` methods that are used to re-assign an array to an internally
   allocated memory block.

5. The methods that are used to access the data.

6. Static methods that provide simplified ways to create an array either from external
   memory or by allocating it within a new object.

.. onedal_class:: oneapi::dal::array

-------------
Usage Example
-------------

The following listing provides a brief introduction to the array API and an example of basic
usage scenario:

.. include:: ../../includes/data-management/array-usage-example.rst
