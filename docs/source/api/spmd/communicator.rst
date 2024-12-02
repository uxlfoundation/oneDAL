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

.. _api_communicator:

=============
Communicators
=============

.. _communicator_programming_interface:

---------------------
Programming interface
---------------------

All types and functions in this section are declared in the
``oneapi::dal::spmd::preview`` namespace and are available via inclusion of the
header file from specified backend.

Communicator
------------

A base implementation of the :term:`communicator` concept.
The :term:`communicator` type and all of its subtypes are :term:`reference-counted <Reference-counted object>`:

1. The instance stores a pointer to the communicator implementation that holds all
   property values and data.

2. The reference count indicates how many communicator objects refer to the same implementation.

3. The communicator increments the reference count
   for it to be equal to the number of communicator objects sharing the same implementation.

4. The communicator decrements the reference count when the
   communicator goes out of the scope. If the reference count is zero, the communicator
   frees its implementation.

USM and non-USM memory usage
----------------------------

There are two types of memory access:

- USM memory access (both USM and non-USM pointers can be used)
- Host, or non-USM, memory access (only non-USM pointers can be used)

Use one of the following tags to select a memory access type:

device_memory_access::none
   Assumes only non-USM pointers are used for a collective operation.

device_memory_access::usm
   Both USM and non-USM can be used. Pointer type is controlled by
   the use of ``sycl::queue`` object as a first parameter for collective
   operations. The use of ``sycl::queue`` object is obligatory for USM
   pointers.

Request
-------

Request is an object to control asynchronous communication.

Reducion operations
-------------------

The following reduction operations are supported:

- Max
- Min
- Sum
