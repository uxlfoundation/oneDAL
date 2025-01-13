.. Copyright 2023 Intel Corporation
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

.. _api_csr_accessor:

=====================================
Compressed Sparse Rows (CSR) Accessor
=====================================

The ``csr_accessor`` class provides read-only access to the rows of the
:txtref:`csr_table` as data arrays in :capterm:`CSR <CSR data>` storage format.

-------------
Usage Example
-------------

.. include:: ../../../includes/data-management/csr-accessor-usage-example.rst

---------------------
Programming Interface
---------------------

All types and functions in this section are declared in the
``oneapi::dal`` namespace and available via the inclusion of the
``oneapi/dal/table/csr_accessor.hpp`` header file.

.. onedal_class:: oneapi::dal::csr_accessor
