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

.. _csv-data-source:

---------------
CSV data source
---------------
Class ``csv::data_source`` is an API for accessing the data source represented
as a :term:`csv file <CSV file>`. CSV data source is used with
:cpp:expr:`read` operation to extract data in text format from the given input file,
process it using provided parameters (such as delimiter and read options),
transform it into numerical representation, and store it as an in-memory
:txtref:`dataset` of a chosen type.

Supported type of in-memory object for :cpp:expr:`read` operation with CSV data
source is :cpp:expr:`oneapi::dal::table`.

CSV data source requires input file name to be set in the constructor, while the
other parameters of the constructor such as delimiter and read options rely on
default values.

Usage Example
-------------

.. include:: ../../../includes/data-management/csv-data-source-usage-example.rst

Programming Interface
---------------------

Refer to :ref:`API Reference: CSV data source <api_csv-data-source>`.
