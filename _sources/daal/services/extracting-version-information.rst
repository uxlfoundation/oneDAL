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

Extracting Version Information
==============================

The Services component provides methods that enable you to extract
information about the version of |short_name|. You can get the
following information about the installed version of the library from
the fields of the LibraryVersionInfo structure:


+-----------------------------------+-----------------------------------+
| Field Name                        | Description                       |
+===================================+===================================+
| majorVersion                      | Major version of the library      |
+-----------------------------------+-----------------------------------+
| minorVersion                      | Minor version of the library      |
+-----------------------------------+-----------------------------------+
| updateVersion                     | Update version of the library     |
+-----------------------------------+-----------------------------------+
| productStatus                     | Status of the library: alpha,     |
|                                   | beta, or product                  |
+-----------------------------------+-----------------------------------+
| build                             | Build number                      |
+-----------------------------------+-----------------------------------+
| name                              | Library name                      |
+-----------------------------------+-----------------------------------+
| processor                         | Processor optimization            |
+-----------------------------------+-----------------------------------+


.. include:: ../../opt-notice.rst

Examples
++++++++

C++: :cpp_example:`services/library_version_info.cpp`

.. Python*: library_version_info.py

