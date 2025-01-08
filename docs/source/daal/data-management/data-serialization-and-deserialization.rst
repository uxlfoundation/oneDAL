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

Data Serialization and Deserialization
======================================

|short_name| provides interfaces for serialization and deserialization
of data objects, which are an essential technique for data exchange
between devices and for implementing data recovery mechanisms on a
device failure.

The InputDataArchive class provides interfaces for creation of a
serialized object archive. The OutputDataArchive class provides
interfaces for deserialization of an object from the archive.

A general structure of an archive is as follows:

.. figure:: ./images/data-archive-structure.png
  :width: 400
  :alt: The first segment contains the archive header,
        the last segment contains the archive footer, and all
        other segments contain a segment header and a segment footer.

  Data Archive Structure

Headers and footers contain information required to reconstruct the
archived object.

All serializable objects, such as numeric tables, a data dictionary,
and models, have serialization and deserialization methods. These
methods take input archive and output archive, respectively, as
method parameters.

Examples
********

C++: :cpp_example:`serialization/serialization.cpp`

