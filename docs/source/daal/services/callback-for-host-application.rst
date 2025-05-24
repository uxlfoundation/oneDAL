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

Providing a Callback for the Host Application
=============================================


|short_name| provides a possibility for the host application to
register a callback interface to be called by the library, e.g. for
the purposes of computation interruption. It is done by means of an
abstract interface for the host application of the library
HostAppIface. In order to use it, the application should define an
instance of the class derived from the abstract interface and set its
pointer to an instance of Algorithm class.

Following methods of the Algorithm class are used:

.. tabularcolumns::  |\Y{0.3}|\Y{0.7}|

.. list-table:: Algorithm class methods
   :widths: 20 60
   :header-rows: 1
   :align: left
   :class: longtable

   * - Name
     - Description
   * - ``setHostApp(const services::HostAppIfacePtr& pHost)``
     - Set pHost as the callback interface
   * - ``hostApp()``
     - Get current value of the callback interface set on the Algorithm


HostAppIface class includes following methods:

.. tabularcolumns::  |\Y{0.3}|\Y{0.7}|

.. list-table:: HostAppIface class Methods
   :widths: 20 60
   :header-rows: 1
   :align: left

   * - Name
     - Description
   * - ``isCancelled()``
     -
        Enables computation cancelling. The method is called by the
        owning algorithm when computation is in progress. If the method
        returns true then computation stops and returns
        ErrorUserCancelled status. Since the method can be called from
        parallel threads when running with |short_name| threaded version, it is
        application responsibility to make its implementation thread-safe. It is not
        recommended for this method to throw exceptions.


Currently HostAppIface is supported in C++ only, cancelling is
available with limited number of algorithms as follows: decision
forest, gradient boosted trees.
