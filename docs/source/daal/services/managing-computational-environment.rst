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

Managing the Computational Environment
======================================

|short_name| provides the Environment class to manage settings of the
computational environment in which the application that uses the
library runs. The methods of this class enable you to specify the
number of threads for the application to use or to check the type of
the processor running the application. The Environment class is a
singleton, which ensures that only one instance of the class is
available in the |short_name| based application. To access the instance
of the class, call the ``getInstance()`` method which returns a pointer
to the instance. Once you get the instance of the class, you can use
it for multiple purposes:

-  Detect the processor type. To do this, call the ``getCpuId()`` method.

-  Restrict dispatching to the required code path.
   To do this, call the ``setCpuId()`` method.

-  Detect and modify the number of threads used by the |short_name|
   based application.
   To do this, call the ``getNumberOfThreads()`` or ``setNumberOfThreads()`` method, respectively.

-  Specify the single-threaded of multi-threaded mode for |short_name| on Windows.
   To do this, call to the ``setDynamicLibraryThreadingTypeOnWindows()`` method.

-  Enable thread pinning.
   To do this, call the ``enableThreadPinning()`` method. This method
   performs binding of the threads that are used to parallelize
   algorithms of the library to physical processing units for
   possible performance improvement. Improper use of the method can
   result in degradation of the application performance depending on
   the system (machine) topology, application, and operating system.
   By default, the method is disabled.


.. include:: ../../opt-notice.rst

Examples
++++++++

C++: :cpp_example:`set_number_of_threads/set_number_of_threads.cpp`

.. Python*: set_number_of_threads.py
