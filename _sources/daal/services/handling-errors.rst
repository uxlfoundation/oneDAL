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

Handling Errors
===============

|short_name| provides classes and methods to handle exceptions or
errors that can occur during library operation.

The methods of the library return the following computation set
status:

-  Success - no errors detected
-  Warning - recoverable errors detected
-  Failure - unrecoverable errors detected

In |short_name| C++ interfaces, the base class for error handling is
Status. If the execution of the library methods provided by the
Algorithm or Data Management classes is unsuccessful, the Status
object returned by the respective routines contains the list of
errors and/or warnings extended with additional details about the
error conditions. The class includes the list of the following
methods for error processing:

-  ``ok()`` - checks whether the Status object contains any unrecoverable errors.

-  ``add()`` - adds information about the error, such as the error identifier or the pointer to the error.

-  ``getDescription()`` - returns the detailed description of the errors contained in the object.

-  ``clear()`` - removes information about the errors from the object.

The error class in |short_name| C++ interfaces is Error. This class
contains an error message and details of the issue. For example,
an Error object can store the number of the row in the
NumericTable that caused the issue or a message that an SQL
database generated to describe the reasons of an unsuccessful
query. A single Error object can store the error description and
an arbitrary number of details of various types: integer or double
values or strings.

The class includes the list of the following methods for error
processing:

-  ``id()`` - returns the identifier of the error.

-  ``setId()`` - sets the identifier of the error.

-  ``description()`` - returns the detailed description of the error.

-  ``add[Int|Double|String]Detail()`` adds data type-based details to the error.

-  ``create()`` - creates an instance of the Error class with the given set of arguments.

By default, the ``compute()`` method of the library algorithms throws
run-time exception when error is detected. To prevent throwing any
exceptions, call the ``computeNoThrow()`` method.

Service methods of the algorithms, such as ``setResult()`` and
``setPartialResult()``, do not throw exceptions and return the status
of the respective operation.

The methods of the Data Management classes do not throw exceptions
and return the status of the respective operation.


Examples
++++++++

C++:

-  :cpp_example:`error_handling/error_handling_nothrow.cpp`
-  :cpp_example:`error_handling/error_handling_throw.cpp`

.. Python*: error_handling_throw.py

