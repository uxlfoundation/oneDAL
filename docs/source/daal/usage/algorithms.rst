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

.. _algorithms:

Algorithms
==========

All Algorithms classes are derived from the base class ``AlgorithmIface``.
It provides interfaces for computations covering a variety of usage scenarios.
Basic methods that you typically call are ``compute()`` and ``finalizeCompute()``.
In a very generic form algorithms accept one or several numeric tables or models as an input and return one or several numeric tables and models as an output.
Algorithms may also require algorithm-specific parameters that you can modify by accessing the ``parameter`` field of the algorithm.
Because most of the algorithm parameters are preset with default values, you can often omit initialization of the parameter.

Algorithm Input
***************

An algorithm can accept one or several numeric tables or models as an input.
In computation modes that permit multiple calls to the ``compute()`` method,
ensure that the structure of the input data, that is, the number of features, their order, and type, is the same for all the calls.
The following methods are available to provide input to an algorithm:

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input
   :widths: 10 60
   :align: left
   :class: longtable

   * - ``input.set(Input ID, InputData)``
     - Use to set a pointer to the input argument with the ``Input ID`` identifier.
       This method overwrites the previous input pointer stored in the algorithm.
   * - ``input.add(Input ID, InputData)``
     - Use in the distributed computation mode to add the pointers with the ``Input ID`` identifier.
       Unlike the ``input.set()`` method, ``input.add()`` does not overwrite the previously set input pointers,
       but stores all the input pointers until the ``compute()`` method is called.
   * - ``input.get(Input ID)``
     - Use to get a reference to the pointer to the input data with the ``Input ID`` identifier.

For the input that each specific algorithm accepts, refer to the description of this algorithm.

Algorithm Output
****************

Output of an algorithm can be one or several models or numeric tables.
To retrieve the results of the algorithm computation, call the ``getResult()`` method.
To access specific results, use the ``get(Result ID)`` method with the appropriate ``Result ID`` identifier.
In the distributed processing mode, to get access to partial results of the algorithm computation,
call the ``getPartialResult()`` method on each computation node.
For a full list of algorithm computation results available, refer to the description of an appropriate algorithm.

By default, all algorithms allocate required memory to store partial and final results.
Follow these steps to provide user allocated memory for partial or final results to the algorithm:

#. Create an object of an appropriate class for the results. For the classes supported, refer to the description of a specific algorithm.
#. Provide a pointer to that object to the algorithm by calling the ``setPartialResult()`` or ``setResult()`` method as appropriate.
#. Call the ``compute()`` method. After the call, the object created contains partial or final results.

Algorithm Parameters
********************

Most of the algorithms in |short_name| have a set of algorithm-specific parameters.
Because most of the parameters are optional and preset with default values, you can often omit parameter modification.
Provide required parameters to the algorithm using the constructor during algorithm initialization.
If you need to change the parameters, you can do it by accessing the public field parameter of the algorithm.
Some algorithms have an initialization procedure that sets or precomputes specific parameters needed to compute the algorithm.
You can use the InitializationProcedureIface interface class to implement your own initialization procedure
when the default implementation does not meet your specific needs.

Each algorithm also has generic parameters, such as the floating-point type, computation method,
and computation step for the distributed processing mode.

- In C++, these parameters are defined as template parameters, and in most cases they are preset with default values.
  You can change the template parameters while declaring the algorithm.

For a list of algorithm parameters, refer to the description of an appropriate algorithm.
