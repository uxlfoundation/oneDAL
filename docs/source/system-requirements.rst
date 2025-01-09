.. Copyright 2022 Intel Corporation
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

System Requirements
===================

Hardware Requirements
*********************

.. tabs::

   .. tab:: GPU

      - 6th Gen Intel® Core™ processor or higher
      - Intel® Iris® Plus Graphics
      - Intel® Iris® Xe Graphics
      - Intel® Iris® Xe Max Graphics
      - Intel® Iris® Graphics
      - Intel® Iris® Pro Graphics

   .. tab:: CPU

      - Intel Atom® Processors
      - Intel® Core™ Processor Family
      - Intel® Xeon® Processor Family
      - Intel® Xeon® Scalable Performance Processor Family
      - Generic X86 processor

Software Requirements
*********************

- GCC* 7.x or higher or Intel® C++ Compiler 19.1 and later
- Intel® oneAPI DPC++ Compiler latest release (for oneAPI DPC++ interfaces)
- Intel® oneAPI Threading Building Blocks latest release (for the multi-threaded version of oneDAL)
- C/C++ Compiler with C++11 support (or C++14 support on Windows*)
- Microsoft Visual Studio* (2019 and 2022 versions) needed only if using Visual Studio IDE for development

.. rubric:: Operating Systems

|short_name| only supports 64-bit operating systems:

- Linux*: Ubuntu* 18.04 or higher
- Windows* 10 or higher
- Windows Server 2019 or higher
- macOS* 10.14 or higher

To build examples with SYCL API extensions, you also need:

- GNU* Make on Linux*, nmake on Windows*