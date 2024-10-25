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

.. |dpcpp_comp| replace:: Intel\ |reg|\  oneAPI DPC++/C++ Compiler
.. _dpcpp_comp: https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html

.. _before_you_begin:

Before You Begin
~~~~~~~~~~~~~~~~

|short_name| is located in :file:`<install_dir>/dal` directory where :file:`<install_dir>`
is the directory in which |base_tk| was installed.

The current version of |short_name| with
SYCL is available for Linux\* and Windows\* 64-bit operating systems. The
prebuilt |short_name| libraries can be found in the :file:`<install_dir>/dal/<version>/redist`
directory.

The dependencies needed to build examples with SYCL extensions:

- |dpcpp_comp|_ 2021.1 release or later (for support)
- OpenCL™ runtime 1.2 or later (to run the SYCL runtime)
- GNU\* Make on Linux\*, nmake on Windows\*
