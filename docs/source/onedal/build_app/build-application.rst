.. Copyright 2014 Intel Corporation
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

Build applications with oneDAL
==============================

This section contains instructions for building applications with |short_name| for SYCL\*.

- :ref:`app_on_lin`
- :ref:`app_on_win`

.. _app_on_lin:

Applications on Linux* OS
-------------------------

#. Install |short_name|.

#. Set environment variables by calling ``<install dir>/vars.sh``.

#. Build the application using ``icpx`` (Linux* OS) and ``icx-cl`` (Windows* OS) commands:

   - Add |short_name| ``includes`` folder:

     .. code-block:: text

        -I<install dir>/dal/latest/include

   - Add |short_name| libraries. Choose the appropriate |short_name| libraries based on |short_name| linking method:

     .. tabularcolumns::  |\Y{0.2}|\Y{0.4}|\Y{0.4}|

     .. list-table:: |short_name| libraries for Linux
          :widths: 15 25
          :header-rows: 1
          :align: left
          :class: longtable

          * - Static linking
            -
              | libonedal_core.a,
              | libonedal_dpc.a,
              | libonedal_parameters.a,
              | libonedal_parameters_dpc.a,
              | libonedal_thread.a
          * - Dynamic linking
            -
              | libonedal_core.so,
              | libonedal_dpc.so,
              | libonedal_parameters.so,
              | libonedal_parameters_dpc.so,
              | libonedal_thread.so

   - Add an additional |short_name| library:

     .. code-block:: text

        <install dir>/mkl/latest/lib/libmkl_sycl.a

.. _app_on_win:


Applications on Windows* OS
---------------------------

#. Install |short_name|.

#. In Microsoft Visual Studio* Integrated Development Environment (IDE),
   open or create a C++ project for your |short_name| application to build.

#. In project properties:

   - Set |dpcpp| platform toolset:

     .. figure:: /onedal/build_app/images/MSVSPlatformToolset.jpg
       :width: 600
       :align: center
       :alt: In General configuration properties, choose Platform Toolset property

   - Add |short_name| ``includes`` folder to :guilabel:`Additional Include Directories`.
   - Add folders with |short_name| and oneTBB libraries to :guilabel:`Library Directories`:

     .. figure:: /onedal/build_app/images/LibraryDirectories.jpg
       :width: 600
       :align: center
       :alt: In VC++ Directories, choose Library Directories property

   - Add |short_name| and OpenCL libraries to :guilabel:`Additional Dependencies`:

     .. figure:: /onedal/build_app/images/AdditionalDependencies.jpg
       :width: 600
       :align: center
       :alt: In Linker configuration properties, choose Input.

#. Add the appropriate libraries to your project based on |short_name| linking method:

   .. tabularcolumns::  |\Y{0.2}|\Y{0.4}|\Y{0.4}|

   .. list-table:: |short_name| libraries for Windows
      :widths: 15 25
      :header-rows: 1
      :align: left
      :class: longtable

      * - Static linking
        -
          | onedal_core.lib,
          | onedal_thread.lib
      * - Dynamic linking
        -
          | onedal_core_dll.lib

   You may also add debug versions of the libraries based on the threading mode and linking method:

   .. tabularcolumns::  |\Y{0.2}|\Y{0.4}|\Y{0.4}|

   .. list-table:: |short_name| debug libraries for Windows
      :widths: 15 25
      :header-rows: 1
      :align: left
      :class: longtable

      * - Static linking
        -
          | onedal_cored.lib,
          | onedald.lib,
          | onedal_dpcd.lib,
          | onedal_sycld.lib,
          | onedal_threadd.lib
      * - Dynamic linking
        -
          | onedal_cored_dll.lib (onedal_cored_dll.3.lib),
          | onedald_dll.lib (onedald_dll.3.lib),
          | onedal_dpcd_dll.lib (onedal_dpcd_dll.3.lib),
          | onedald.3.dll,
          | onedal_cored.3.dll,
          | onedal_dpcd.3.dll,
          | onedal_threadd.3.dll

Examples
********

Dynamic linking |short_name| with the GPU target:

* Linux* OS:

  .. code-block:: text

     icpx -fsycl my_first_dal_program.cpp -Wl,
     --start-group -L<install dir>/dal/latest/lib -lonedal_core -lonedal_dpc -lonedal_parameters_dpc -lonedal_thread -lpthread -ldl -lOpenCL -L<install dir>/tbb/latest/lib/ -ltbb -ltbbmalloc <install dir>/dal/latest/lib/libonedal_sycl.a -Wl,--end-group

* Windows* OS:

  .. code-block:: text

     icx-cl -fsycl my_first_dal_program.cpp -Wl,
     --start-group -L<install dir>/dal/latest/lib/ -lonedal_core -lonedal_dpc -lonedal_thread -lpthread -ldl -lOpenCL -L<install dir>/tbb/latest/lib/ -ltbb -ltbbmalloc <install dir>/dal/latest/lib/libonedal_sycl.a -Wl,--end-group
