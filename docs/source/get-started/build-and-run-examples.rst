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

.. |dpcpp_gsg| replace:: Get Started with Intel\ |reg|\  oneAPI DPC++/C++ Compiler
.. _dpcpp_gsg: https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/get-started-guide/current/overview.html

Build and Run Examples
~~~~~~~~~~~~~~~~~~~~~~~

Perform the following steps to build and run examples demonstrating the
basic usage scenarios of |short_name| with DPCPP. Go to
:file:`<install_dir>/dal/<version>` and then set up an environment as shown in the example below:

.. note::

   All content below that starts with ``#`` is considered a comment and
   should not be run with the code.

#. Set up the required environment for |short_name|
   (variables such as ``CPATH``, ``LIBRARY_PATH``, and ``LD_LIBRARY_PATH``):

   .. tabs::

      .. group-tab:: Linux

         On Linux, there are two possible ways to set up the required environment:
         via ``vars.sh`` script or via ``modulefiles``.

         * To set up |short_name| environment via ``vars.sh`` script, run ``source ./env/vars.sh``.
         * To set up |short_name| environment via ``setvars.sh`` script, run ``source ./setvars.sh``.
         * To set up |short_name| environment via ``modulefiles``:

           #. Initialize ``modules``:

              .. code-block:: bash

                 source $MODULESHOME/init/bash

              .. note:: Refer to `Environment Modules documentation <https://modules.readthedocs.io/en/latest/index.html>`_ for details.

           #. Provide ``modules`` with a path to the ``modulefiles`` directory:

              .. code-block:: bash

                 module use ./modulefiles

           #. Run the module:

              .. code-block:: bash

                 module load dal

      .. group-tab:: Windows

         To set up |short_name| environment, run ``source /env/vars.bat`` or ``source setvars.bat``.

#. Copy ``./examples/oneapi/dpc`` to a writable directory if necessary (since it creates temporary files):

   .. code-block:: bash

      cp –r ./examples/oneapi/dpc ${WRITABLE_DIR}

#. Set up the compiler environment for |dpcpp|.
   See |dpcpp_gsg|_ for details.

#. Set up oneMKL environment in case of static linking:

    .. code-block:: bash

      source mkl/latest/env/vars.sh

#. Build and run examples:

   .. note::

      You need to have write permissions to the :file:`examples` folder
      to build examples, and execute permissions to run them.
      Otherwise, you need to copy :file:`examples/oneapi/dpc` and :file:`examples/oneapi/data` folders
      to the directory with right permissions. These two folders must be retained
      in the same directory level relative to each other.

   .. tabs::

      .. group-tab:: Linux

         .. code-block:: bash

           # Navigate to examples directory and build examples
           cd /examples/oneapi/dpc
           cmake -G "Unix Makefiles⁮" -DEXAMPLES_LIST=svm_two_class_thunder # This would generate makefiles for all svm examples matching passed name
           make               # This will compile and run generated svm examples
           cmake -G "Unix Makefiles⁮" -DONEDAL_LINK=static # This wouldgenerate make for static version
           make               # This will compile and run all the examples

      .. group-tab:: Windows

         .. code-block:: bash

            # Navigate to examples directory and build examples
            cd /examples/oneapi/dpc
           cmake  -G "NMake Makefiles" -DEXAMPLES_LIST=svm_two_class_thunder # This would generate makefiles for all svm examples matching passed name
           nmake             # This will compile and run generated svm examples
           cmake  -G "NMake Makefiles" -DONEDAL_LINK=static # This wouldgenerate make for static version
           nmake              # This will compile and run all the examples


#. The resulting example binaries and log files are written into the :file:`_results` directory.

   .. note::

      You should run the examples from :file:`examples/oneapi/dpc` folder, not from :file:`_results` folder.
      Most examples require data to be stored in :file:`examples/oneapi/data` folder and to have a relative link to it
      started from :file:`examples/oneapi/dpc` folder.


   You can build traditional C++ examples located in ``examples/oneapi/cpp`` folder in a similar way.

