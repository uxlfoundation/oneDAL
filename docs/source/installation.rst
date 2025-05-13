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

Installation
============

.. |base_tk_link| replace:: |base_tk|
.. _base_tk_link: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html
.. |standalone_link| replace:: |full_name|
.. _standalone_link: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onedal-download.html

There are several options available for installing |short_name|:

- **Binary Distribution**: pre-built binary packages are available from the following sources:

  - Intel® oneAPI:

    - Download as Part of the |base_tk_link|_
    - Download as the Stand-Alone |standalone_link|_
  - Conda: ::

      conda install -c conda-forge dal-devel

  - `NuGet <https://www.nuget.org/packages/inteldal.devel.linux-x64>`__

- **Source Distribution**: Clone the `GitHub repository <https://github.com/uxlfoundation/oneDAL>`__ or `download a specific version <https://github.com/uxlfoundation/oneDAL/releases>`__ of |short_name| from the GitHub releases page and follow the instructions in the `INSTALL.md file <https://github.com/uxlfoundation/oneDAL/blob/main/INSTALL.md>`__.

.. seealso::

   - :ref:`onedal_get_started`
