.. Copyright contributors to the oneDAL project
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

Using oneDAL Verbose Mode
=========================

When building applications that call Intel® oneAPI Data Analytics Library functions, it may be useful to determine:

- Which computational kernels are called
- What parameters are passed to them
- How much time is spent to execute the functions
- Which device (CPU/GPU) the kernel is executed on

You can get an application to print this information to a standard output
device by enabling **Intel® oneAPI Data Analytics Library Verbose Mode**.

When Verbose mode is active in oneDAL, every call of a verbose-enabled function finishes with 
printing a human-readable line describing the call. Even if the application gets terminated during 
the function call, information for that function will be printed. 
The first call to a verbose-enabled function also prints a version information line.

For GPU applications, additional information (one or more GPU information lines) will also 
be printed on the first call to a verbose-enabled function, following the version information lines printed
for the host CPU.

Verbosity Levels
----------------

We have common implementations for verbose mode with CPU and GPU applications.

  - **Levels**:
  - Disabled (default)
  - Logger enabled
  - Tracer enabled
  - Analyzer enabled
  - All tools enabled
  - All tools and service functions enabled

Setting Verbose Mode
--------------------

You can change the verbose mode either by setting the environment variable ``ONEDAL_VERBOSE``.

**Using environment variable:**

+--------------------------+-----------------------------------------------+
| Set ``ONEDAL_VERBOSE=0`` | Disabled (default)                            |
+--------------------------+-----------------------------------------------+
| Set ``ONEDAL_VERBOSE=1`` | Logger enabled                                |
+--------------------------+-----------------------------------------------+
| Set ``ONEDAL_VERBOSE=2`` | Tracer enabled                                |
+--------------------------+-----------------------------------------------+
| Set ``ONEDAL_VERBOSE=3`` | Analyzer enabled                              |
+--------------------------+-----------------------------------------------+
| Set ``ONEDAL_VERBOSE=4`` | All tools enabled (Logger + Tracer + Analyzer)|
+--------------------------+-----------------------------------------------+
| Set ``ONEDAL_VERBOSE=5`` | All tools and service functions enabled       |
+--------------------------+-----------------------------------------------+


When enabled with timing, execution timing is taken synchronously, meaning previous kernel execution will block later kernels.

General Notes on Verbose Mode
-----------------------------

