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

.. |32e_make| replace:: 32e.mk
.. _32e_make: https://github.com/oneapi-src/oneDAL/blob/main/dev/make/function_definitions/32e.mk
.. |riscv_make| replace:: riscv64.mk
.. _riscv_make: https://github.com/oneapi-src/oneDAL/blob/main/dev/make/function_definitions/riscv64.mk
.. |arm_make| replace:: arm.mk
.. _arm_make: https://github.com/oneapi-src/oneDAL/blob/main/dev/make/function_definitions/arm.mk

.. highlight:: cpp

CPU Features Dispatching
^^^^^^^^^^^^^^^^^^^^^^^^

For each algorithm |short_name| provides several code paths for x86-64-compatible architectural extensions.

Following extensions are currently supported:

- Intel\ |reg|\  Streaming SIMD Extensions 2 (Intel\ |reg|\  SSE2)
- Intel\ |reg|\  Streaming SIMD Extensions 4.2 (Intel\ |reg|\  SSE4.2)
- Intel\ |reg|\  Advanced Vector Extensions 2 (Intel\ |reg|\  AVX2)
- Intel\ |reg|\  Advanced Vector Extensions 512 (Intel\ |reg|\  AVX-512)

The particular code path is chosen at runtime based on underlying hardware properties.

This chapter describes how the code is organized to support this variety of extensions.

Algorithm Implementation Options
********************************

In addition to the architectural extensions, an algorithm in |short_name| may have various
implementation options. Below is a description of these options to help you better understand
the |short_name| code structure and conventions.

Computational Tasks
-------------------

An algorithm might have various tasks to compute. The most common options are:

- `Classification <https://oneapi-src.github.io/oneDAL/onedal/glossary.html#term-Classification>`_,
- `Regression <https://oneapi-src.github.io/oneDAL/onedal/glossary.html#term-Regression>`_.

Computational Stages
--------------------

An algorithm might have ``training`` and ``inference`` computation stages aimed
at training a model on the input dataset and computing the inference results, respectively.

Computational Methods
---------------------

An algorithm can support several methods for the same type of computations.
For example, kNN algorithm supports
`brute_force <https://oneapi-src.github.io/oneDAL/onedal/algorithms/nearest-neighbors/knn.html#knn-t-math-brute-force>`_
and `kd_tree <https://oneapi-src.github.io/oneDAL/onedal/algorithms/nearest-neighbors/knn.html#knn-t-math-kd-tree>`_
methods for algorithm training and inference.

Computational Modes
-------------------

|short_name| can provide several computational modes for an algorithm.
See `Computational Modes <https://oneapi-src.github.io/oneDAL/onedal/programming-model/computational-modes.html>`_
chapter for details.

Folders and Files
*****************

Suppose that you are working on some algorithm ``Abc`` in |short_name|.

The part of the implementation of this algorithms that is running on CPU should be located in
`cpp/daal/src/algorithms/abc` folder.

Suppose that it provides:

- ``classification`` and ``regression`` learning tasks;
- ``training`` and ``inference`` stages;
- ``method1`` and ``method2`` for the ``training`` stage and only ``method1`` for ``inference`` stage;
- only ``batch`` computational mode.

Then the `cpp/daal/src/algorithms/abc` folder should contain at least the following files:

::

  cpp/daal/src/algorithms/abc/
    |-- abc_classification_predict_method1_batch_fpt_cpu.cpp
    |-- abc_classification_predict_method1_impl.i
    |-- abc_classification_predict_kernel.h
    |-- abc_classification_train_method1_batch_fpt_cpu.cpp
    |-- abc_classification_train_method2_batch_fpt_cpu.cpp
    |-- abc_classification_train_method1_impl.i
    |-- abc_classification_train_method2_impl.i
    |-- abc_classification_train_kernel.h
    |-- abc_regression_predict_method1_batch_fpt_cpu.cpp
    |-- abc_regression_predict_method1_batch_fpt_cpu.cpp
    |-- abc_regression_predict_method1_impl.i
    |-- abc_regression_predict_kernel.h
    |-- abc_regression_train_method1_batch_fpt_cpu.cpp
    |-- abc_regression_train_method2_batch_fpt_cpu.cpp
    |-- abc_regression_train_method1_impl.i
    |-- abc_regression_train_method2_impl.i
    |-- abc_regression_train_kernel.h

Alternative variant of the folder structure to avoid storing too many files within a single folder
could be:

::

  cpp/daal/src/algorithms/abc/
    |-- classification/
    |     |-- abc_classification_predict_method1_batch_fpt_cpu.cpp
    |     |-- abc_classification_predict_method1_impl.i
    |     |-- abc_classification_predict_kernel.h
    |     |-- abc_classification_train_method1_batch_fpt_cpu.cpp
    |     |-- abc_classification_train_method2_batch_fpt_cpu.cpp
    |     |-- abc_classification_train_method1_impl.i
    |     |-- abc_classification_train_method2_impl.i
    |     |-- abc_classification_train_kernel.h
    |-- regression/
          |-- abc_regression_predict_method1_batch_fpt_cpu.cpp
          |-- abc_regression_predict_method1_impl.i
          |-- abc_regression_predict_kernel.h
          |-- abc_regression_train_method1_batch_fpt_cpu.cpp
          |-- abc_regression_train_method2_batch_fpt_cpu.cpp
          |-- abc_regression_train_method1_impl.i
          |-- abc_regression_train_method2_impl.i
          |-- abc_regression_train_kernel.h

The names of the files stay the same in this case, just the folder layout differs.

The folders of the algorithms that are already implemented can contain additional files.
For example, files with ``container.h``, ``dispatcher.cpp`` suffixes, etc.
These files are used in the Data Analytics Acceleration Library (DAAL) interface implementation.
That interface is still available to users, but it is not recommended for use in new code.
The files related to the DAAL interface are not described here as they are not part of the CPU features
dispatching mechanism.

Further the purpose and contents of each file are to be described on the example of classification
training task. For other types of the tasks the structure of the code is similar.

\*_kernel.h
-----------

In the directory structure of the ``Abc`` algorithm, there are files with a `_kernel.h` suffix.
These files contain the definitions of one or several template classes that define member functions that
do the actual computations. Here is a variant of the ``Abc`` training algorithm kernel definition in the file
`abc_classification_train_kernel.h`:

.. include:: ../includes/cpu_features/abc-classification-train-kernel.rst

Typical template parameters are:

- ``algorithmFPType``  Data type to use in intermediate computations for the algorithm,
                       ``float`` or ``double``.
- ``method`` Computational methods of the algorithm. ``method1`` or ``method2`` in the case of ``Abc``.
- ``cpu`` Version of the cpu-specific implementation of the algorithm, ``daal::CpuType``.

Implementations for different methods are usually defined using partial class templates specialization.

\*_impl.i
---------

In the directory structure of the ``Abc`` algorithm, there are files with a `_impl.i` suffix.
These files contain the implementations of the computational functions defined in the files with a `_kernel.h` suffix.
Here is a variant of ``method1`` implementation for ``Abc`` training algorithm that does not contain any
instruction set specific code. The implementation is located in the file `abc_classification_train_method1_impl.i`:

.. include:: ../includes/cpu_features/abc-classification-train-method1-impl.rst

Although the implementation of the ``method1`` does not contain any instruction set specific code, it is
expected that the developers leverage SIMD related macros available in |short_name|.
For example, ``PRAGMA_IVDEP``, ``PRAGMA_VECTOR_ALWAYS``, ``PRAGMA_VECTOR_ALIGNED`` and other pragmas defined in
`service_defines.h <https://github.com/oneapi-src/oneDAL/blob/main/cpp/daal/src/services/service_defines.h>`_.
This will guide the compiler to generate more efficient code for the target architecture.

Consider that the implementation of the ``method2`` for the same algorithm will be different and will contain
AVX-512-specific code located in ``cpuSpecificCode`` function. Note that all the compiler-specific code
should be gated by values of compiler-specific defines.
For example, the Intel\ |reg|\  oneAPI DPC++/C++ Compiler specific code should be gated the existence of the
``DAAL_INTEL_CPP_COMPILER`` define. All the CPU-specific code should be gated on the value of CPU-specific define.
For example, the AVX-512 specific code should be gated on the value ``__CPUID__(DAAL_CPU) == __avx512__``.

Then the implementation of the ``method2`` in the file `abc_classification_train_method2_impl.i` will look like:

.. include:: ../includes/cpu_features/abc-classification-train-method2-impl.rst

\*_fpt_cpu.cpp
--------------

In the directory structure of the ``Abc`` algorithm, there are files with a `_fpt_cpu.cpp` suffix.
These files contain the instantiations of the template classes defined in the files with a `_kernel.h` suffix.
The instantiation of the ``Abc`` training algorithm kernel for ``method1`` is located in the file
`abc_classification_train_method1_batch_fpt_cpu.cpp`:

.. include:: ../includes/cpu_features/abc-classification-train-method1-fpt-cpu.rst

`_fpt_cpu.cpp` files are not compiled directly into object files. First, multiple copies of those files
are made replacing the ``fpt``, which stands for 'floating point type', and ``cpu`` parts of the file name
as well as the corresponding ``DAAL_FPTYPE`` and ``DAAL_CPU`` macros with the actual data type and CPU type values.
Then the resulting files are compiled with appropriate CPU-specific compiler optimization options.

The values for ``fpt`` file name part replacement are:

- ``flt`` for ``float`` data type, and
- ``dbl`` for ``double`` data type.

The values for ``DAAL_FPTYPE`` macro replacement are ``float`` and ``double``, respectively.

The values for ``cpu`` file name part replacement are:

- ``nrh`` for Intel\ |reg|\  SSE2 architecture, which stands for Northwood,
- ``neh`` for Intel\ |reg|\  SSE4.2 architecture, which stands for Nehalem,
- ``hsw`` for Intel\ |reg|\  AVX2 architecture, which stands for Haswell,
- ``skx`` for Intel\ |reg|\  AVX-512 architecture, which stands for Skylake-X.

The values for ``DAAL_CPU`` macro replacement are:

- ``__sse2__`` for Intel\ |reg|\  SSE2 architecture,
- ``__sse42__`` for Intel\ |reg|\  SSE4.2 architecture,
- ``__avx2__`` for Intel\ |reg|\  AVX2 architecture,
- ``__avx512__`` for Intel\ |reg|\  AVX-512 architecture.

Build System Configuration
**************************

This chapter describes which parts of the build system need to be modified to add new architectural
extension or to remove an outdated one.

Makefile
--------

The most important definitions and functions for CPU features dispatching are located in the files
|32e_make|_ for x86-64 architecture, |riscv_make|_ for RISC-V 64-bit architecture, and |arm_make|_
for ARM architecture.
Those files are included into operating system related makefiles.
For example, the |32e_make| file is included into ``lnx32e.mk`` file:

::

  include dev/make/function_definitions/32e.mk

And ``lnx32e.mk`` and similar files are included into the main Makefile:

::

  include dev/make/function_definitions/$(PLAT).mk

Where ``$(PLAT)`` is the platform name, for example, ``lnx32e``, ``win32e``, ``lnxriscv64``, etc.

To add a new architectural extension into |32e_make| file, ``CPUs`` and ``CPUs.files`` lists need to be updated.
The functions like ``set_uarch_options_for_compiler`` and others should also be updated accordingly.

The compiler options for the new architectural extension should be added to the respective file in the
`compiler_definitions <https://github.com/oneapi-src/oneDAL/tree/main/dev/make/compiler_definitions>`_ folder.

For example, `gnu.32e.mk <https://github.com/oneapi-src/oneDAL/blob/main/dev/make/compiler_definitions/gnu.32e.mk>`_
file contains the compiler options for the GNU compiler for x86-64 architecture in the form
``option_name.compiler_name``:

::

  p4_OPT.gnu   = $(-Q)march=nocona
  mc3_OPT.gnu  = $(-Q)march=corei7
  avx2_OPT.gnu = $(-Q)march=haswell
  skx_OPT.gnu  = $(-Q)march=skylake

Bazel
-----

For now, Bazel build is supported only for Linux x86-64 platform
It provides ``cpu`` `option <https://github.com/oneapi-src/oneDAL/tree/main/dev/bazel#bazel-options>`_
that allows to specify the list of target architectural extensions.

To add a new architectural extension into Bazel configuration, following steps should be done:

- Add the new extension to the list of allowed values in the ``_ISA_EXTENSIONS`` variable in the
  `config.bzl <https://github.com/oneapi-src/oneDAL/blob/main/dev/bazel/config/config.bzl>`_ file;
- Update the ``get_cpu_flags`` function in the
  `flags.bzl <https://github.com/oneapi-src/oneDAL/blob/main/dev/bazel/flags.bzl>`_
  file to provide the compiler flags for the new extension;
- Update the ``cpu_defines`` dictionaries in
  `dal.bzl <https://github.com/oneapi-src/oneDAL/blob/main/dev/bazel/dal.bzl>`_ and
  `daal.bzl <https://github.com/oneapi-src/oneDAL/blob/main/dev/bazel/daal.bzl>`_ files accordingly.