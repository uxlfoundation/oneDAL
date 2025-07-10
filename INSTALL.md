<!--
******************************************************************************
* Copyright 2014 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/-->

# Installation from Sources

Required Software:
* C/C++ Compiler
* [DPC++ Compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html) and [oneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) if building with SYCL support
* BLAS and LAPACK libraries - both provided by oneMKL
* Python version 3.9 or higher
* oneTBB library (repository contains script to download it)
* oneDPL library
* Microsoft Visual Studio\* (Windows\* only)
* [MSYS2](http://msys2.github.io) (Windows\* only)
* `make`; which can be installed using MSYS2 on Windows\* as follows:

        pacman -S msys/make

For details, see [System Requirements for oneDAL](https://www.intel.com/content/www/us/en/developer/articles/system-requirements/system-requirements-for-oneapi-data-analytics-library.html).

Note: the Intel(R) oneAPI components listed here can be installed together through the oneAPI Base Toolkit bundle:

https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html

All of these dependencies can alternatively be installed through the `conda` software, but doing so will require a few additional setup steps - see [Conda Development Environment Setup](https://github.com/uxlfoundation/oneDAL/blob/main/INSTALL.md#conda-development-environment-setup) for details.

## Docker Development Environment

[Docker file](https://github.com/uxlfoundation/oneDAL/tree/main/dev/docker) with the oneDAL development environment
is available as an alternative to the manual setup.

## Installation Steps


1. Clone the sources from GitHub\* as follows:

        git clone https://github.com/uxlfoundation/oneDAL.git

2. Set the PATH environment variable to the MSYS2\* bin directory (Windows\* only). For example:

        set PATH=C:\msys64\usr\bin;%PATH%

3. Set the environment variables for one of the supported C/C++ compilers, such as [Intel(R)'s DPC++ compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html). For example:

    - **Microsoft Visual Studio\* 2022**:

            call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" x64

    - **Intel(R) oneAPI DPC++/C++ Compiler 2023.2 (Linux\*)**:

            source /opt/intel/oneapi/compiler/latest/env/vars.sh

    - **Intel(R) oneAPI DPC++/C++ Compiler 2023.2 (Windows\*)**:

            call "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\env\vars.bat"

    Note: if the Intel compilers were installed as part of a bundle such as oneAPI Base Toolkit, it's also possible to set the environment variables at once for all oneAPI components used here (Compilers, oneMKL, oneTBB) through the more general script that they provide - for Linux:

            source /opt/intel/oneapi/setvars.sh

4. Set up MKL:

    _Note: if you used the general oneAPI setvars script from a Base Toolkit installation, this step will not be necessary as oneMKL will already have been set up._

    Download and install [Intel(R) oneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html).
    Set the environment variables for for Intel(R) oneMKL. For example:

    - **Windows\***:

            call "C:\Program Files (x86)\Intel\oneAPI\mkl\latest\env\vars.bat" intel64

    - **Linux\***:

            source /opt/intel/oneapi/mkl/latest/env/vars.sh

5. Set up oneAPI Threading Building Blocks (oneTBB):

    _Note: if you used the general oneAPI setvars script from a Base Toolkit installation, this step will not be necessary as oneTBB will already have been set up._

    Download and install [oneTBB](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onetbb.html).
    Set the environment variables for for oneTBB. For example:

    - oneTBB (Windows\*):

            call "C:\Program Files (x86)\Intel\oneAPI\tbb\latest\env\vars.bat" intel64

    - oneTBB (Linux\*):

            source /opt/intel/oneapi/tbb/latest/env/vars.sh intel64

    Alternatively, you can use scripts to do this for you (Linux\*):

            ./dev/download_tbb.sh

6. Set up oneDPL
  _Note: if you used the general oneAPI setvars script from a Base Toolkit installation, this step will not be necessary as oneDPL will already have been set up._

    Download and install [Intel(R) oneDPL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-library.html).
    Set the environment variables for for Intel(R) oneDPL. For example:

    - oneDPL (Windows\*):

            call "C:\Program Files (x86)\Intel\oneAPI\dpl\latest\env\vars.bat" intel64

    - oneDPL (Linux\*):

            source /opt/intel/oneapi/dpl/latest/env/vars.sh intel64


7. Download and install Python (version 3.9 or higher).

8. Build oneDAL via command-line interface. Choose the appropriate commands based on the interface, platform, and the compiler you use. Interface and platform are required arguments of makefile while others are optional. Below you can find the set of examples for building oneDAL. You may use a combination of them to get the desired build configuration:

    - DAAL interfaces on **Linux\*** using **Intel(R) C++ Compiler**:

            make -f makefile daal PLAT=lnx32e

    - DAAL interfaces on **Linux\*** using **GNU Compiler Collection\***:

            make -f makefile daal PLAT=lnx32e COMPILER=gnu

    - DAAL interfaces on **Linux\*** using **Clang\***:

            make -f makefile daal PLAT=lnx32e COMPILER=clang

    - oneAPI C++/DPC++ interfaces on **Windows\*** using **Intel(R) DPC++ compiler**:

            make -f makefile oneapi PLAT=win32e

    - oneAPI C++ interfaces on **Windows\*** using **Microsoft Visual\* C++ Compiler**:

            make -f makefile oneapi_c PLAT=win32e COMPILER=vc

    - DAAL and oneAPI C++ interfaces on **Linux\*** using **GNU Compiler Collection\***:

            make -f makefile daal oneapi_c PLAT=lnx32e COMPILER=gnu

It is possible to build oneDAL libraries with selected set of algorithms and/or CPU optimizations. `CORE.ALGORITHMS.CUSTOM` and `REQCPUS` makefile defines are used for it.

- To build oneDAL with Linear Regression and Support Vector Machine algorithms, run:

            make -f makefile daal PLAT=win32e CORE.ALGORITHMS.CUSTOM="linear_regression svm" -j16


- To build oneDAL with AVX2 and AVX512 CPU optimizations, run:

            make -f makefile daal PLAT=win32e REQCPU="avx2 avx512" -j16


- To build oneDAL with Moments of Low Order algorithm and AVX2 CPU optimizations, run:

            make -f makefile daal PLAT=win32e CORE.ALGORITHMS.CUSTOM=low_order_moments REQCPU=avx2 -j16

On **Linux\*** it is possible to build debug version of oneDAL or the version that allows to do kernel profiling using <ittnotify.h>.

- To build debug version of oneDAL (including debug symbols and asserts), run:

            make -f makefile daal oneapi_c PLAT=lnx32e REQDBG=yes

- To build oneDAL to include only debug symbols, run:

            make -f makefile daal oneapi_c PLAT=lnx32e REQDBG=symbols

It is possible to integrate various sanitizers by specifying the REQSAN flag, available sanitizers are dependent on the compiler.

- To integrate [AddressSanitizer](https://github.com/google/sanitizers/wiki/addresssanitizer) in a debug oneDAL build (recommended), run:

    _Note: Windows support of REQSAN in oneDAL is experimental, static AddressSanitizer can be set with value: static_

            make -f makefile daal oneapi_c PLAT=lnx32e REQSAN=address REQDBG=yes

- To integrate [MemorySanitizer](https://github.com/google/sanitizers/wiki/memorysanitizer) in a debug oneDAL build, run:

    _Note: Clang and Clang-derived compilers (including the Intel DPC++ compiler) support additional sanitizers such MSan, TSan, and UBSan_

            make -f makefile daal oneapi_c PLAT=lnx32e REQSAN=memory REQDBG=yes
  
- To build oneDAL with gcov code coverage tool integration, run:

    _Note: Only available when building with the Intel DPC++ compiler on Linux operating systems_

            make -f makefile daal oneapi_c PLAT=lnx32e CODE_COVERAGE=yes  

- To build oneDAL with kernel profiling information (`REQPROFILE=yes`):

    _Note: if you used the general oneAPI setvars script from a Base Toolkit installation, those steps will not be necessary as Intel(R) VTune(TM) Profiler will already have been set up._

    - Download and install [Intel(R) VTune(TM) Profiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler-download.html).

    - Set the environment variables for Intel(R) VTune(TM) Profiler. For example:

            source /opt/intel/oneapi/vtune/latest/vtune-vars.sh

    - Run `make` to build oneDAL:

            make -f makefile daal oneapi_c PLAT=lnx32e REQPROFILE=yes

---
**NOTE:** Built libraries are located in the `__release_{os_name}[_{compiler_name}]/daal` directory.

---

After having built the library, if one wishes to use it for building [scikit-learn-intelex](https://github.com/uxlfoundation/scikit-learn-intelex/tree/main) or for executing the usage examples, one can set the required environment variables to point to the generated build by sourcing the script that it creates under the `env` folder. The script will be located under `__release_{os_name}[_{compiler_name}]/daal/latest/env/vars.sh` and can be sourced with a POSIX-compliant shell such as `bash`, by executing something like the following from inside the `__release*` folder:

```shell
cd daal/latest
source env/vars.sh
```

The provided unit tests for the library can be executed through the Bazel system - see the [Bazel docs](https://github.com/uxlfoundation/oneDAL/tree/main/dev/bazel) for more information.

Examples of library usage for both the DAAL and oneAPI interfaces will also be auto-generated as part of the build, under paths `daal/latest/examples/daal/cpp/source` and `daal/latest/examples/oneapi/cpp/source`. These can be built through CMake - assuming one starts from the release path `__release_{os_name}[_{compiler_name}]`, the following would do:

* DAAL examples:

```shell
cd daal/latest/examples/daal/cpp
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

* oneAPI examples:

```shell
cd daal/latest/examples/oneapi/cpp
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

This will generate executables under paths `daal/latest/examples/daal/cpp/_cmake_results/{platform_name}` and `daal/latest/examples/oneapi/cpp/_cmake_results/{platform_name}`. They can be executed as follows (note that they require access to the data files under `daal/latest/examples/daal/data` and `daal/latest/examples/oneapi/data`), **assuming that one starts from inside the `build` folder** (as at the end of the previous steps):

```shell
cd ..
./_cmake_results/{platform_name}/{example}
```

For example, in a Linux platform, assuming one wishes to execute the `adaboost_dense_batch` example:

```shell
./_cmake_results/intel_intel64_so/adaboost_dense_batch
```

DPC++ examples (running on devices supported by SYCL, such as GPU) from oneAPI are also auto-generated within these folders when oneDAL is built with DPC++ support (target `oneapi` in the Makefile), but be aware that it requires a DPC++ compiler such as ICX, and executing the examples requires the DPC++ runtime as well as the GPGPU drivers. The DPC++ examples can be found under `examples/oneapi/dpc`.

### Executing examples with ASAN

When building oneDAL with ASAN (flags `REQSAN=address`, typically combined with `REQDBG=yes`), building and executing the generated examples requires additional steps - **assuming a Linux system** (ASAN on Windows has not been tested):

* Configure CMake to build the examples with the same compiler as was one for oneDAL (ICX by default) and with dynamic linkage to ASAN - e.g. by setting these flags:
    ```shell
    export CC=icx
    export CXX=icpx
    export CXXFLAGS="-fsanitize=address"
    export LDFLAGS="-shared-libasan"
    ```
* Create a symlink to the ASAN runtime in the same folder from where the examples are executed. ICX uses the same ASAN runtime as CLANG, so something like this should do:
    ```shell
    ln -s $(clang -print-file-name=libclang_rt.asan-x86_64.so) libclang_rt.asan.so
    ```
* Use the verbose mode in oneDAL when executing examples (otherwise ASAN won't produce prints):
    ```shell
    export ONEDAL_VERBOSE=1
    ```

Putting it all together, the earlier snippets for executing the examples but with ASAN enabled should look like this:

```shell
cd daal/latest/examples/daal/cpp
mkdir -p build
cd build
CC=icx CXX=icpx CXXFLAGS="-fsanitize=address" LDFLAGS="-shared-libasan" cmake ..
make -j$(nproc)
ln -s $(clang -print-file-name=libclang_rt.asan-x86_64.so) libclang_rt.asan.so
ONEDAL_VERBOSE=1 ./_cmake_results/intel_intel64_so/adaboost_dense_batch
```

_Be aware that ASAN is known to generate many false-positive reports of memory leaks when used with oneDAL._

## Conda Development Environment Setup

The previous instructions assumed system-wide installs of the necessary dependencies. On Linux*, these can also be installed at a user-level through the `conda` or [mamba](https://github.com/conda-forge/miniforge) ecosystems.

First, create a conda environment for building oneDAL, after `conda` has been installed:

```shell
conda create -y -n onedal_env
conda activate onedal_env
```

Then, install the necessary dependencies from the appropriate channels with `conda`:

```shell
conda install -y \
    -c https://software.repos.intel.com/python/conda/ `# Intel's repository` \
    -c conda-forge `# for tools like 'make'` \
    make "python>=3.9" `# used by the build system` \
    dpcpp-cpp-rt dpcpp_linux-64 intel-sycl-rt `# Intel compiler packages` \
    tbb tbb-devel `# required TBB packages` \
    onedpl-devel `# required oneDPL package` \
    mkl mkl-devel mkl-static mkl-dpcpp mkl-devel-dpcpp `# required MKL packages` \
    cmake `# required to build the examples only`
```

Then modify the relevant environment variables to point to the conda-installed libraries:

```shell
export MKLROOT="${CONDA_PREFIX}"
export TBBROOT="${CONDA_PREFIX}"
export DPL_ROOT="${CONDA_PREFIX}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export LIBRARY_PATH="${CONDA_PREFIX}/lib:${LIBRARY_PATH}"
export CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/include:${CPLUS_INCLUDE_PATH}"
export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH}"
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}/lib/cmake:${CMAKE_PREFIX_PATH}"
```

_Note: variable `$PATH` is also required to contain `${CONDA_PREFIX}/bin`, but that should have been handled automatically by `conda activate`._

After that, it should be possible to build oneDAL and run the examples using the ICX compiler and the oneMKL libraries as per the instructions.

For other setups in **Linux\***, such as building for platforms like `aarch64` that are not supported by Intel's toolkits or using non-default options offered by the Makefile, other software can be installed as follows:

* GCC compilers (option `COMPILER=gnu`):

```shell
conda install -y -c conda-forge \
    gcc gxx c-compiler cxx-compiler
```

(no environment variables are needed for `COMPILER=gnu`)

* Reference (non-tuned) computational backends, and BLAS/LAPACK backends from OpenBLAS (both through option `BACKEND_CONFIG=ref`):

```shell
conda install -y -c conda-forge \
    blas=*=*openblas* openblas
```

* Optionally, if one wishes to install the OpenMP variant of OpenBLAS instead of the pthreads one, or to use the ILP64 variant:
```shell
conda install -y -c conda-forge \
    blas=*=*openblas* openblas-ilp64=*=*openmp*
```

Then set environment variables as needed:
```shell
export OPENBLASROOT=${CONDA_PREFIX}
```

(note that other variables such as `TBBROOT` and `CMAKE_PREFIX_PATH` are still required)
