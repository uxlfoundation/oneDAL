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
* TBB library (repository contains script to download it)
* Microsoft Visual Studio\* (Windows\* only)
* [MSYS2](http://msys2.github.io) (Windows\* only)
* `make` and `dos2unix` tools; install these packages using MSYS2 on Windows\* as follows:

        pacman -S msys/make msys/dos2unix

For details, see [System Requirements for oneDAL](https://www.intel.com/content/www/us/en/developer/articles/system-requirements/system-requirements-for-oneapi-data-analytics-library.html).

Note: the Intel(R) oneAPI components listed here can be installed together through the oneAPI Base Toolkit bundle:

https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html

## Docker Development Environment

[Docker file](https://github.com/uxlfoundation/oneDAL/tree/main/dev/docker) with the oneDAL development environment
is available as an alternative to the manual setup.

## Installation Steps
1. Clone the sources from GitHub\* as follows:

        git clone https://github.com/uxlfoundation/oneDAL.git

2. Set the PATH environment variable to the MSYS2\* bin directory (Windows\* only). For example:

        set PATH=C:\msys64\usr\bin;%PATH%

3. Set the environment variables for one of the supported C/C++ compilers, such as [Intel(R)'s DPC compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html). For example:

    - **Microsoft Visual Studio\* 2022**:

            call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" x64

    - **Intel(R) C++ Compiler 19.1 (Windows\*)**:

            call "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\bin\compilervars.bat" intel64

    - **Intel(R) C++ Compiler 19.1 (Linux\*)**:

            source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64

    - **Intel(R) oneAPI DPC++/C++ Compiler 2023.2 (Linux\*)**:

            source /opt/intel/oneapi/compiler/latest/env/vars.sh

    - **Intel(R) oneAPI DPC++/C++ Compiler 2023.2 (Windows\*)**:

            call "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\env\vars.bat"

    Note: if the Intel compilers were installed as part of a bundle such as oneAPI Base Toolkit, it's also possible to set the environment variables at once for all oneAPI components used here (compilers, MKL, oneMKL, TBB) through the more general script that they provide - for Linux:

            source /opt/intel/oneapi/setvars.sh

4. Set up MKL:

    _Note: if you used the general oneAPI setvars script from a Base Toolkit installation, this step will not be necessary as oneMKL will already have been set up._

    Download and install [Intel(R) oneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html).
    Set the environment variables for for Intel(R) oneMKL. For example:

    - **Windows\***:

            call "C:\Program Files (x86)\Intel\oneAPI\mkl\latest\env\vars.bat" intel64

    - **Linux\***:

            source /opt/intel/oneapi/mkl/latest/env/vars.sh

5. Set up Intel(R) Threading Building Blocks (Intel(R) TBB):

    _Note: if you used the general oneAPI setvars script from a Base Toolkit installation, this step will not be necessary as oneTBB will already have been set up._

    Download and install [Intel(R) TBB](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onetbb.html).
    Set the environment variables for for Intel(R) TBB. For example:

    - oneTBB (Windows\*):

            call "C:\Program Files (x86)\Intel\oneAPI\tbb\latest\env\vars.bat" intel64

    - oneTBB (Linux\*):

            source /opt/intel/oneapi/tbb/latest/env/vars.sh intel64

    Alternatively, you can use scripts to do this for you (Linux\*):

            ./dev/download_tbb.sh

6. Download and install Python (version 3.9 or higher).

7. Build oneDAL via command-line interface. Choose the appropriate commands based on the interface, platform, and the compiler you use. Interface and platform are required arguments of makefile while others are optional. Below you can find the set of examples for building oneDAL. You may use a combination of them to get the desired build configuration:

    - DAAL interfaces on **Linux\*** using **Intel(R) C++ Compiler**:

            make -f makefile daal PLAT=lnx32e

    - DAAL interfaces on **Linux\*** using **GNU Compiler Collection\***:

            make -f makefile daal PLAT=lnx32e COMPILER=gnu

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

- To build debug version of oneDAL, run:

            make -f makefile daal oneapi_c PLAT=lnx32e REQDBG=yes

- To build oneDAL with kernel profiling information, run:

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

Examples of library usage will also be auto-generated as part of the build under path `daal/latest/examples/daal/cpp/source`. These can be built through CMake - assuming one starts from the release path `__release_{os_name}[_{compiler_name}]`, the following would do:

```shell
cd daal/latest/examples/daal/cpp
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

This will generate executables under path `daal/latest/examples/daal/cpp/_cmake_results/{platform_name}`. They can be executed as follows (note that they require access to the data files under `daal/latest/examples/daal/data`), assuming that one starts from inside the `build` folder (as at the end of the previous step):

```shell
cd ..
./_cmake_results/{platform_name}/{example}
```

For example, in a Linux platform, assuming one wishes to execute the `adaboost_dense_batch` example:

```shell
./_cmake_results/intel_intel64_so/adaboost_dense_batch
```
